"""Packed-sequence encoder: token, segment, and slot embeddings plus blocks."""

from __future__ import annotations

import torch
from torch import nn

from src.modeling.autoencoder.config import AttentionAutoencoderConfig
from src.modeling.checkpointing import autocast_for, run_checkpointed
from src.modeling.shared.dca import DCAStackState, DCALayer, GRNv3
from src.modeling.shared.embeddings import slot_indices
from src.modeling.shared.layers import VarlenTransformerLayer
from src.modeling.varlen import cu_seqlens, flatten_valid, scatter_valid


class PackedSegmentEmbedding(nn.Module):
    """Embed a flat packed stream of segments.

    Unlike :class:`~src.modeling.shared.embeddings.SegmentEmbedding`, this
    works on the packed ``[batch, seq]`` layout and reserves one extra slot id
    for CLS/SEP/padding.
    """

    def __init__(self, config: AttentionAutoencoderConfig) -> None:
        super().__init__()
        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.segment_embedding = nn.Embedding(config.max_segments + 1, config.d_model)
        self.slot_embedding = nn.Embedding(config.segment_len + 1, config.d_model)
        return

    def forward(self, tokens: torch.Tensor, segment_ids: torch.Tensor) -> torch.Tensor:
        slots = slot_indices(tokens, segment_ids)
        return (
            self.token_embedding(tokens)
            + self.segment_embedding(segment_ids)
            + self.slot_embedding(slots)
        )


class SequenceEncoder(nn.Module):
    """Self-attention stack over every valid token in the packed window.

    Attention crosses segment boundaries, letting the window bottleneck see
    how positions relate along the game instead of encoding boards in
    isolation. Only valid tokens enter the blocks (varlen flash path); the
    padded tail is scattered back afterwards.

    With ``config.dca`` the blocks compose their q/k/v inputs from the
    encoder's own residual stack. The stack never leaves this module: the
    bottleneck pools the final stream and destroys it.
    """

    def __init__(self, config: AttentionAutoencoderConfig) -> None:
        super().__init__()
        self.config = config
        self.embedding = PackedSegmentEmbedding(config)
        self.dca = config.routes("encoder")
        if self.dca:
            self.blocks = nn.ModuleList(
                DCALayer(
                    config.d_model,
                    config.n_heads,
                    config.d_ff,
                    config.dropout,
                    variant=config.dca_variant,
                )
                for _ in range(config.n_layers)
            )
            outputs = 3 if config.dca_variant == "dca" else 1
            self.grn = nn.ModuleList(
                GRNv3(config.d_model, config.dca_slots, outputs=outputs)
                for _ in range(config.n_layers)
            )
            self.past_k = config.dca_past_layers_k
        else:
            self.blocks = nn.ModuleList(
                VarlenTransformerLayer(
                    config.d_model, config.n_heads, config.d_ff, config.dropout
                )
                for _ in range(config.n_layers)
            )
        self.ln_final = nn.LayerNorm(config.d_model)
        return

    def _run_blocks(self, flat: torch.Tensor, cu, max_seqlen: int) -> torch.Tensor:
        with autocast_for(flat.device.type):
            return self._block_loop(flat, cu, max_seqlen)

    def _block_loop(self, flat: torch.Tensor, cu, max_seqlen: int) -> torch.Tensor:
        if not self.dca:
            for block in self.blocks:
                flat = block(flat, cu, max_seqlen)
            return flat
        stream = flat
        stack = DCAStackState(flat, self.past_k, self.config.dca_slots)
        for grn, block in zip(self.grn, self.blocks):
            composed = grn(stack.stack())
            if len(composed) == 3:
                q_in, k_in, v_in = composed
            else:
                q_in = k_in = v_in = composed[0]
            increment = block(stream, q_in, k_in, v_in, cu, max_seqlen)
            stream = stream + increment
            stack.update(increment)
        return stream

    def forward(
        self,
        tokens: torch.Tensor,
        segment_ids: torch.Tensor,
        padding_mask: torch.Tensor,
    ) -> torch.Tensor:
        valid = padding_mask.bool()
        x = self.embedding(tokens, segment_ids)
        cu, max_seqlen = cu_seqlens(valid)
        flat = flatten_valid(x, valid)
        flat = run_checkpointed(
            self._run_blocks,
            flat,
            cu,
            max_seqlen,
            enabled=self.config.grad_checkpointing and self.training,
        )
        return scatter_valid(self.ln_final(flat), valid)
