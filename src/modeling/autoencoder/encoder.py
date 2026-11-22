"""Packed-sequence encoder: token, segment, and slot embeddings plus blocks."""

from __future__ import annotations

import torch
from torch import nn

from src.modeling.autoencoder.config import AttentionAutoencoderConfig
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
    """

    def __init__(self, config: AttentionAutoencoderConfig) -> None:
        super().__init__()
        self.embedding = PackedSegmentEmbedding(config)
        self.blocks = nn.ModuleList(
            VarlenTransformerLayer(
                config.d_model, config.n_heads, config.d_ff, config.dropout
            )
            for _ in range(config.n_layers)
        )
        self.ln_final = nn.LayerNorm(config.d_model)
        return

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
        for block in self.blocks:
            flat = block(flat, cu, max_seqlen)
        return scatter_valid(self.ln_final(flat), valid)
