"""Latent-conditioned decoder over the packed token sequence."""

from __future__ import annotations

import torch
from torch import nn

from src.modeling.autoencoder.config import AttentionAutoencoderConfig
from src.modeling.shared.embeddings import slot_indices
from src.modeling.shared.layers import VarlenTransformerLayer
from src.modeling.varlen import cu_seqlens, flatten_valid, scatter_valid


class LatentDecoder(nn.Module):
    """Reconstruct every position from the window bottleneck.

    The decoder never sees the input token identities: only the projected
    latent plus segment/slot embeddings. That makes the bottleneck the sole
    path from input to output, so the reconstruction loss cannot be satisfied
    by copying visible tokens around it.
    """

    def __init__(self, config: AttentionAutoencoderConfig) -> None:
        super().__init__()
        self.from_latent = nn.Linear(config.latent_dim, config.d_model)
        self.segment_embedding = nn.Embedding(config.max_segments + 1, config.d_model)
        self.slot_embedding = nn.Embedding(config.segment_len + 1, config.d_model)
        self.blocks = nn.ModuleList(
            VarlenTransformerLayer(
                config.d_model, config.n_heads, config.d_ff, config.dropout
            )
            for _ in range(config.n_decoder_layers)
        )
        self.ln_final = nn.LayerNorm(config.d_model)
        return

    def forward(
        self,
        latent: torch.Tensor,
        tokens: torch.Tensor,
        segment_ids: torch.Tensor,
        padding_mask: torch.Tensor,
    ) -> torch.Tensor:
        valid = padding_mask.bool()
        slots = slot_indices(tokens, segment_ids)
        positions = self.segment_embedding(segment_ids) + self.slot_embedding(slots)
        x = self.from_latent(latent).unsqueeze(1) + positions
        cu, max_seqlen = cu_seqlens(valid)
        flat = flatten_valid(x, valid)
        for block in self.blocks:
            flat = block(flat, cu, max_seqlen)
        return scatter_valid(self.ln_final(flat), valid)
