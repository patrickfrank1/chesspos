"""Segment-aware cross-attention wrappers."""

from __future__ import annotations

import torch
from torch import nn

from src.modeling.shared.layers import CrossAttention


class SegmentCrossAttention(nn.Module):
    """Cross-attend per-segment queries to per-segment key/value states."""

    def __init__(self, d_model: int, n_heads: int, dropout: float) -> None:
        super().__init__()
        self.attention = CrossAttention(d_model, n_heads, dropout)
        return

    def forward(
        self,
        queries: torch.Tensor,
        key_values: torch.Tensor,
        segment_valid: torch.Tensor,
        key_valid: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Attend ``[B, M, Lq, D]`` queries over ``[B, M, Lk, D]`` states."""
        batch, n_segments, n_queries, d_model = queries.shape
        n_keys = key_values.shape[2]
        q = queries.reshape(batch * n_segments, n_queries, d_model)
        kv = key_values.reshape(batch * n_segments, n_keys, d_model)
        key_mask = (
            segment_valid[:, :, None]
            .expand(-1, -1, n_keys)
            .reshape(batch * n_segments, n_keys)
        )
        if key_valid is not None:
            key_mask = key_mask & key_valid.reshape(batch * n_segments, n_keys)
        attended = self.attention(q, kv, key_padding_mask=key_mask)
        return attended.view(batch, n_segments, n_queries, d_model)
