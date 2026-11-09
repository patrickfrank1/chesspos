"""Token, segment, and slot embeddings shared by model variants."""

from __future__ import annotations

import torch
from torch import nn

from src.dataset.token_stream import CLS, PAD, SEGMENT_LEN, SEP

STRUCTURAL_SLOT = SEGMENT_LEN


def slot_indices(tokens: torch.Tensor, segment_ids: torch.Tensor) -> torch.Tensor:
    """Map tokens to their slot inside the fixed 67-token segment layout.

    Content slots are 0-63, turn/castle/en-passant slots are 64-66, and
    CLS/SEP/padding map to ``STRUCTURAL_SLOT``. This mirrors the legacy
    transformer helper; it is duplicated here so the Perceiver does not depend
    on the legacy module.
    """
    positions = torch.arange(tokens.shape[1], device=tokens.device)
    is_group_start = torch.ones_like(segment_ids, dtype=torch.bool)
    is_group_start[:, 1:] = segment_ids[:, 1:] != segment_ids[:, :-1]
    group_start = torch.cummax(
        torch.where(
            is_group_start,
            positions.unsqueeze(0).expand_as(segment_ids),
            torch.zeros_like(segment_ids),
        ),
        dim=1,
    ).values
    offset = positions.unsqueeze(0) - group_start
    structural = (tokens == CLS) | (tokens == SEP) | (tokens == PAD)
    content = (offset - 1).clamp(0, SEGMENT_LEN - 1)
    return torch.where(structural, STRUCTURAL_SLOT, content)


class SegmentEmbedding(nn.Module):
    """Add token, segment, and within-segment slot embeddings."""

    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        max_segments: int,
        segment_len: int = SEGMENT_LEN,
    ) -> None:
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.segment_embedding = nn.Embedding(max_segments + 1, d_model)
        self.slot_embedding = nn.Embedding(segment_len, d_model)
        return

    def forward(
        self, token_table: torch.Tensor, segment_ids: torch.Tensor
    ) -> torch.Tensor:
        """Embed ``[batch, segments, segment_len]`` tokens and segment ids."""
        segment_len = token_table.shape[-1]
        slots = torch.arange(segment_len, device=token_table.device)
        x = self.token_embedding(token_table)
        x = x + self.segment_embedding(segment_ids)[:, :, None, :]
        return x + self.slot_embedding(slots)[None, None, :, :]
