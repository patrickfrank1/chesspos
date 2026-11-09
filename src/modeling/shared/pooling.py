"""Segment pooling helpers used by the legacy compatibility methods."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from src.dataset.token_stream import CLS, EMPTY, MASK, SEP, TURN_WHITE, VOCAB_SIZE
from src.modeling.masking import IGNORE_INDEX

IGNORE_TOKENS = (CLS, SEP, MASK)


def _pool(
    hidden: torch.Tensor,
    tokens: torch.Tensor,
    segment_ids: torch.Tensor,
    padding_mask: torch.Tensor,
    token_filter: torch.Tensor | None,
    max_segments: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    valid = padding_mask.bool()
    for token in IGNORE_TOKENS:
        valid = valid & tokens.ne(token)
    if token_filter is not None:
        valid = valid & token_filter
    valid = valid & segment_ids.lt(max_segments)

    batch, _, d_model = hidden.shape
    sums = torch.zeros(batch, max_segments, d_model, device=hidden.device)
    counts = torch.zeros(batch, max_segments, device=hidden.device)
    index = segment_ids.clamp(0, max_segments - 1)
    sums.scatter_add_(
        1,
        index.unsqueeze(-1).expand_as(hidden),
        hidden.masked_fill(~valid.unsqueeze(-1), 0.0),
    )
    counts.scatter_add_(1, index, valid.to(counts.dtype))
    pooled = sums / counts.clamp_min(1.0).unsqueeze(-1)
    return pooled, counts > 0


def pool_hidden(
    hidden: torch.Tensor,
    tokens: torch.Tensor,
    segment_ids: torch.Tensor,
    padding_mask: torch.Tensor,
    max_segments: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    return _pool(hidden, tokens, segment_ids, padding_mask, None, max_segments)


def pool_content(
    hidden: torch.Tensor,
    tokens: torch.Tensor,
    segment_ids: torch.Tensor,
    padding_mask: torch.Tensor,
    max_segments: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    token_filter = (tokens >= EMPTY) & (tokens < TURN_WHITE)
    return _pool(hidden, tokens, segment_ids, padding_mask, token_filter, max_segments)


def pool_state(
    hidden: torch.Tensor,
    tokens: torch.Tensor,
    segment_ids: torch.Tensor,
    padding_mask: torch.Tensor,
    max_segments: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    token_filter = (tokens >= TURN_WHITE) & (tokens < VOCAB_SIZE)
    return _pool(hidden, tokens, segment_ids, padding_mask, token_filter, max_segments)


def masked_loss(
    hidden: torch.Tensor,
    targets: torch.Tensor,
    head: torch.nn.Module,
    weights: torch.Tensor | None = None,
) -> torch.Tensor:
    """Cross-entropy on positions with targets, mirroring the legacy model."""
    valid = targets.ne(IGNORE_INDEX)
    if not valid.any():
        return hidden.new_zeros(())
    logits = head(hidden[valid])
    per_token = F.cross_entropy(logits, targets[valid], reduction="none")
    if weights is not None:
        w = weights[valid].to(per_token.dtype)
        total = w.sum()
        if total > 0:
            return (per_token * w).sum() / total
    return per_token.mean()
