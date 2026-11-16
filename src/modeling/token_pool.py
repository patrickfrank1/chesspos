from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import nn

from src.dataset.token_stream import CLS, EMPTY, MASK, PAD, SEP, TURN_WHITE

POOL_MODES = ("mean", "attention")
TOKEN_FILTERS = ("full", "content")


@dataclass
class TokenPoolConfig:
    d_model: int
    mode: str = "mean"
    token_filter: str = "full"

    def __post_init__(self) -> None:
        if self.d_model <= 0:
            raise ValueError("d_model must be positive")
        if self.mode not in POOL_MODES:
            raise ValueError(f"mode must be one of {POOL_MODES}")
        if self.token_filter not in TOKEN_FILTERS:
            raise ValueError(f"token_filter must be one of {TOKEN_FILTERS}")
        return


class TokenPool(nn.Module):
    """Pool the per-token hidden states of each segment.

    ``mean`` reproduces the encoder's own pooling. ``attention`` learns one
    query vector and pools with a masked softmax over the valid tokens of each
    segment; the query is zero-initialised so training starts at the mean.
    """

    def __init__(self, config: TokenPoolConfig):
        super().__init__()
        self.config = config
        if config.mode == "attention":
            self.query: nn.Parameter | None = nn.Parameter(torch.zeros(config.d_model))
        else:
            self.query = None
        return

    def valid_mask(
        self,
        tokens: torch.Tensor,
        segment_ids: torch.Tensor,
        padding_mask: torch.Tensor,
    ) -> torch.Tensor:
        valid = padding_mask.bool()
        for token in (CLS, SEP, MASK, PAD):
            valid = valid & tokens.ne(token)
        if self.config.token_filter == "content":
            valid = valid & (tokens >= EMPTY) & (tokens < TURN_WHITE)
        return valid & segment_ids.ge(0)

    def forward(
        self,
        hidden: torch.Tensor,
        tokens: torch.Tensor,
        segment_ids: torch.Tensor,
        padding_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch, _, dim = hidden.shape
        n_segments = int(segment_ids.max().item()) + 1
        segment = segment_ids.clamp(0, n_segments - 1).long()
        valid = self.valid_mask(tokens, segment_ids, padding_mask)
        counts = torch.zeros(
            batch, n_segments, device=hidden.device, dtype=hidden.dtype
        )
        counts.scatter_add_(1, segment, valid.to(hidden.dtype))
        expanded = segment.unsqueeze(-1).expand_as(hidden)

        if self.config.mode == "mean":
            sums = torch.zeros(
                batch, n_segments, dim, device=hidden.device, dtype=hidden.dtype
            )
            sums.scatter_add_(1, expanded, hidden * valid.unsqueeze(-1))
            pooled = sums / counts.clamp_min(1.0).unsqueeze(-1)
        else:
            query = self.query.to(hidden.dtype)
            scores = (hidden @ query / math.sqrt(dim)).masked_fill(
                ~valid, float("-inf")
            )
            max_scores = torch.full(
                (batch, n_segments),
                float("-inf"),
                device=hidden.device,
                dtype=hidden.dtype,
            )
            max_scores.scatter_reduce_(
                1, segment, scores, reduce="amax", include_self=True
            )
            max_scores = torch.where(
                torch.isfinite(max_scores),
                max_scores,
                torch.zeros_like(max_scores),
            )
            weights = torch.exp(scores - max_scores.gather(1, segment))
            weights = weights * valid.to(hidden.dtype)
            denom = torch.zeros(
                batch, n_segments, device=hidden.device, dtype=hidden.dtype
            )
            denom.scatter_add_(1, segment, weights)
            weights = weights / denom.gather(1, segment).clamp_min(1e-9)
            pooled = torch.zeros(
                batch, n_segments, dim, device=hidden.device, dtype=hidden.dtype
            )
            pooled.scatter_add_(1, expanded, hidden * weights.unsqueeze(-1))
        return pooled, counts > 0
