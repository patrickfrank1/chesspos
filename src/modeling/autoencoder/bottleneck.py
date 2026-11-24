"""Attention-pooled window bottleneck."""

from __future__ import annotations

import math

import torch
from torch import nn


class AttentionBottleneck(nn.Module):
    """Compress a packed window into one ``latent_dim`` vector.

    A single learned query scores every valid token, the values are scaled by
    a sigmoid gate, and the gated attention weights pool the window. The
    pooled vector is projected down to ``latent_dim`` and normalized.
    """

    def __init__(self, d_model: int, latent_dim: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.d_model = d_model
        self.query = nn.Parameter(torch.empty(d_model))
        nn.init.normal_(self.query, mean=0.0, std=0.02)
        self.gate = nn.Linear(d_model, d_model)
        self.key = nn.Linear(d_model, d_model)
        self.value = nn.Linear(d_model, d_model)
        self.out = nn.Linear(d_model, latent_dim)
        self.norm = nn.LayerNorm(latent_dim)
        self.dropout = nn.Dropout(dropout)
        return

    def forward(self, x: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        """Pool ``[batch, seq, d_model]`` into ``[batch, latent_dim]``."""
        valid = valid.bool()
        gated = self.value(x) * torch.sigmoid(self.gate(x))
        scores = (self.key(x) * self.query).sum(dim=-1) / math.sqrt(self.d_model)
        scores = scores.masked_fill(~valid, torch.finfo(scores.dtype).min)
        weights = torch.softmax(scores, dim=-1)
        pooled = (weights.unsqueeze(-1) * gated).sum(dim=1)
        pooled = pooled * valid.any(dim=-1, keepdim=True).to(pooled.dtype)
        return self.norm(self.out(self.dropout(pooled)))
