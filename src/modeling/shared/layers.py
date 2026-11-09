"""Reusable feed-forward and attention blocks."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from src.modeling.shared.attention import scaled_dot_product_attention


class FeedForward(nn.Module):
    """Two-layer GELU feed-forward network."""

    def __init__(self, d_model: int, d_ff: int, dropout: float) -> None:
        super().__init__()
        self.fc_in = nn.Linear(d_model, d_ff)
        self.fc_out = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        return

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc_in(x)
        x = F.gelu(x)
        x = self.fc_out(x)
        return self.dropout(x)


class SelfAttention(nn.Module):
    """Pre-norm multi-head self-attention over independent sequences."""

    def __init__(self, d_model: int, n_heads: int, dropout: float) -> None:
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.dropout = dropout
        self.ln = nn.LayerNorm(d_model)
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.out = nn.Linear(d_model, d_model)
        self.resid_dropout = nn.Dropout(dropout)
        return

    def forward(
        self,
        x: torch.Tensor,
        *,
        key_padding_mask: torch.Tensor | None = None,
        attn_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        batch, seq_len, _ = x.shape
        qkv = self.qkv(self.ln(x)).view(batch, seq_len, 3, self.n_heads, self.head_dim)
        q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(0)
        attended = scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask,
            dropout_p=self.dropout if self.training else 0.0,
        )
        attended = attended.transpose(1, 2).reshape(batch, seq_len, self.d_model)
        return x + self.resid_dropout(self.out(attended))


class CrossAttention(nn.Module):
    """Pre-norm multi-head cross-attention from queries to key/value states."""

    def __init__(self, d_model: int, n_heads: int, dropout: float) -> None:
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.dropout = dropout
        self.ln_q = nn.LayerNorm(d_model)
        self.ln_kv = nn.LayerNorm(d_model)
        self.q_proj = nn.Linear(d_model, d_model)
        self.kv_proj = nn.Linear(d_model, 2 * d_model)
        self.out = nn.Linear(d_model, d_model)
        self.resid_dropout = nn.Dropout(dropout)
        return

    def forward(
        self,
        queries: torch.Tensor,
        key_values: torch.Tensor,
        *,
        key_padding_mask: torch.Tensor | None = None,
        attn_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        batch, n_queries, _ = queries.shape
        n_keys = key_values.shape[1]
        q = self.q_proj(self.ln_q(queries)).view(
            batch, n_queries, self.n_heads, self.head_dim
        )
        kv = self.kv_proj(self.ln_kv(key_values)).view(
            batch, n_keys, 2, self.n_heads, self.head_dim
        )
        k, v = kv.permute(2, 0, 3, 1, 4).unbind(0)
        attended = scaled_dot_product_attention(
            q.transpose(1, 2),
            k,
            v,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask,
            dropout_p=self.dropout if self.training else 0.0,
        )
        attended = attended.transpose(1, 2).reshape(batch, n_queries, self.d_model)
        return queries + self.resid_dropout(self.out(attended))


class TransformerLayer(nn.Module):
    """Pre-norm self-attention plus feed-forward residual block."""

    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float) -> None:
        super().__init__()
        self.attn = SelfAttention(d_model, n_heads, dropout)
        self.ln_ff = nn.LayerNorm(d_model)
        self.ff = FeedForward(d_model, d_ff, dropout)
        return

    def forward(
        self,
        x: torch.Tensor,
        *,
        key_padding_mask: torch.Tensor | None = None,
        attn_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        x = self.attn(x, key_padding_mask=key_padding_mask, attn_mask=attn_mask)
        return x + self.ff(self.ln_ff(x))


class CrossAttentionLayer(nn.Module):
    """Pre-norm cross-attention plus feed-forward residual block."""

    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float) -> None:
        super().__init__()
        self.attn = CrossAttention(d_model, n_heads, dropout)
        self.ln_ff = nn.LayerNorm(d_model)
        self.ff = FeedForward(d_model, d_ff, dropout)
        return

    def forward(
        self,
        queries: torch.Tensor,
        key_values: torch.Tensor,
        *,
        key_padding_mask: torch.Tensor | None = None,
        attn_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        queries = self.attn(
            queries,
            key_values,
            key_padding_mask=key_padding_mask,
            attn_mask=attn_mask,
        )
        return queries + self.ff(self.ln_ff(queries))
