"""Attention primitives shared by the transformer and Perceiver models."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def _attention_mask(
    attn_mask: torch.Tensor | None,
    key_padding_mask: torch.Tensor | None,
    n_queries: int,
    n_keys: int,
) -> torch.Tensor | None:
    """Combine an optional query/key mask with an optional key-padding mask."""
    allowed = None
    if key_padding_mask is not None:
        allowed = key_padding_mask.bool()[:, None, None, :].expand(
            -1, 1, n_queries, n_keys
        )
    if attn_mask is not None:
        mask = attn_mask.bool()
        if mask.ndim == 2:
            mask = mask[None, None, :, :]
        elif mask.ndim == 3:
            mask = mask[:, None, :, :]
        allowed = mask if allowed is None else allowed & mask
    return allowed


def scaled_dot_product_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    attn_mask: torch.Tensor | None = None,
    key_padding_mask: torch.Tensor | None = None,
    dropout_p: float = 0.0,
) -> torch.Tensor:
    """SDPA with ``True``-means-attend masks and safe fully-masked queries.

    ``q`` is ``[N, H, Lq, D]`` and ``k``/``v`` are ``[N, H, Lk, D]``. Queries
    with no allowed key produce a zero output instead of a NaN row.
    """
    allowed = _attention_mask(attn_mask, key_padding_mask, q.shape[2], k.shape[2])
    if allowed is None:
        return F.scaled_dot_product_attention(q, k, v, dropout_p=dropout_p)
    any_allowed = allowed.any(dim=-1, keepdim=True)
    safe_allowed = allowed | ~any_allowed
    attended = F.scaled_dot_product_attention(
        q, k, v, attn_mask=safe_allowed, dropout_p=dropout_p
    )
    return attended * any_allowed.to(attended.dtype)
