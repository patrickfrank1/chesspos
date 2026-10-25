from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

from src.dataset.token_stream import CLS, MASK, PAD, SEGMENT_LEN, SEP, VOCAB_SIZE
from src.modeling.masking import IGNORE_INDEX
from src.modeling.varlen import (
    cu_seqlens,
    flatten_valid,
    scatter_valid,
    varlen_attention,
)

IGNORE_TOKENS = (CLS, SEP, MASK)
STRUCTURAL_SLOT = SEGMENT_LEN


def slot_indices(tokens: torch.Tensor, segment_ids: torch.Tensor) -> torch.Tensor:
    """Map every token to its slot within the segment's fixed content layout.

    Content tokens are indexed by their offset inside the 67-token block
    (squares 0-63, turn 64, castle 65, en-passant 66); structural tokens
    (CLS/SEP) and padding map to ``STRUCTURAL_SLOT``. Without this index the
    trunk's attention is permutation-equivariant over each segment, so a fully
    masked board would receive one identical prediction for every square.
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


@dataclass
class TransformerConfig:
    vocab_size: int = VOCAB_SIZE
    d_model: int = 512
    n_heads: int = 16
    n_layers: int = 8
    d_ff: int = 2048
    dropout: float = 0.1
    max_seq_len: int = 2048
    max_segments: int = 128


class TransformerBlock(nn.Module):
    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.n_heads = config.n_heads
        self.dropout = config.dropout
        self.ln_attn = nn.LayerNorm(config.d_model)
        self.qkv = nn.Linear(config.d_model, 3 * config.d_model)
        self.attn_out = nn.Linear(config.d_model, config.d_model)
        self.ln_ffn = nn.LayerNorm(config.d_model)
        self.fc_in = nn.Linear(config.d_model, config.d_ff)
        self.fc_out = nn.Linear(config.d_ff, config.d_model)
        return

    def forward(
        self, x: torch.Tensor, cu: torch.Tensor, max_seqlen: int
    ) -> torch.Tensor:
        """Run the block over flat valid tokens ``[T, d_model]``.

        ``cu`` delimits each row's valid run so attention never crosses rows;
        see :mod:`src.modeling.varlen`.
        """
        d_model = x.shape[-1]
        h = self.ln_attn(x)
        qkv = self.qkv(h).view(-1, 3, self.n_heads, d_model // self.n_heads)
        q, k, v = qkv.permute(1, 0, 2, 3).unbind(0)
        attended = varlen_attention(
            q,
            k,
            v,
            cu,
            max_seqlen,
            dropout_p=self.dropout if self.training else 0.0,
        )
        x = x + self.attn_out(attended.reshape(x.shape[0], d_model))
        x = x + self.fc_out(F.gelu(self.fc_in(self.ln_ffn(x))))
        return x


class MaskedStreamTransformer(nn.Module):
    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.config = config
        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.segment_embedding = nn.Embedding(config.max_segments + 1, config.d_model)
        self.slot_embedding = nn.Embedding(SEGMENT_LEN + 1, config.d_model)
        self.blocks = nn.ModuleList(
            TransformerBlock(config) for _ in range(config.n_layers)
        )
        self.ln_final = nn.LayerNorm(config.d_model)
        self.head = nn.Linear(config.d_model, config.vocab_size)
        return

    def forward(
        self,
        tokens: torch.Tensor,
        segment_ids: torch.Tensor,
        padding_mask: torch.Tensor | None = None,
        targets: torch.Tensor | None = None,
        weights: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        if padding_mask is None:
            padding_mask = tokens.ne(0)
        hidden = self._trunk(tokens, segment_ids, padding_mask)
        if targets is None:
            return {"logits": self.head(hidden), "hidden_states": hidden}
        return {
            "loss": self._loss(hidden, targets, weights),
            "hidden_states": hidden,
        }

    def _trunk(
        self,
        tokens: torch.Tensor,
        segment_ids: torch.Tensor,
        padding_mask: torch.Tensor,
    ) -> torch.Tensor:
        x = (
            self.token_embedding(tokens)
            + self.segment_embedding(segment_ids)
            + self.slot_embedding(slot_indices(tokens, segment_ids))
        )
        cu, max_seqlen = cu_seqlens(padding_mask)
        flat = flatten_valid(x, padding_mask)
        for block in self.blocks:
            flat = block(flat, cu, max_seqlen)
        return scatter_valid(self.ln_final(flat), padding_mask)

    def _loss(
        self,
        hidden: torch.Tensor,
        targets: torch.Tensor,
        weights: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Supervised loss computed only at positions with targets.

        The vocabulary head runs solely on supervised rows (masked tokens);
        ignored positions (unmasked tokens, CLS/SEP, padding) never reach the
        head or the cross-entropy.
        """
        valid = targets.ne(IGNORE_INDEX)
        if not valid.any():
            return hidden.new_zeros(())
        flat_hidden = hidden[valid]
        logits = self.head(flat_hidden)
        per_token = F.cross_entropy(logits, targets[valid], reduction="none")
        if weights is not None:
            w = weights[valid].to(per_token.dtype)
            total = w.sum()
            if total > 0:
                return (per_token * w).sum() / total
        return per_token.mean()

    def encode(
        self,
        tokens: torch.Tensor,
        segment_ids: torch.Tensor,
        padding_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return per-segment mean embeddings with gradients enabled."""
        if padding_mask is None:
            padding_mask = tokens.ne(0)
        hidden = self._trunk(tokens, segment_ids, padding_mask)
        return self.pool_hidden(hidden, tokens, segment_ids, padding_mask)

    @torch.no_grad()
    def embed(
        self,
        tokens: torch.Tensor,
        segment_ids: torch.Tensor,
        padding_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self.encode(tokens, segment_ids, padding_mask)

    def pool_hidden(
        self,
        hidden: torch.Tensor,
        tokens: torch.Tensor,
        segment_ids: torch.Tensor,
        padding_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Mean-pool hidden states per segment, ignoring structural/mask tokens."""
        if padding_mask is None:
            padding_mask = tokens.ne(0)
        valid = padding_mask
        for token in IGNORE_TOKENS:
            valid = valid & tokens.ne(token)
        valid = valid & segment_ids.lt(self.config.max_segments)

        batch, _, d_model = hidden.shape
        n_segments = int(segment_ids[valid].max().item()) + 1 if valid.any() else 1
        sums = torch.zeros(batch, n_segments, d_model, device=hidden.device)
        counts = torch.zeros(batch, n_segments, device=hidden.device)
        index = segment_ids.clamp(0, n_segments - 1)
        sums.scatter_add_(
            1,
            index.unsqueeze(-1).expand_as(hidden),
            hidden.masked_fill(~valid.unsqueeze(-1), 0.0),
        )
        counts.scatter_add_(1, index, valid.to(counts.dtype))
        pooled = sums / counts.clamp_min(1.0).unsqueeze(-1)
        segment_mask = counts > 0
        return pooled, segment_mask
