from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

from src.dataset.token_stream import (
    CLS,
    EMPTY,
    MASK,
    PAD,
    SEGMENT_LEN,
    SEP,
    TURN_WHITE,
    VOCAB_SIZE,
)
from src.modeling.masking import IGNORE_INDEX
from src.modeling.varlen import (
    cu_seqlens,
    flatten_valid,
    scatter_valid,
    varlen_attention,
)
from src.training.aux_heads import AuxiliaryHeads

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


def content_isolation_mask(
    tokens: torch.Tensor,
    segment_ids: torch.Tensor,
    padding_mask: torch.Tensor,
) -> torch.Tensor:
    """Block content queries from attending to state or other segments.

    The mask is ``[batch, 1, query, key]``. State/CLS/SEP queries may attend to
    the whole segment; content queries may only attend to square tokens in the
    same segment. This makes content embeddings exactly invariant to the turn,
    castling, and en-passant tokens.
    """
    content = (tokens >= EMPTY) & (tokens < TURN_WHITE)
    same_segment = segment_ids[:, :, None] == segment_ids[:, None, :]
    allowed = padding_mask.bool()[:, None, :] & same_segment
    content_keys = content[:, None, :] & same_segment & padding_mask.bool()[:, None, :]
    allowed = torch.where(content[:, :, None], content_keys, allowed)
    return allowed[:, None, :, :]


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
    content_pooling: bool = False
    aux_heads: bool = False
    phase_classes: int = 3


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

    def forward_padded(
        self,
        x: torch.Tensor,
        padding_mask: torch.Tensor,
        attn_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Run the block over equal-length padded rows in one SDPA call.

        This is used for the many short alone views, where launching a
        separate variable-length attention kernel per view is launch-bound.
        ``attn_mask`` may additionally restrict which keys each query sees.
        """
        batch, seq_len, d_model = x.shape
        h = self.ln_attn(x)
        qkv = self.qkv(h).view(batch, seq_len, 3, self.n_heads, d_model // self.n_heads)
        q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(0)
        if attn_mask is None:
            attn_mask = padding_mask[:, None, None, :]
        attended = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=attn_mask,
            dropout_p=self.dropout if self.training else 0.0,
        )
        attended = attended.transpose(1, 2).reshape(batch, seq_len, d_model)
        x = x + self.attn_out(attended)
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
        self.aux_heads = (
            AuxiliaryHeads(config.d_model, config.phase_classes)
            if config.aux_heads
            else None
        )
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

    def _trunk_padded(
        self,
        tokens: torch.Tensor,
        segment_ids: torch.Tensor,
        padding_mask: torch.Tensor,
        *,
        isolate_content: bool = False,
    ) -> torch.Tensor:
        """Trunk for equal-length padded rows (one SDPA call per layer)."""
        x = (
            self.token_embedding(tokens)
            + self.segment_embedding(segment_ids)
            + self.slot_embedding(slot_indices(tokens, segment_ids))
        )
        attn_mask = (
            content_isolation_mask(tokens, segment_ids, padding_mask)
            if isolate_content
            else None
        )
        for block in self.blocks:
            x = block.forward_padded(x, padding_mask, attn_mask)
        return self.ln_final(x)

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

    def encode_content(
        self,
        tokens: torch.Tensor,
        segment_ids: torch.Tensor,
        padding_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Pool only square tokens (empty squares and pieces)."""
        if padding_mask is None:
            padding_mask = tokens.ne(0)
        hidden = self._trunk_padded(
            tokens, segment_ids, padding_mask, isolate_content=True
        )
        return self.pool_content(hidden, tokens, segment_ids, padding_mask)

    def encode_state(
        self,
        tokens: torch.Tensor,
        segment_ids: torch.Tensor,
        padding_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Pool only turn/castling/en-passant tokens.

        State queries see every token in their segment. They use the same
        content-isolated trunk as :meth:`encode_views`, so state embeddings
        stay consistent between training and standalone evaluation while
        content queries cannot read state tokens.
        """
        if padding_mask is None:
            padding_mask = tokens.ne(0)
        hidden = self._trunk_padded(
            tokens, segment_ids, padding_mask, isolate_content=True
        )
        return self.pool_state(hidden, tokens, segment_ids, padding_mask)

    def encode_views(
        self,
        tokens: torch.Tensor,
        segment_ids: torch.Tensor,
        padding_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute content and state pools from a single trunk pass."""
        if padding_mask is None:
            padding_mask = tokens.ne(0)
        hidden = self._trunk_padded(
            tokens, segment_ids, padding_mask, isolate_content=True
        )
        content, segment_mask = self.pool_content(
            hidden, tokens, segment_ids, padding_mask
        )
        state, _ = self.pool_state(hidden, tokens, segment_ids, padding_mask)
        return content, state, segment_mask

    @torch.no_grad()
    def embed(
        self,
        tokens: torch.Tensor,
        segment_ids: torch.Tensor,
        padding_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.config.content_pooling:
            return self.encode_content(tokens, segment_ids, padding_mask)
        return self.encode(tokens, segment_ids, padding_mask)

    def content_segment_mask(
        self,
        tokens: torch.Tensor,
        segment_ids: torch.Tensor,
        padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return which segments contain at least one content token."""
        if padding_mask is None:
            padding_mask = tokens.ne(0)
        valid = padding_mask & (tokens >= EMPTY) & (tokens < TURN_WHITE)
        valid = valid & segment_ids.lt(self.config.max_segments)
        batch = tokens.shape[0]
        n_segments = int(segment_ids[valid].max().item()) + 1 if valid.any() else 1
        counts = torch.zeros(
            batch, n_segments, device=tokens.device, dtype=torch.float32
        )
        index = segment_ids.clamp(0, n_segments - 1)
        counts.scatter_add_(1, index, valid.to(counts.dtype))
        return counts > 0

    def _pool(
        self,
        hidden: torch.Tensor,
        tokens: torch.Tensor,
        segment_ids: torch.Tensor,
        padding_mask: torch.Tensor | None,
        token_filter: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if padding_mask is None:
            padding_mask = tokens.ne(0)
        valid = padding_mask
        for token in IGNORE_TOKENS:
            valid = valid & tokens.ne(token)
        if token_filter is not None:
            valid = valid & token_filter
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

    def pool_hidden(
        self,
        hidden: torch.Tensor,
        tokens: torch.Tensor,
        segment_ids: torch.Tensor,
        padding_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Mean-pool hidden states per segment, ignoring structural/mask tokens."""
        return self._pool(hidden, tokens, segment_ids, padding_mask, None)

    def pool_content(
        self,
        hidden: torch.Tensor,
        tokens: torch.Tensor,
        segment_ids: torch.Tensor,
        padding_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        token_filter = (tokens >= EMPTY) & (tokens < TURN_WHITE)
        return self._pool(hidden, tokens, segment_ids, padding_mask, token_filter)

    def pool_state(
        self,
        hidden: torch.Tensor,
        tokens: torch.Tensor,
        segment_ids: torch.Tensor,
        padding_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        token_filter = (tokens >= TURN_WHITE) & (tokens < VOCAB_SIZE)
        return self._pool(hidden, tokens, segment_ids, padding_mask, token_filter)
