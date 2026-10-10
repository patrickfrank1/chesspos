from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

from src.dataset.token_stream import CLS, SEP, PIECE_SQUARE_BASE, VOCAB_SIZE
from src.modeling.masking import IGNORE_INDEX

IGNORE_TOKENS = (CLS, SEP)


@dataclass
class TransformerConfig:
    vocab_size: int = VOCAB_SIZE
    d_model: int = 256
    n_heads: int = 8
    n_layers: int = 4
    d_ff: int = 1024
    dropout: float = 0.1
    max_seq_len: int = 2048
    max_segments: int = 128
    piece_square_base: int = PIECE_SQUARE_BASE
    n_pieces: int = 12
    n_squares: int = 64


class FactorizedTokenEmbedding(nn.Module):
    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.piece_square_base = config.piece_square_base
        self.piece = nn.Embedding(config.n_pieces, config.d_model)
        self.square = nn.Embedding(config.n_squares, config.d_model)
        self.state = nn.Embedding(config.piece_square_base, config.d_model)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        is_piece = tokens >= self.piece_square_base
        offset = (tokens - self.piece_square_base).clamp_min(0)
        piece_index = torch.where(is_piece, offset // 64, torch.zeros_like(offset))
        square_index = torch.where(is_piece, offset % 64, torch.zeros_like(offset))
        state_index = tokens.clamp(0, self.piece_square_base - 1)
        piece_emb = self.piece(piece_index) + self.square(square_index)
        state_emb = self.state(state_index)
        return torch.where(is_piece.unsqueeze(-1), piece_emb, state_emb)


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

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor) -> torch.Tensor:
        batch, length, d_model = x.shape
        h = self.ln_attn(x)
        qkv = self.qkv(h).view(batch, length, 3, self.n_heads, d_model // self.n_heads)
        q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(0)
        attended = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=attn_mask,
            dropout_p=self.dropout if self.training else 0.0,
        )
        attended = attended.transpose(1, 2).reshape(batch, length, d_model)
        x = x + self.attn_out(attended)
        x = x + self.fc_out(F.gelu(self.fc_in(self.ln_ffn(x))))
        return x


class MaskedStreamTransformer(nn.Module):
    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.config = config
        self.token_embedding = FactorizedTokenEmbedding(config)
        self.segment_embedding = nn.Embedding(config.max_segments + 1, config.d_model)
        self.blocks = nn.ModuleList(
            TransformerBlock(config) for _ in range(config.n_layers)
        )
        self.ln_final = nn.LayerNorm(config.d_model)
        self.head = nn.Linear(config.d_model, config.vocab_size)

    def forward(
        self,
        tokens: torch.Tensor,
        segment_ids: torch.Tensor,
        padding_mask: torch.Tensor | None = None,
        targets: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        if padding_mask is None:
            padding_mask = tokens.ne(0)
        hidden = self._trunk(tokens, segment_ids, padding_mask)
        logits = self.head(hidden)
        output = {"logits": logits, "hidden_states": hidden}
        if targets is not None:
            output["loss"] = self._loss(logits, targets)
        return output

    def _trunk(
        self,
        tokens: torch.Tensor,
        segment_ids: torch.Tensor,
        padding_mask: torch.Tensor,
    ) -> torch.Tensor:
        batch, length = tokens.shape
        x = self.token_embedding(tokens) + self.segment_embedding(segment_ids)
        key_allowed = padding_mask[:, None, None, :]
        eye = torch.eye(length, dtype=torch.bool, device=tokens.device)
        attn_mask = key_allowed | eye
        for block in self.blocks:
            x = block(x, attn_mask)
        return self.ln_final(x)

    @staticmethod
    def _loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        per_token = F.cross_entropy(
            logits.view(-1, logits.size(-1)),
            targets.view(-1),
            ignore_index=IGNORE_INDEX,
            reduction="none",
        )
        valid = targets.ne(IGNORE_INDEX).view(-1)
        if not valid.any():
            return logits.new_zeros(())
        return per_token[valid].mean()

    @torch.no_grad()
    def embed(
        self,
        tokens: torch.Tensor,
        segment_ids: torch.Tensor,
        padding_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if padding_mask is None:
            padding_mask = tokens.ne(0)
        hidden = self._trunk(tokens, segment_ids, padding_mask)
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
