from __future__ import annotations

import math
from contextlib import nullcontext
from dataclasses import dataclass

import torch
from torch import nn

from src.modeling.frozen import FEATURE_RECIPES, encoder_embeddings, freeze_encoder
from src.modeling.transformer import MaskedStreamTransformer


@dataclass
class PoolingClassifierConfig:
    d_model: int
    dropout: float = 0.0
    hidden_dim: int | None = None
    pooling: str = "mean"

    def __post_init__(self):
        if self.d_model <= 0:
            raise ValueError("d_model must be positive")
        if not 0 <= self.dropout < 1:
            raise ValueError("dropout must be in [0, 1)")
        if self.hidden_dim is not None and self.hidden_dim <= 0:
            raise ValueError("hidden_dim must be positive")
        if self.pooling not in ("mean", "attention"):
            raise ValueError("pooling must be 'mean' or 'attention'")
        return


class PoolingClassifier(nn.Module):
    """Pool a sequence of position embeddings and classify the sequence."""

    def __init__(self, config: PoolingClassifierConfig):
        super().__init__()
        self.config = config
        self.dropout = nn.Dropout(config.dropout)
        self.activation = nn.GELU()
        if config.pooling == "attention":
            self.query: nn.Parameter | None = nn.Parameter(torch.zeros(config.d_model))
        else:
            self.query = None
        if config.hidden_dim is None:
            self.projection = None
            self.classifier = nn.Linear(config.d_model, 1)
        else:
            self.projection = nn.Linear(config.d_model, config.hidden_dim)
            self.classifier = nn.Linear(config.hidden_dim, 1)
        return

    def attention_weights(
        self, embeddings: torch.Tensor, mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        if self.query is None:
            raise RuntimeError("attention_weights requires pooling='attention'")
        scores = embeddings @ self.query / math.sqrt(self.config.d_model)
        if mask is None:
            return torch.softmax(scores, dim=1)
        valid = mask.to(torch.bool)
        weights = torch.softmax(scores.masked_fill(~valid, float("-inf")), dim=1)
        return weights.masked_fill(~valid.any(dim=1, keepdim=True), 0.0)

    def pool(
        self, embeddings: torch.Tensor, mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        if self.config.pooling == "attention":
            weights = self.attention_weights(embeddings, mask)
            return torch.bmm(weights.unsqueeze(1), embeddings).squeeze(1)
        if mask is None:
            return embeddings.mean(dim=1)
        weights = mask.unsqueeze(-1).to(embeddings.dtype)
        return (embeddings * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)

    def forward(
        self, embeddings: torch.Tensor, mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        hidden = self.dropout(self.pool(embeddings, mask))
        if self.projection is not None:
            hidden = self.dropout(self.activation(self.projection(hidden)))
        return self.classifier(hidden).squeeze(-1)


class FrozenEncoderClassifier(nn.Module):
    """Foundation encoder plus a trainable sequence classification head.

    The encoder is frozen by default. ``unfreeze_last_block`` trains the last
    transformer block and final norm; ``unfreeze_all`` trains the whole
    encoder. Fine-tuning requires ``feature="full"`` because ``embed()`` runs
    under ``torch.no_grad()``.
    """

    def __init__(
        self,
        encoder: MaskedStreamTransformer,
        head: PoolingClassifier,
        feature: str = "native",
        unfreeze_last_block: bool = False,
        unfreeze_all: bool = False,
    ):
        super().__init__()
        if feature not in FEATURE_RECIPES:
            raise ValueError(
                f"feature must be one of {FEATURE_RECIPES}, got {feature!r}"
            )
        if unfreeze_last_block and unfreeze_all:
            raise ValueError(
                "unfreeze_last_block and unfreeze_all are mutually exclusive"
            )
        native_finetuning = bool(getattr(encoder, "supports_native_finetuning", False))
        if unfreeze_last_block and feature != "full":
            raise ValueError("fine-tuning the last block requires feature='full'")
        if unfreeze_all and feature != "full" and not native_finetuning:
            raise ValueError(
                "fine-tuning requires feature='full': the native embedding "
                "runs under torch.no_grad()"
            )
        self.encoder = encoder
        self.head = head
        self.feature = feature
        self.unfreeze_last_block = unfreeze_last_block
        self.unfreeze_all = unfreeze_all
        freeze_encoder(self.encoder)
        if unfreeze_all:
            for parameter in self.encoder.parameters():
                parameter.requires_grad = True
            self.encoder.train()
        elif unfreeze_last_block:
            for parameter in self.encoder.blocks[-1].parameters():
                parameter.requires_grad = True
            for parameter in self.encoder.ln_final.parameters():
                parameter.requires_grad = True
            self.encoder.train()
        else:
            self.encoder.eval()
        return

    @property
    def is_fine_tuning(self) -> bool:
        return self.unfreeze_last_block or self.unfreeze_all

    def train(self, mode: bool = True) -> "FrozenEncoderClassifier":
        super().train(mode)
        if self.is_fine_tuning:
            self.encoder.train(mode)
        else:
            self.encoder.eval()
        return self

    def forward(
        self,
        tokens: torch.Tensor,
        segment_ids: torch.Tensor,
        padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        grad_context = nullcontext() if self.is_fine_tuning else torch.no_grad()
        with grad_context:
            embeddings, mask = encoder_embeddings(
                self.encoder,
                tokens,
                segment_ids,
                padding_mask=padding_mask,
                feature=self.feature,
            )
        return self.head(embeddings, mask)
