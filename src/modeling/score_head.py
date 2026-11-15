from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import nn

from src.modeling.frozen import FEATURE_RECIPES, encoder_embeddings, freeze_encoder
from src.modeling.transformer import MaskedStreamTransformer

POOLING_MODES = ("mean", "attention")


@dataclass
class ScoreHeadConfig:
    d_model: int
    hidden_dim: int | None = 256
    dropout: float = 0.0
    pooling: str = "mean"
    state_features: int = 0

    def __post_init__(self) -> None:
        if self.d_model <= 0:
            raise ValueError("d_model must be positive")
        if self.hidden_dim is not None and self.hidden_dim <= 0:
            raise ValueError("hidden_dim must be positive")
        if not 0 <= self.dropout < 1:
            raise ValueError("dropout must be in [0, 1)")
        if self.pooling not in POOLING_MODES:
            raise ValueError(f"pooling must be one of {POOLING_MODES}")
        if self.state_features < 0:
            raise ValueError("state_features must be non-negative")
        return


class ScoreHead(nn.Module):
    """Pool position embeddings and regress a White-POV score in [-1, 1]."""

    def __init__(self, config: ScoreHeadConfig):
        super().__init__()
        self.config = config
        self.dropout = nn.Dropout(config.dropout)
        self.activation = nn.GELU()
        if config.pooling == "attention":
            self.query: nn.Parameter | None = nn.Parameter(torch.zeros(config.d_model))
        else:
            self.query = None
        in_features = config.d_model + config.state_features
        if config.hidden_dim is None:
            self.projection = None
            self.classifier = nn.Linear(in_features, 1)
        else:
            self.projection = nn.Linear(in_features, config.hidden_dim)
            self.classifier = nn.Linear(config.hidden_dim, 1)
        # Start near a draw instead of inside the tanh saturation region.
        nn.init.normal_(self.classifier.weight, std=0.01)
        nn.init.zeros_(self.classifier.bias)
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
        self,
        embeddings: torch.Tensor,
        mask: torch.Tensor | None = None,
        features: torch.Tensor | None = None,
    ) -> torch.Tensor:
        hidden = self.dropout(self.pool(embeddings, mask))
        if self.config.state_features:
            if features is None:
                raise ValueError(
                    f"this head expects {self.config.state_features} state features"
                )
            if features.shape[-1] != self.config.state_features:
                raise ValueError(
                    f"expected {self.config.state_features} state features, "
                    f"got {features.shape[-1]}"
                )
            hidden = torch.cat([hidden, features.to(hidden.dtype)], dim=-1)
        if self.projection is not None:
            hidden = self.dropout(self.activation(self.projection(hidden)))
        return torch.tanh(self.classifier(hidden).squeeze(-1))


class FrozenEncoderScore(nn.Module):
    """Frozen foundation encoder plus a trainable score-regression head."""

    def __init__(
        self,
        encoder: MaskedStreamTransformer,
        head: ScoreHead,
        feature: str = "native",
        unfreeze_last_block: bool = False,
    ):
        super().__init__()
        if feature not in FEATURE_RECIPES:
            raise ValueError(
                f"feature must be one of {FEATURE_RECIPES}, got {feature!r}"
            )
        if unfreeze_last_block and feature != "full":
            raise ValueError(
                "unfreeze_last_block requires feature='full': the native "
                "embedding runs under torch.no_grad()"
            )
        self.encoder = encoder
        self.head = head
        self.feature = feature
        self.unfreeze_last_block = unfreeze_last_block
        freeze_encoder(self.encoder)
        if unfreeze_last_block:
            for parameter in self.encoder.blocks[-1].parameters():
                parameter.requires_grad = True
            for parameter in self.encoder.ln_final.parameters():
                parameter.requires_grad = True
            self.encoder.train()
        else:
            self.encoder.eval()
        return

    def train(self, mode: bool = True) -> "FrozenEncoderScore":
        super().train(mode)
        if self.unfreeze_last_block:
            self.encoder.train(mode)
        else:
            self.encoder.eval()
        return self

    def forward(
        self,
        tokens: torch.Tensor,
        segment_ids: torch.Tensor,
        padding_mask: torch.Tensor | None = None,
        features: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if self.unfreeze_last_block:
            embeddings, mask = encoder_embeddings(
                self.encoder,
                tokens,
                segment_ids,
                padding_mask=padding_mask,
                feature=self.feature,
            )
        else:
            with torch.no_grad():
                embeddings, mask = encoder_embeddings(
                    self.encoder,
                    tokens,
                    segment_ids,
                    padding_mask=padding_mask,
                    feature=self.feature,
                )
        return self.head(embeddings, mask, features)
