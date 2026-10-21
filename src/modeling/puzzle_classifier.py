from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from src.modeling.transformer import MaskedStreamTransformer


@dataclass
class MeanPoolingClassifierConfig:
    d_model: int
    dropout: float = 0.0

    def __post_init__(self):
        if self.d_model <= 0:
            raise ValueError("d_model must be positive")
        if not 0 <= self.dropout < 1:
            raise ValueError("dropout must be in [0, 1)")
        return


class MeanPoolingClassifier(nn.Module):
    """Mean-pool a sequence of position embeddings into one sequence logit."""

    def __init__(self, config: MeanPoolingClassifierConfig):
        super().__init__()
        self.config = config
        self.dropout = nn.Dropout(config.dropout)
        self.classifier = nn.Linear(config.d_model, 1)
        return

    def forward(
        self, embeddings: torch.Tensor, mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        if mask is None:
            pooled = embeddings.mean(dim=1)
        else:
            weights = mask.unsqueeze(-1).to(embeddings.dtype)
            pooled = (embeddings * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(
                1.0
            )
        return self.classifier(self.dropout(pooled)).squeeze(-1)


class FrozenEncoderClassifier(nn.Module):
    """Frozen foundation encoder plus a trainable sequence classification head."""

    def __init__(
        self,
        encoder: MaskedStreamTransformer,
        head: MeanPoolingClassifier,
    ):
        super().__init__()
        self.encoder = encoder
        self.head = head
        for parameter in self.encoder.parameters():
            parameter.requires_grad = False
        self.encoder.eval()
        return

    def train(self, mode: bool = True) -> "FrozenEncoderClassifier":
        super().train(mode)
        self.encoder.eval()
        return self

    def forward(
        self,
        tokens: torch.Tensor,
        segment_ids: torch.Tensor,
        padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        with torch.no_grad():
            embeddings, mask = self.encoder.embed(
                tokens, segment_ids, padding_mask=padding_mask
            )
        return self.head(embeddings, mask)
