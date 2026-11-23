from __future__ import annotations

import torch
from torch import nn

FEATURE_RECIPES = ("native", "full")


def encoder_embeddings(
    encoder: nn.Module,
    tokens: torch.Tensor,
    segment_ids: torch.Tensor,
    padding_mask: torch.Tensor | None = None,
    feature: str = "native",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pool a frozen encoder into per-segment embeddings.

    ``native`` uses the checkpoint's own retrieval embedding (``embed``),
    which is content-only when the checkpoint was trained with
    ``content_pooling``. ``full`` uses the non-isolated trunk (``encode``), so
    every pooled segment carries the turn/castling/en-passant tokens.
    """
    if feature == "native":
        return encoder.embed(tokens, segment_ids, padding_mask=padding_mask)
    if feature == "full":
        return encoder.encode(tokens, segment_ids, padding_mask=padding_mask)
    raise ValueError(f"feature must be one of {FEATURE_RECIPES}, got {feature!r}")


def freeze_encoder(encoder: nn.Module) -> None:
    """Freeze every encoder parameter."""
    for parameter in encoder.parameters():
        parameter.requires_grad = False
    return


def embedding_dim(encoder: nn.Module, feature: str = "native") -> int:
    """Return the width of the vectors produced by the feature recipe."""
    if feature == "native":
        return int(getattr(encoder.config, "latent_dim", encoder.config.d_model))
    if feature == "full":
        return int(encoder.config.d_model)
    raise ValueError(f"feature must be one of {FEATURE_RECIPES}, got {feature!r}")
