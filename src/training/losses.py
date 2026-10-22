from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass

import torch
from torch import nn


@dataclass(frozen=True)
class FocalLossConfig:
    gamma: float
    alpha: float


def focal_loss_with_logits(
    logits: torch.Tensor,
    targets: torch.Tensor,
    gamma: float = 2.0,
    alpha: float = 0.5,
    reduction: str = "mean",
) -> torch.Tensor:
    """Binary focal loss computed from logits.

    ``alpha`` weights the positive class, ``1 - alpha`` the negative class.
    ``gamma`` controls how strongly easy examples are downweighted. The base
    cross entropy is computed from logits so saturated predictions stay finite.
    """
    if reduction not in ("none", "sum", "mean"):
        raise ValueError(f"unsupported reduction {reduction!r}")

    targets = targets.to(logits.dtype)
    cross_entropy = nn.functional.binary_cross_entropy_with_logits(
        logits, targets, reduction="none"
    )
    probabilities = torch.sigmoid(logits)
    p_t = probabilities * targets + (1.0 - probabilities) * (1.0 - targets)
    alpha_t = alpha * targets + (1.0 - alpha) * (1.0 - targets)
    loss = alpha_t * (1.0 - p_t).pow(gamma) * cross_entropy

    if reduction == "sum":
        return loss.sum()
    if reduction == "none":
        return loss
    return loss.mean()


def calibrate_focal_loss(class_counts: Mapping[int, int]) -> FocalLossConfig:
    """Pick focal parameters from a binary label distribution.

    ``alpha`` is the negative prevalence, which equalizes the expected loss
    contribution of both classes. ``gamma`` maps the imbalance ratio onto the
    standard 0-2 range: balanced labels disable modulation, and ratios of 4:1
    or worse saturate at 2.
    """
    if any(label not in (0, 1) for label in class_counts):
        raise ValueError("class counts must be keyed by 0 and 1")
    n_negative = int(class_counts.get(0, 0))
    n_positive = int(class_counts.get(1, 0))
    if n_negative < 0 or n_positive < 0:
        raise ValueError("class counts must be non-negative")
    if n_negative == 0 or n_positive == 0:
        raise ValueError("both classes are required to calibrate focal loss")

    prevalence = n_positive / (n_negative + n_positive)
    alpha = 1.0 - prevalence
    imbalance = max(prevalence, 1.0 - prevalence) / min(prevalence, 1.0 - prevalence)
    gamma = min(2.0, math.log2(imbalance))
    return FocalLossConfig(gamma=gamma, alpha=alpha)


def resolve_focal_loss(
    loss: str,
    class_counts: Mapping[int, int],
    gamma: float | None,
    alpha: float | None,
) -> FocalLossConfig | None:
    """Resolve the focal configuration, calibrating any missing parameter."""
    if loss == "bce":
        return None
    if loss != "focal":
        raise ValueError(f"unsupported loss {loss!r}")

    calibrated = calibrate_focal_loss(class_counts)
    return FocalLossConfig(
        gamma=calibrated.gamma if gamma is None else float(gamma),
        alpha=calibrated.alpha if alpha is None else float(alpha),
    )
