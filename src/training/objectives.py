from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from src.dataset.token_stream import CLS, PAD, SEP

_WEIGHT_NAMES = (
    "mlm_weight",
    "inv_weight",
    "var_weight",
    "cov_weight",
    "nce_weight",
)


@dataclass(frozen=True)
class AuxiliaryLossConfig:
    """Weights for MLM plus optional geometry and contrastive terms.

    Setting ``nce_weight`` to zero (the default) removes the InfoNCE term
    entirely, so the same training entry point can run the MLM-only baseline.
    """

    mlm_weight: float = 1.0
    inv_weight: float = 0.1
    var_weight: float = 0.1
    cov_weight: float = 0.01
    nce_weight: float = 0.0
    nce_temperature: float = 0.1
    vicreg_gamma: float = 1.0
    vicreg_eps: float = 1e-4
    warmup_steps: int = 0

    def __post_init__(self):
        for name in _WEIGHT_NAMES:
            if getattr(self, name) < 0:
                raise ValueError(f"{name} must be non-negative")
        if self.nce_temperature <= 0:
            raise ValueError("nce_temperature must be positive")
        if self.vicreg_gamma <= 0:
            raise ValueError("vicreg_gamma must be positive")
        if self.vicreg_eps <= 0:
            raise ValueError("vicreg_eps must be positive")
        if self.warmup_steps < 0:
            raise ValueError("warmup_steps must be non-negative")
        return

    def auxiliary_enabled(self) -> bool:
        return any(
            weight > 0
            for weight in (
                self.inv_weight,
                self.var_weight,
                self.cov_weight,
                self.nce_weight,
            )
        )

    def auxiliary_scale(self, step: int) -> float:
        if self.warmup_steps <= 0:
            return 1.0
        return min(1.0, max(0.0, step / self.warmup_steps))


def vicreg_terms(
    embeddings: torch.Tensor,
    gamma: float = 1.0,
    eps: float = 1e-4,
) -> tuple[torch.Tensor, torch.Tensor]:
    """VICReg variance hinge and off-diagonal covariance penalty."""
    z = embeddings.float()
    z = z - z.mean(dim=0, keepdim=True)
    count = max(1, z.shape[0] - 1)
    variance = (z * z).sum(dim=0) / count
    variance_loss = torch.clamp(gamma - torch.sqrt(variance + eps), min=0).mean()
    covariance = (z.T @ z) / count
    off_diagonal = covariance - torch.diag_embed(torch.diagonal(covariance))
    covariance_loss = (off_diagonal**2).sum() / covariance.shape[0]
    return variance_loss, covariance_loss


def alignment_loss(anchors: torch.Tensor, positives: torch.Tensor) -> torch.Tensor:
    """Mean cosine distance between context-removal views."""
    similarity = F.cosine_similarity(
        anchors.float(), positives.float(), dim=-1, eps=1e-8
    )
    return (1.0 - similarity).mean()


def info_nce_loss(
    anchors: torch.Tensor,
    positives: torch.Tensor,
    temperature: float = 0.1,
) -> torch.Tensor:
    """In-batch InfoNCE with the matching pair on the diagonal."""
    anchors = F.normalize(anchors.float(), dim=-1)
    positives = F.normalize(positives.float(), dim=-1)
    logits = anchors @ positives.T / temperature
    labels = torch.arange(anchors.shape[0], device=anchors.device)
    return F.cross_entropy(logits, labels)


def auxiliary_losses(
    mlm_loss: torch.Tensor,
    embeddings: torch.Tensor | None,
    window_targets: torch.Tensor | None,
    alone_targets: torch.Tensor | None,
    config: AuxiliaryLossConfig,
    step: int = 0,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Combine MLM with the enabled geometry terms.

    ``components`` contains unweighted losses keyed by name; ``total`` applies
    the configured weights and the optional auxiliary warmup.
    """
    components = {"mlm": mlm_loss}
    total = config.mlm_weight * mlm_loss
    scale = config.auxiliary_scale(step)

    if config.var_weight > 0 or config.cov_weight > 0:
        if embeddings is None:
            raise ValueError("embeddings are required for variance/covariance")
        var_loss, cov_loss = vicreg_terms(
            embeddings, gamma=config.vicreg_gamma, eps=config.vicreg_eps
        )
        if config.var_weight > 0:
            components["var"] = var_loss
            total = total + config.var_weight * scale * var_loss
        if config.cov_weight > 0:
            components["cov"] = cov_loss
            total = total + config.cov_weight * scale * cov_loss

    if config.inv_weight > 0:
        if window_targets is None or alone_targets is None:
            raise ValueError("paired embeddings are required for invariance")
        inv_loss = alignment_loss(window_targets, alone_targets)
        components["inv"] = inv_loss
        total = total + config.inv_weight * scale * inv_loss

    if config.nce_weight > 0:
        if window_targets is None or alone_targets is None:
            raise ValueError("paired embeddings are required for InfoNCE")
        nce_loss = info_nce_loss(
            window_targets, alone_targets, temperature=config.nce_temperature
        )
        components["nce"] = nce_loss
        total = total + config.nce_weight * scale * nce_loss

    return total, components


def select_target_segments(
    segment_mask: torch.Tensor,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Pick one valid segment per window for the context-removal view."""
    weights = segment_mask.to(torch.float32)
    empty = weights.sum(dim=1) <= 0
    if empty.any():
        weights = weights.clone()
        weights[empty, 0] = 1.0
    return torch.multinomial(weights, 1, generator=generator).squeeze(1)


def extract_alone_views(
    tokens: torch.Tensor,
    segment_ids: torch.Tensor,
    padding_mask: torch.Tensor,
    target_ids: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build one-segment packs for the target positions, without context."""
    views: list[torch.Tensor] = []
    for row, target in enumerate(target_ids.tolist()):
        positions = (
            (segment_ids[row] == int(target))
            & padding_mask[row]
            & tokens[row].ne(CLS)
            & tokens[row].ne(SEP)
        )
        segment = tokens[row][positions]
        views.append(
            torch.cat(
                [
                    torch.tensor([CLS], dtype=tokens.dtype, device=tokens.device),
                    segment,
                    torch.tensor([SEP], dtype=tokens.dtype, device=tokens.device),
                ]
            )
        )

    max_len = max(view.shape[0] for view in views)
    alone_tokens = torch.full(
        (len(views), max_len), PAD, dtype=tokens.dtype, device=tokens.device
    )
    alone_segments = torch.zeros(
        (len(views), max_len), dtype=torch.long, device=tokens.device
    )
    alone_padding = torch.zeros(
        (len(views), max_len), dtype=torch.bool, device=tokens.device
    )
    for row, view in enumerate(views):
        length = view.shape[0]
        alone_tokens[row, :length] = view
        alone_segments[row, length - 1] = 1
        alone_padding[row, :length] = True
    return alone_tokens, alone_segments, alone_padding


def unwrap_model(model):
    """Strip torch.compile/DDP wrappers to reach the underlying module."""
    while True:
        inner = getattr(model, "module", None)
        if inner is None:
            inner = getattr(model, "_orig_mod", None)
        if inner is None or inner is model:
            return model
        model = inner


def compute_training_loss(
    model,
    batch: dict[str, torch.Tensor],
    config: AuxiliaryLossConfig,
    step: int = 0,
    generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Run the model and return the weighted total loss plus components."""
    tokens = batch["tokens"]
    segment_ids = batch["segment_ids"]
    padding_mask = batch["padding_mask"]
    output = model(
        tokens,
        segment_ids,
        padding_mask=padding_mask,
        targets=batch["targets"],
        weights=batch["weights"],
    )
    mlm_loss = output["loss"]
    if not config.auxiliary_enabled():
        return mlm_loss, {"mlm": mlm_loss.detach()}

    encoder = unwrap_model(model)
    pooled, segment_mask = encoder.pool_hidden(
        output["hidden_states"], tokens, segment_ids, padding_mask
    )
    embeddings = (
        pooled[segment_mask]
        if (config.var_weight > 0 or config.cov_weight > 0)
        else None
    )

    window_targets = None
    alone_targets = None
    if config.inv_weight > 0 or config.nce_weight > 0:
        target_ids = select_target_segments(segment_mask, generator=generator)
        alone_tokens, alone_segments, alone_padding = extract_alone_views(
            tokens, segment_ids, padding_mask, target_ids
        )
        alone_pooled, _ = encoder.encode(alone_tokens, alone_segments, alone_padding)
        batch_indices = torch.arange(pooled.shape[0], device=pooled.device)
        window_targets = pooled[batch_indices, target_ids]
        alone_targets = alone_pooled[:, 0]

    total, components = auxiliary_losses(
        mlm_loss,
        embeddings,
        window_targets,
        alone_targets,
        config,
        step=step,
    )
    return total, {key: value.detach() for key, value in components.items()}
