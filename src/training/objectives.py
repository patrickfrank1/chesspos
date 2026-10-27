from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from src.dataset.token_stream import (
    CLS,
    N_SLOTS,
    PAD,
    SEP,
    SEGMENT_LEN,
    TURN_BLACK,
    TURN_WHITE,
)
from src.modeling.masking import IGNORE_INDEX
from src.modeling.transformer import slot_indices
from src.training.aux_heads import auxiliary_head_losses, segment_global_targets
from src.training.contrastive import build_contrastive_masks, supcon_loss

_WEIGHT_NAMES = (
    "mlm_weight",
    "inv_weight",
    "var_weight",
    "cov_weight",
    "nce_weight",
    "alone_var_weight",
    "alone_cov_weight",
    "material_weight",
    "piece_count_weight",
    "phase_weight",
    "turn_weight",
    "castle_weight",
    "ep_weight",
)


@dataclass(frozen=True)
class AuxiliaryLossConfig:
    """Weights for MLM plus optional geometry and contrastive terms.

    Setting ``nce_weight`` to zero removes the InfoNCE term entirely, so the
    same training entry point can run the MLM-only baseline. The defaults come
    from the tiny objective sweep (2026-10-25): strong VICReg variance plus a
    context-removal invariance term and a small InfoNCE weight.
    """

    mlm_weight: float = 1.0
    inv_weight: float = 0.5
    var_weight: float = 1.0
    cov_weight: float = 0.1
    nce_weight: float = 0.1
    nce_temperature: float = 0.1
    alone_var_weight: float = 1.0
    alone_cov_weight: float = 0.1
    alone_targets_per_window: int = 4
    vicreg_gamma: float = 1.0
    vicreg_eps: float = 1e-4
    warmup_steps: int = 200
    content_pooling: bool = False
    use_supcon: bool = False
    turn_flip: bool = False
    temporal_radius: int = 3
    ignore_opening_negatives: bool = False
    opening_ply_threshold: int = 30
    material_weight: float = 0.0
    piece_count_weight: float = 0.0
    phase_weight: float = 0.0
    turn_weight: float = 0.0
    castle_weight: float = 0.0
    ep_weight: float = 0.0

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
        if self.alone_targets_per_window < 1:
            raise ValueError("alone_targets_per_window must be at least 1")
        if self.temporal_radius < 0:
            raise ValueError("temporal_radius must be non-negative")
        if self.opening_ply_threshold < 0:
            raise ValueError("opening_ply_threshold must be non-negative")
        return

    def auxiliary_enabled(self) -> bool:
        return any(
            weight > 0
            for weight in (
                self.inv_weight,
                self.var_weight,
                self.cov_weight,
                self.nce_weight,
                self.alone_var_weight,
                self.alone_cov_weight,
                self.material_weight,
                self.piece_count_weight,
                self.phase_weight,
                self.turn_weight,
                self.castle_weight,
                self.ep_weight,
            )
        )

    def auxiliary_scale(self, step: int) -> float:
        if self.warmup_steps <= 0:
            return 1.0
        return min(1.0, max(0.0, step / self.warmup_steps))

    def heads_enabled(self) -> bool:
        return any(
            weight > 0
            for weight in (
                self.material_weight,
                self.piece_count_weight,
                self.phase_weight,
                self.turn_weight,
                self.castle_weight,
                self.ep_weight,
            )
        )

    def head_weights(self) -> dict[str, float]:
        return {
            "material": self.material_weight,
            "piece_count": self.piece_count_weight,
            "phase": self.phase_weight,
            "turn": self.turn_weight,
            "castle": self.castle_weight,
            "ep": self.ep_weight,
        }


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

    if config.alone_var_weight > 0 or config.alone_cov_weight > 0:
        if alone_targets is None:
            raise ValueError(
                "alone embeddings are required for alone variance/covariance"
            )
        alone_var_loss, alone_cov_loss = vicreg_terms(
            alone_targets, gamma=config.vicreg_gamma, eps=config.vicreg_eps
        )
        if config.alone_var_weight > 0:
            components["alone_var"] = alone_var_loss
            total = total + config.alone_var_weight * scale * alone_var_loss
        if config.alone_cov_weight > 0:
            components["alone_cov"] = alone_cov_loss
            total = total + config.alone_cov_weight * scale * alone_cov_loss

    if config.inv_weight > 0:
        if window_targets is None or alone_targets is None:
            raise ValueError("paired embeddings are required for invariance")
        inv_loss = alignment_loss(window_targets, alone_targets)
        components["inv"] = inv_loss
        total = total + config.inv_weight * scale * inv_loss

    if config.nce_weight > 0 and not config.use_supcon:
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


def select_target_segments_multi(
    segment_mask: torch.Tensor,
    count: int,
    generator: torch.Generator | None = None,
) -> list[list[int]]:
    """Pick up to ``count`` distinct valid segments per window.

    Returning several targets per window increases the effective number of
    InfoNCE negatives and makes the other targets from the same window hard
    negatives for each other.
    """
    scores = torch.rand(
        segment_mask.shape, generator=generator, device=segment_mask.device
    ).masked_fill(~segment_mask, float("-inf"))
    order = torch.argsort(scores, dim=1, descending=True)
    counts = segment_mask.sum(dim=1).clamp_max(count)
    order_cpu = order.cpu()
    counts_cpu = counts.cpu().tolist()
    selected: list[list[int]] = []
    for row, take in enumerate(counts_cpu):
        if take == 0:
            selected.append([0])
        else:
            selected.append(order_cpu[row, :take].tolist())
    return selected


def restore_visible_tokens(tokens: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Replace masked tokens with their original targets.

    The training window is masked, but the retrieval/evaluation view is fully
    visible. The alone view used by invariance, InfoNCE and alone VICReg must
    therefore be reconstructed without mask tokens.
    """
    visible = tokens.clone()
    mask = targets.ne(IGNORE_INDEX)
    visible[mask] = targets[mask]
    return visible


def flip_turn_tokens(
    tokens: torch.Tensor,
    segment_ids: torch.Tensor,
) -> torch.Tensor:
    """Flip the side-to-move token in every segment, leaving the rest intact."""
    slots = slot_indices(tokens, segment_ids)
    is_turn = (slots == N_SLOTS) & ((tokens == TURN_WHITE) | (tokens == TURN_BLACK))
    flipped = tokens.clone()
    flipped[is_turn] = torch.where(
        tokens[is_turn] == TURN_WHITE, TURN_BLACK, TURN_WHITE
    )
    return flipped


def extract_alone_views_multi(
    tokens: torch.Tensor,
    segment_ids: torch.Tensor,
    padding_mask: torch.Tensor,
    targets_per_row: list[list[int]],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build one-segment packs for multiple target positions per window.

    Returns the packed views plus the window row and target segment id for
    each view, so the caller can gather the matching in-context embeddings.
    """
    view_rows: list[int] = []
    view_targets: list[int] = []
    for row, target_ids in enumerate(targets_per_row):
        for target in target_ids:
            view_rows.append(row)
            view_targets.append(int(target))

    device = tokens.device
    rows = torch.tensor(view_rows, dtype=torch.long, device=device)
    targets = torch.tensor(view_targets, dtype=torch.long, device=device)
    if rows.numel() == 0:
        empty = torch.empty((0, 0), dtype=tokens.dtype, device=device)
        empty_long = torch.empty((0, 0), dtype=torch.long, device=device)
        empty_bool = torch.empty((0, 0), dtype=torch.bool, device=device)
        return empty, empty_long, empty_bool, rows, targets

    row_tokens = tokens[rows]
    row_segments = segment_ids[rows]
    board = (
        padding_mask[rows]
        & row_tokens.ne(CLS)
        & row_tokens.ne(SEP)
        & row_segments.eq(targets.unsqueeze(1))
    )
    lengths = board.sum(dim=1)
    max_len = int(lengths.max().item())
    total_len = max_len + 2
    alone_tokens = torch.full(
        (len(rows), total_len), PAD, dtype=tokens.dtype, device=device
    )
    alone_tokens[:, 0] = CLS
    rank = torch.cumsum(board.long(), dim=1) - 1
    view_index, position_index = board.nonzero(as_tuple=True)
    alone_tokens[view_index, 1 + rank[view_index, position_index]] = row_tokens[
        view_index, position_index
    ]
    view_index = torch.arange(len(rows), device=device)
    alone_tokens[view_index, 1 + lengths] = SEP
    alone_segments = torch.zeros(
        (len(rows), total_len), dtype=torch.long, device=device
    )
    alone_segments[view_index, 1 + lengths] = 1
    alone_padding = torch.arange(total_len, device=device).unsqueeze(0) <= (
        lengths + 1
    ).unsqueeze(1)
    return alone_tokens, alone_segments, alone_padding, rows, targets


def extract_alone_views(
    tokens: torch.Tensor,
    segment_ids: torch.Tensor,
    padding_mask: torch.Tensor,
    target_ids: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build one-segment packs for the target positions, without context."""
    targets_per_row = [[int(target)] for target in target_ids.tolist()]
    alone_tokens, alone_segments, alone_padding, _, _ = extract_alone_views_multi(
        tokens, segment_ids, padding_mask, targets_per_row
    )
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
    if config.content_pooling:
        pooled, segment_mask = encoder.pool_content(
            output["hidden_states"], tokens, segment_ids, padding_mask
        )
    else:
        pooled, segment_mask = encoder.pool_hidden(
            output["hidden_states"], tokens, segment_ids, padding_mask
        )
    embeddings = (
        pooled[segment_mask]
        if (config.var_weight > 0 or config.cov_weight > 0)
        else None
    )

    needs_state = any(
        weight > 0
        for weight in (config.turn_weight, config.castle_weight, config.ep_weight)
    )
    needs_alone = (
        config.inv_weight > 0
        or config.nce_weight > 0
        or config.alone_var_weight > 0
        or config.alone_cov_weight > 0
        or config.turn_flip
        or config.heads_enabled()
    )
    window_targets = None
    alone_targets = None
    alone_state = None
    alone_tokens = None
    alone_segments = None
    alone_padding = None
    view_rows = None
    view_targets = None
    if needs_alone:
        visible_tokens = restore_visible_tokens(tokens, batch["targets"])
        targets_per_row = select_target_segments_multi(
            segment_mask,
            count=config.alone_targets_per_window,
            generator=generator,
        )
        (
            alone_tokens,
            alone_segments,
            alone_padding,
            view_rows,
            view_targets,
        ) = extract_alone_views_multi(
            visible_tokens, segment_ids, padding_mask, targets_per_row
        )
        use_content_views = (
            config.content_pooling
            or config.use_supcon
            or config.turn_flip
            or config.heads_enabled()
        )
        if use_content_views:
            alone_content, alone_state, _ = encoder.encode_views(
                alone_tokens, alone_segments, alone_padding
            )
            alone_targets = alone_content[:, 0]
            alone_state = alone_state[:, 0] if needs_state else None
        else:
            alone_pooled, _ = encoder.encode(
                alone_tokens, alone_segments, alone_padding
            )
            alone_targets = alone_pooled[:, 0]
        window_targets = pooled[view_rows, view_targets]
        raw_segments = alone_tokens[:, 1 : 1 + SEGMENT_LEN].long()
        global_targets = segment_global_targets(raw_segments)

    total, components = auxiliary_losses(
        mlm_loss,
        embeddings,
        window_targets,
        alone_targets,
        config,
        step=step,
    )
    scale = config.auxiliary_scale(step)

    if config.use_supcon and config.nce_weight > 0:
        if alone_targets is None or alone_tokens is None:
            raise ValueError("SupCon requires alone views")
        rows_tensor = view_rows.detach().cpu()
        targets_tensor = view_targets.detach().cpu()
        rows = rows_tensor.tolist()
        targets = targets_tensor.tolist()
        game_ids = batch.get("game_id")
        start_index = batch.get("start_index")
        if game_ids is not None and start_index is not None:
            view_games = game_ids.detach().cpu()[rows_tensor].tolist()
            view_plies = (
                start_index.detach().cpu()[rows_tensor] + targets_tensor
            ).tolist()
        else:
            view_games = rows
            view_plies = targets
        opening = None
        if config.ignore_opening_negatives:
            opening = [bool(ply < config.opening_ply_threshold) for ply in view_plies]
        positive, ignore = build_contrastive_masks(
            view_games,
            view_plies,
            temporal_radius=config.temporal_radius,
            opening=opening,
        )
        positive = positive.to(alone_targets.device)
        ignore = ignore.to(alone_targets.device)
        embeddings_for_supcon = alone_targets
        if config.turn_flip:
            flipped_tokens = flip_turn_tokens(alone_tokens, alone_segments)
            flipped_content, _, _ = encoder.encode_views(
                flipped_tokens, alone_segments, alone_padding
            )
            flipped_targets = flipped_content[:, 0]
            embeddings_for_supcon = torch.cat([alone_targets, flipped_targets], dim=0)
            eye = torch.eye(
                alone_targets.shape[0],
                dtype=torch.bool,
                device=alone_targets.device,
            )
            cross = positive | eye
            positive = torch.cat(
                [
                    torch.cat([positive, cross], dim=1),
                    torch.cat([cross, positive], dim=1),
                ],
                dim=0,
            )
            ignore = torch.cat(
                [
                    torch.cat([ignore, ignore], dim=1),
                    torch.cat([ignore, ignore], dim=1),
                ],
                dim=0,
            )
        supcon = supcon_loss(
            embeddings_for_supcon,
            positive,
            ignore_mask=ignore,
            temperature=config.nce_temperature,
        )
        components["supcon"] = supcon
        total = total + config.nce_weight * scale * supcon

    if config.heads_enabled():
        if encoder.aux_heads is None:
            raise ValueError("model was built without auxiliary heads")
        if alone_targets is None or alone_tokens is None:
            raise ValueError("auxiliary heads require alone views")
        aux_total, aux_components = auxiliary_head_losses(
            alone_targets,
            alone_state,
            global_targets,
            encoder.aux_heads,
            config.head_weights(),
        )
        components.update(aux_components)
        total = total + scale * aux_total

    return total, {key: value.detach() for key, value in components.items()}
