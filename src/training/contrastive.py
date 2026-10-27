from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn.functional as F


def supcon_loss(
    embeddings: torch.Tensor,
    positive_mask: torch.Tensor,
    ignore_mask: torch.Tensor | None = None,
    temperature: float = 0.1,
) -> torch.Tensor:
    """Multi-positive contrastive loss over a batch of embeddings.

    ``positive_mask[i, j]`` marks j as a positive for anchor i. ``ignore_mask``
    marks pairs that are neither positives nor negatives; they are excluded
    from the denominator so same-game neighbours cannot act as false negatives.
    Anchors without a positive contribute nothing.
    """
    if embeddings.ndim != 2:
        raise ValueError("embeddings must be two-dimensional")
    if positive_mask.shape != (len(embeddings), len(embeddings)):
        raise ValueError("positive_mask must be square and match embeddings")
    if ignore_mask is not None and ignore_mask.shape != positive_mask.shape:
        raise ValueError("ignore_mask must match positive_mask")
    if temperature <= 0:
        raise ValueError("temperature must be positive")

    z = F.normalize(embeddings.float(), dim=-1)
    logits = z @ z.T / temperature
    n = len(z)
    identity = torch.eye(n, dtype=torch.bool, device=z.device)
    ignore = identity if ignore_mask is None else identity | ignore_mask.bool()
    positives = positive_mask.bool() & ~ignore

    logits = logits.masked_fill(ignore, float("-inf"))
    rows = positives.any(dim=1)
    if not bool(rows.any()):
        return embeddings.sum() * 0.0

    selected = torch.nonzero(rows, as_tuple=False).flatten()
    denominator = torch.logsumexp(logits[selected], dim=1)
    positive_counts = positives.sum(dim=1).clamp_min(1)
    positive_sum = logits.masked_fill(~positives, 0.0).sum(dim=1)
    mean_positive = (positive_sum / positive_counts)[selected]
    return -(mean_positive - denominator).mean()


def build_contrastive_masks(
    game_ids: Sequence[object],
    ply_indices: Sequence[int],
    temporal_radius: int = 3,
    opening: Sequence[bool] | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build positive and ignored-pair masks for contrastive views.

    Pairs from the same game within ``temporal_radius`` plies are positives.
    Same-game pairs farther apart than the radius are ignored, so they never
    become false negatives. Pairs from different games are ordinary negatives,
    except that any pair involving an ``opening`` view is ignored: opening
    positions are too similar across games to be useful negatives.
    """
    if len(game_ids) != len(ply_indices):
        raise ValueError("game_ids and ply_indices must have the same length")
    if opening is not None and len(opening) != len(game_ids):
        raise ValueError("opening must match game_ids")
    if temporal_radius < 0:
        raise ValueError("temporal_radius must be non-negative")

    labels: dict[object, int] = {}
    game_index = torch.tensor(
        [labels.setdefault(game, len(labels)) for game in game_ids],
        dtype=torch.long,
    )
    plies = torch.as_tensor(ply_indices, dtype=torch.long)
    same_game = game_index[:, None] == game_index[None, :]
    distance = (plies[:, None] - plies[None, :]).abs()
    positive = same_game & (distance <= temporal_radius)
    ignore = same_game & (distance > temporal_radius)
    if opening is not None:
        is_opening = torch.as_tensor(opening, dtype=torch.bool)
        ignore = ignore | ((~same_game) & (is_opening[:, None] | is_opening[None, :]))
    positive.fill_diagonal_(False)
    ignore.fill_diagonal_(False)
    return positive, ignore
