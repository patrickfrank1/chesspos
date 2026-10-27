from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from src.dataset.token_stream import (
    CASTLE_BASE,
    EP_BASE,
    N_SLOTS,
    PIECE_BASE,
    SEGMENT_LEN,
    TURN_BLACK,
    TURN_WHITE,
)

_PIECE_VALUES = torch.tensor([1.0, 3.0, 3.0, 5.0, 9.0, 0.0] * 2)
_PIECE_SIGNS = torch.tensor([1.0] * 6 + [-1.0] * 6)


def segment_global_targets(tokens: torch.Tensor) -> dict[str, torch.Tensor]:
    """Recover global board targets directly from raw 67-token segments."""
    if tokens.ndim == 1:
        tokens = tokens.unsqueeze(0)
    if tokens.ndim != 2 or tokens.shape[1] < SEGMENT_LEN:
        raise ValueError("tokens must have shape (N, SEGMENT_LEN)")
    tokens = tokens[:, :SEGMENT_LEN].long()
    device = tokens.device

    piece_mask = (tokens >= PIECE_BASE) & (tokens < TURN_WHITE)
    piece_index = (tokens - PIECE_BASE).clamp(0, 11)
    values = _PIECE_VALUES.to(device)[piece_index]
    signs = _PIECE_SIGNS.to(device)[piece_index]
    material = (values * signs * piece_mask).sum(dim=1)
    piece_count = piece_mask.sum(dim=1).to(torch.float32)
    phase = torch.where(
        piece_count >= 24,
        torch.zeros_like(piece_count, dtype=torch.long),
        torch.where(
            piece_count >= 14,
            torch.ones_like(piece_count, dtype=torch.long),
            torch.full_like(piece_count, 2, dtype=torch.long),
        ),
    )
    turn = (tokens[:, N_SLOTS] == TURN_BLACK).long()
    castle = (tokens[:, N_SLOTS + 1] - CASTLE_BASE).long()
    ep = (tokens[:, N_SLOTS + 2] - EP_BASE).long()
    return {
        "material": material,
        "piece_count": piece_count,
        "phase": phase,
        "turn": turn,
        "castle": castle,
        "ep": ep,
    }


class AuxiliaryHeads(nn.Module):
    """Small probes attached to the content and state embeddings.

    Content heads (material, piece count, phase) shape the retrieval vector.
    State heads (turn, castling, en passant) only see the structural-token pool,
    so they cannot reintroduce the turn shortcut into the content representation.
    """

    def __init__(self, d_model: int, phase_classes: int = 3):
        super().__init__()
        self.content = nn.ModuleDict(
            {
                "material": nn.Linear(d_model, 1),
                "piece_count": nn.Linear(d_model, 1),
                "phase": nn.Linear(d_model, phase_classes),
            }
        )
        self.state = nn.ModuleDict(
            {
                "turn": nn.Linear(d_model, 2),
                "castle": nn.Linear(d_model, 16),
                "ep": nn.Linear(d_model, 17),
            }
        )
        self.phase_classes = phase_classes
        return

    def forward_content(self, embeddings: torch.Tensor) -> dict[str, torch.Tensor]:
        return {name: head(embeddings) for name, head in self.content.items()}

    def forward_state(self, embeddings: torch.Tensor) -> dict[str, torch.Tensor]:
        return {name: head(embeddings) for name, head in self.state.items()}


def auxiliary_head_losses(
    content: torch.Tensor | None,
    state: torch.Tensor | None,
    targets: dict[str, torch.Tensor],
    heads: AuxiliaryHeads,
    weights: dict[str, float],
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Weighted multi-task losses for the content and state probes."""
    components: dict[str, torch.Tensor] = {}
    total = content.sum() * 0.0 if content is not None else state.sum() * 0.0

    content_weights = ("material", "piece_count", "phase")
    if any(weights.get(name, 0.0) > 0 for name in content_weights) and content is None:
        raise ValueError("content embeddings are required for content heads")
    state_weights = ("turn", "castle", "ep")
    if any(weights.get(name, 0.0) > 0 for name in state_weights) and state is None:
        raise ValueError("state embeddings are required for state heads")

    content_out = heads.forward_content(content) if content is not None else {}
    state_out = heads.forward_state(state) if state is not None else {}

    if weights.get("material", 0.0) > 0:
        loss = F.mse_loss(
            content_out["material"].squeeze(-1), targets["material"] / 39.0
        )
        components["aux_material"] = loss
        total = total + weights["material"] * loss
    if weights.get("piece_count", 0.0) > 0:
        loss = F.mse_loss(
            content_out["piece_count"].squeeze(-1), targets["piece_count"] / 32.0
        )
        components["aux_piece_count"] = loss
        total = total + weights["piece_count"] * loss
    if weights.get("phase", 0.0) > 0:
        loss = F.cross_entropy(content_out["phase"], targets["phase"])
        components["aux_phase"] = loss
        total = total + weights["phase"] * loss
    if weights.get("turn", 0.0) > 0:
        loss = F.cross_entropy(state_out["turn"], targets["turn"])
        components["aux_turn"] = loss
        total = total + weights["turn"] * loss
    if weights.get("castle", 0.0) > 0:
        loss = F.cross_entropy(state_out["castle"], targets["castle"])
        components["aux_castle"] = loss
        total = total + weights["castle"] * loss
    if weights.get("ep", 0.0) > 0:
        loss = F.cross_entropy(state_out["ep"], targets["ep"])
        components["aux_ep"] = loss
        total = total + weights["ep"] * loss
    return total, components
