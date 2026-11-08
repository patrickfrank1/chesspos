from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from src.evaluation.observability.samples import (
    SURROGATE_KEYS,
    PositionSample,
    surrogate_matrix,
)


@dataclass(frozen=True)
class SurrogateTask:
    name: str
    key: str
    label: str
    classes: tuple[str, ...]
    description: str
    class_definitions: tuple[tuple[str, str], ...] = ()


SURROGATE_TASKS = (
    SurrogateTask(
        "material",
        "material",
        "Material balance",
        ("white_advantage", "equal", "black_advantage"),
        "Material balance in piece points (pawn=1, knight/bishop=3, rook=5, queen=9).",
        (
            ("white_advantage", "material > +2"),
            ("equal", "material between -2 and +2"),
            ("black_advantage", "material < -2"),
        ),
    ),
    SurrogateTask(
        "phase",
        "phase",
        "Game phase",
        ("opening", "middlegame", "endgame"),
        "Game phase from the total number of pieces on the board.",
        (
            ("opening", ">= 24 pieces"),
            ("middlegame", "14-23 pieces"),
            ("endgame", "< 14 pieces"),
        ),
    ),
    SurrogateTask(
        "open_files",
        "open_files",
        "Open files",
        ("none", "some"),
        "Files with no pawns of either colour.",
        (("none", "0 open files"), ("some", ">= 1 open file")),
    ),
    SurrogateTask(
        "isolated_pawns",
        "isolated_pawns",
        "Isolated pawns",
        ("none", "some"),
        "Pawns with no friendly pawn on an adjacent file.",
        (("none", "0 isolated pawns"), ("some", ">= 1 isolated pawn")),
    ),
    SurrogateTask(
        "passed_pawns",
        "passed_pawns",
        "Passed pawns",
        ("none", "some"),
        "Pawns with no enemy pawn on the same or adjacent files ahead of them.",
        (("none", "0 passed pawns"), ("some", ">= 1 passed pawn")),
    ),
    SurrogateTask(
        "doubled_pawns",
        "doubled_pawns",
        "Doubled pawns",
        ("none", "some"),
        "Extra pawns on a file beyond the first.",
        (("none", "0 doubled pawns"), ("some", ">= 1 doubled pawn")),
    ),
    SurrogateTask(
        "pawn_islands",
        "pawn_islands",
        "Pawn islands",
        ("low", "mid", "high"),
        "Contiguous groups of friendly pawn files, summed over both colours.",
        (("low", "<= 2 islands"), ("mid", "3-4 islands"), ("high", ">= 5 islands")),
    ),
    SurrogateTask(
        "king_safety",
        "king_safety",
        "King safety",
        ("low", "mid", "high"),
        "Minimum pawn-shield count around either king.",
        (("low", "<= 1 shield pawn"), ("mid", "2-3 shield pawns"), ("high", ">= 4")),
    ),
    SurrogateTask(
        "mobility",
        "mobility",
        "Mobility",
        ("low", "mid", "high"),
        "Number of legal moves for the side to move.",
        (("low", "<= 20 moves"), ("mid", "21-35 moves"), ("high", ">= 36 moves")),
    ),
    SurrogateTask(
        "center_control",
        "center_control",
        "Centre control",
        ("low", "mid", "high"),
        "Central squares (d4, e4, d5, e5) attacked by either side.",
        (("low", "<= 1 square"), ("mid", "2-3 squares"), ("high", "4 squares")),
    ),
    SurrogateTask(
        "bishop_pair",
        "bishop_pair",
        "Bishop pair",
        ("none", "yes"),
        "Whether either side still has both bishops.",
        (("none", "no bishop pair"), ("yes", "at least one bishop pair")),
    ),
    SurrogateTask(
        "in_check",
        "is_check",
        "In check",
        ("no", "yes"),
        "Whether the side to move is in check.",
        (("no", "not in check"), ("yes", "in check")),
    ),
    SurrogateTask(
        "king_attack",
        "king_attack",
        "King attack",
        ("low", "mid", "high"),
        "Maximum number of squares around either king attacked by the opponent.",
        (("low", "0 attacked squares"), ("mid", "1-2"), ("high", ">= 3")),
    ),
    SurrogateTask(
        "knight_on_edge",
        "knights_on_edge",
        "Knight on a/h file",
        ("none", "some"),
        "Whether any knight is on the a-file or h-file.",
        (("none", "0 edge knights"), ("some", ">= 1 edge knight")),
    ),
    SurrogateTask(
        "inactive_bishop",
        "inactive_bishops",
        "Inactive bishop",
        ("none", "some"),
        "Whether any bishop has at most two pseudo-legal moves.",
        (("none", "no inactive bishop"), ("some", ">= 1 inactive bishop")),
    ),
    SurrogateTask(
        "smothered",
        "smothered_king",
        "Smothered position",
        ("none", "yes"),
        "Whether the side to move is in check and has no legal king move.",
        (("none", "not smothered"), ("yes", "in check with no king move")),
    ),
)


def _bucket(value: float, low: float, high: float) -> str:
    if value <= low:
        return "low"
    if value <= high:
        return "mid"
    return "high"


def classify_surrogates(surrogates: dict[str, float | int]) -> dict[str, str]:
    material = float(surrogates.get("material", 0.0))
    phase = int(surrogates.get("phase", 0))
    return {
        "material": (
            "white_advantage"
            if material > 2
            else "black_advantage"
            if material < -2
            else "equal"
        ),
        "phase": ("opening", "middlegame", "endgame")[min(2, max(0, phase))],
        "open_files": "none" if float(surrogates.get("open_files", 0)) <= 0 else "some",
        "isolated_pawns": "none"
        if float(surrogates.get("isolated_pawns", 0)) <= 0
        else "some",
        "passed_pawns": "none"
        if float(surrogates.get("passed_pawns", 0)) <= 0
        else "some",
        "doubled_pawns": "none"
        if float(surrogates.get("doubled_pawns", 0)) <= 0
        else "some",
        "pawn_islands": _bucket(float(surrogates.get("pawn_islands", 0)), 2, 4),
        "king_safety": _bucket(float(surrogates.get("king_safety", 0)), 1, 3),
        "mobility": _bucket(float(surrogates.get("mobility", 0)), 20, 35),
        "center_control": _bucket(float(surrogates.get("center_control", 0)), 1, 3),
        "bishop_pair": "yes" if float(surrogates.get("bishop_pair", 0)) > 0 else "none",
        "in_check": "yes" if float(surrogates.get("is_check", 0)) > 0 else "no",
        "king_attack": _bucket(float(surrogates.get("king_attack", 0)), 0, 2),
        "knight_on_edge": "none"
        if float(surrogates.get("knights_on_edge", 0)) <= 0
        else "some",
        "inactive_bishop": "none"
        if float(surrogates.get("inactive_bishops", 0)) <= 0
        else "some",
        "smothered": "yes"
        if float(surrogates.get("smothered_king", 0)) > 0
        else "none",
    }


def sample_surrogate_classes(sample: PositionSample) -> dict[str, str]:
    return classify_surrogates(sample.surrogates)


def surrogate_class_labels(
    samples: Sequence[PositionSample], task: SurrogateTask
) -> np.ndarray:
    return np.array(
        [classify_surrogates(sample.surrogates)[task.name] for sample in samples],
        dtype=object,
    )


def matching_matrix(
    samples: Sequence[PositionSample], *, exclude_key: str | None = None
) -> np.ndarray:
    matrix = surrogate_matrix(samples)
    if exclude_key is not None:
        keep = [index for index, key in enumerate(SURROGATE_KEYS) if key != exclude_key]
        if keep:
            matrix = matrix[:, keep]
    return matrix
