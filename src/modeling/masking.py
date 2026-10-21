from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from src.dataset.token_stream import (
    CASTLE_BASE,
    CLS,
    EMPTY,
    EP_BASE,
    MASK,
    SEP,
    TURN_WHITE,
    VOCAB_SIZE,
)

IGNORE_INDEX = -100


@dataclass
class MaskingConfig:
    p_random: float = 0.45
    p_board: float = 0.30
    p_span: float = 0.25
    random_rate: float = 0.15
    mask_replace_prob: float = 0.8
    random_replace_prob: float = 0.1
    min_span: int = 2
    max_span: int = 4
    board_span: int = 4
    min_context_segments: int = 8
    mode_weights: dict[str, float] = field(
        default_factory=lambda: {"random": 1.0, "board": 2.0, "span": 2.0}
    )


TOKEN_CLASS_RANGES = (
    (EMPTY, TURN_WHITE),
    (TURN_WHITE, CASTLE_BASE),
    (CASTLE_BASE, EP_BASE),
    (EP_BASE, VOCAB_SIZE),
)


def token_class_range(token: int) -> tuple[int, int]:
    for low, high in TOKEN_CLASS_RANGES:
        if low <= token < high:
            return low, high
    raise ValueError(f"Token {token} cannot be masked")


def segment_bounds(tokens: np.ndarray) -> list[tuple[int, int]]:
    sep_positions = np.flatnonzero(tokens == SEP)
    start = 1 if tokens.size and tokens[0] == CLS else 0
    bounds = []
    for sep in sep_positions:
        if start < sep:
            bounds.append((int(start), int(sep)))
        start = sep + 1
    if start < tokens.size:
        bounds.append((int(start), int(tokens.size)))
    return bounds


def mask_stream(
    tokens: np.ndarray,
    config: MaskingConfig,
    rng: np.random.Generator,
    mode: str | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    tokens = np.asarray(tokens)
    masked = tokens.copy()
    targets = np.full(tokens.shape, IGNORE_INDEX, dtype=np.int16)
    weights = np.zeros(tokens.shape, dtype=np.float32)

    bounds = segment_bounds(tokens)
    if not bounds:
        return masked, targets, weights

    if mode is None:
        mode = _draw_mode(config, rng)
    if mode in ("board", "span") and len(bounds) < config.min_context_segments:
        mode = "random"

    if mode == "random":
        positions = _random_mode(tokens, bounds, config, rng)
    elif mode == "board":
        positions = _board_mode(bounds, config, rng)
    else:
        positions = _span_mode(bounds, config, rng)

    for pos in positions:
        original = int(tokens[pos])
        targets[pos] = original
        weights[pos] = config.mode_weights[mode]
        r = rng.random()
        if r < config.mask_replace_prob:
            masked[pos] = MASK
        elif r < config.mask_replace_prob + config.random_replace_prob:
            low, high = token_class_range(original)
            masked[pos] = rng.integers(low, high)

    return masked, targets, weights


def _draw_mode(config: MaskingConfig, rng: np.random.Generator) -> str:
    u = rng.random()
    if u < config.p_random:
        return "random"
    if u < config.p_random + config.p_board:
        return "board"
    return "span"


def _random_mode(
    tokens: np.ndarray,
    bounds: list[tuple[int, int]],
    config: MaskingConfig,
    rng: np.random.Generator,
) -> list[int]:
    candidates = [pos for start, end in bounds for pos in range(start, end)]
    if not candidates:
        return []
    n = max(1, int(round(config.random_rate * len(candidates))))
    n = min(n, len(candidates))
    chosen = rng.choice(len(candidates), size=n, replace=False)
    return [candidates[i] for i in chosen]


def _board_mode(
    bounds: list[tuple[int, int]], config: MaskingConfig, rng: np.random.Generator
) -> list[int]:
    n = min(config.board_span, len(bounds))
    start = int(rng.integers(len(bounds) - n + 1))
    positions = []
    for i in range(start, start + n):
        seg_start, seg_end = bounds[i]
        positions.extend(range(seg_start, seg_end))
    return positions


def _span_mode(
    bounds: list[tuple[int, int]],
    config: MaskingConfig,
    rng: np.random.Generator,
) -> list[int]:
    max_span = min(config.max_span, len(bounds))
    min_span = min(config.min_span, max_span)
    span = int(rng.integers(min_span, max_span + 1))
    start_idx = int(rng.integers(0, len(bounds) - span + 1))
    positions = []
    for i in range(start_idx, start_idx + span):
        start, end = bounds[i]
        positions.extend(range(start, end))
    return positions
