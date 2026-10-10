from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.dataset.token_stream import (
    CASTLE_BASE,
    CLS,
    EP_BASE,
    MASK,
    PIECE_SQUARE_BASE,
    SEP,
    TURN_BLACK,
    TURN_WHITE,
    VOCAB_SIZE,
)

IGNORE_INDEX = -100


@dataclass
class MaskingConfig:
    p_random: float = 0.5
    p_board: float = 0.25
    p_span: float = 0.25
    random_rate: float = 0.15
    mask_replace_prob: float = 0.8
    random_replace_prob: float = 0.1
    min_span: int = 2
    max_span: int = 4


TOKEN_CLASS_RANGES = (
    (PIECE_SQUARE_BASE, VOCAB_SIZE),
    (EP_BASE, EP_BASE + 16),
    (CASTLE_BASE, CASTLE_BASE + 16),
    (TURN_WHITE, TURN_BLACK + 1),
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
    tokens: np.ndarray, config: MaskingConfig, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray]:
    tokens = np.asarray(tokens)
    masked = tokens.copy()
    targets = np.full(tokens.shape, IGNORE_INDEX, dtype=np.int16)

    bounds = segment_bounds(tokens)
    if not bounds:
        return masked, targets

    u = rng.random()
    if u < config.p_random:
        positions = _random_mode(tokens, bounds, config, rng)
    elif u < config.p_random + config.p_board:
        positions = _board_mode(bounds, rng)
    else:
        positions = _span_mode(bounds, config, rng)

    for pos in positions:
        original = int(tokens[pos])
        targets[pos] = original
        r = rng.random()
        if r < config.mask_replace_prob:
            masked[pos] = MASK
        elif r < config.mask_replace_prob + config.random_replace_prob:
            low, high = token_class_range(original)
            masked[pos] = rng.integers(low, high)

    return masked, targets


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


def _board_mode(bounds: list[tuple[int, int]], rng: np.random.Generator) -> list[int]:
    start, end = bounds[int(rng.integers(len(bounds)))]
    return list(range(start, end))


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
