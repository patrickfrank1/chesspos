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
    p_random: float = 0.0
    p_board: float = 0.0
    p_span: float = 0.0
    p_rect: float = 1.0
    p_skip_gram: float = 0.0
    random_rate: float = 0.15
    mask_replace_prob: float = 0.8
    random_replace_prob: float = 0.1
    min_span: int = 2
    max_span: int = 4
    board_span: int = 4
    rect_min_rate: float = 0.30
    rect_max_rate: float = 0.50
    rect_min_side: int = 2
    rect_max_side: int = 4
    rect_synchronize: bool = True
    skip_gram_unmask_rate: float = 0.10
    min_context_segments: int = 8
    mode_weights: dict[str, float] = field(
        default_factory=lambda: {
            "random": 1.0,
            "board": 2.0,
            "span": 2.0,
            "rect": 1.0,
            "skip_gram": 1.0,
        }
    )

    def __post_init__(self):
        if not 0 < self.rect_min_rate <= self.rect_max_rate < 1:
            raise ValueError("rectangle mask rates must satisfy 0 < min <= max < 1")
        if not 1 <= self.rect_min_side <= self.rect_max_side <= 8:
            raise ValueError("rectangle sides must satisfy 1 <= min <= max <= 8")
        if not 0 < self.skip_gram_unmask_rate < 0.30:
            raise ValueError("skip_gram_unmask_rate must satisfy 0 < rate < 0.30")
        return


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
    if mode in ("board", "span", "skip_gram") and (
        len(bounds) < config.min_context_segments
    ):
        mode = "random"

    if mode == "random":
        positions = _random_mode(tokens, bounds, config, rng)
    elif mode == "board":
        positions = _board_mode(bounds, config, rng)
    elif mode == "span":
        positions = _span_mode(bounds, config, rng)
    elif mode == "rect":
        positions = _rect_mode(bounds, config, rng)
    else:
        positions = _skip_gram_mode(bounds, config, rng)

    for pos in positions:
        original = int(tokens[pos])
        targets[pos] = original
        weights[pos] = config.mode_weights.get(mode, 1.0)
        r = rng.random()
        if r < config.mask_replace_prob:
            masked[pos] = MASK
        elif r < config.mask_replace_prob + config.random_replace_prob:
            low, high = token_class_range(original)
            masked[pos] = rng.integers(low, high)

    return masked, targets, weights


def _draw_mode(config: MaskingConfig, rng: np.random.Generator) -> str:
    modes = (
        ("random", config.p_random),
        ("board", config.p_board),
        ("span", config.p_span),
        ("rect", config.p_rect),
        ("skip_gram", config.p_skip_gram),
    )
    total = sum(weight for _, weight in modes)
    if total <= 0:
        return "random"
    u = rng.random() * total
    for mode, weight in modes:
        if u < weight:
            return mode
        u -= weight
    return "rect"


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


def _sample_rect_mask(config: MaskingConfig, rng: np.random.Generator) -> np.ndarray:
    """Sample a per-board boolean mask covering 30-50% of the 64 squares."""
    rate = float(rng.uniform(config.rect_min_rate, config.rect_max_rate))
    target = max(1, min(64, int(round(rate * 64))))
    mask = np.zeros(64, dtype=bool)
    attempts = 0
    while int(mask.sum()) < target and attempts < 200:
        attempts += 1
        height = min(
            8, int(rng.integers(config.rect_min_side, config.rect_max_side + 1))
        )
        width = min(
            8, int(rng.integers(config.rect_min_side, config.rect_max_side + 1))
        )
        row = int(rng.integers(0, 8 - height + 1))
        column = int(rng.integers(0, 8 - width + 1))
        reached = False
        for square_row in range(row, row + height):
            start = square_row * 8 + column
            for square in range(start, start + width):
                if int(mask.sum()) >= target:
                    reached = True
                    break
                mask[square] = True
            if reached:
                break
    return mask


def _skip_gram_mode(
    bounds: list[tuple[int, int]],
    config: MaskingConfig,
    rng: np.random.Generator,
) -> list[int]:
    """Mask whole boards except a small visible fraction.

    The visible boards are chosen at random; every token of the remaining
    boards (including turn/castling/en-passant) becomes an MLM target. This
    forces the model to reconstruct boards across the window instead of
    copying them from a near-identical neighbour.
    """
    count = len(bounds)
    if count <= 1:
        return []
    visible = max(
        1,
        min(count - 1, int(config.skip_gram_unmask_rate * count)),
    )
    visible_indices = set(rng.choice(count, size=visible, replace=False).tolist())
    positions: list[int] = []
    for index, (start, end) in enumerate(bounds):
        if index in visible_indices:
            continue
        positions.extend(range(start, end))
    return positions


def _rect_mode(
    bounds: list[tuple[int, int]],
    config: MaskingConfig,
    rng: np.random.Generator,
) -> list[int]:
    """Mask board rectangles, optionally the same squares in every segment.

    Synchronizing the square set across segments prevents the model from
    copying the masked squares out of an adjacent (near-identical) position.
    The three state tokens (turn/castling/en-passant) are left unmasked.
    """
    if config.rect_synchronize:
        masks = [_sample_rect_mask(config, rng)] * len(bounds)
    else:
        masks = [_sample_rect_mask(config, rng) for _ in bounds]
    positions: list[int] = []
    for (start, end), mask in zip(bounds, masks):
        board_len = min(64, end - start)
        for square in np.flatnonzero(mask[:board_len]):
            positions.append(start + int(square))
    return positions
