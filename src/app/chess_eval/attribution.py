from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Mapping, Sequence

import chess
import numpy as np
import torch

from src.app.chess_eval.backend import (
    ModelBatch,
    ModelHandle,
    forward_tokens,
    model_batch,
)
from src.dataset.token_stream import EMPTY, MASK, TURN_WHITE, piece_id

METHODS = ("occlusion", "shap")
OCCLUSION_MODES = ("mask", "empty")
DEFAULT_SHAP_SAMPLES = 128

_OCCLUSION_REPLACEMENTS = {"mask": MASK, "empty": EMPTY}
_KING_TOKENS = frozenset(
    {
        piece_id(chess.Piece(chess.KING, chess.WHITE)),
        piece_id(chess.Piece(chess.KING, chess.BLACK)),
    }
)


def occlusion_replacement(occlusion: str) -> int:
    """Token used to replace an occluded square."""
    if occlusion not in _OCCLUSION_REPLACEMENTS:
        raise ValueError(
            f"occlusion must be one of {OCCLUSION_MODES}, got {occlusion!r}"
        )
    return _OCCLUSION_REPLACEMENTS[occlusion]


ScoreBatch = Callable[[np.ndarray], np.ndarray]
ValueFunction = Callable[[np.ndarray], np.ndarray]


@dataclass(frozen=True)
class AttributionResult:
    """Per-square and per-position attributions for one model output.

    Signs follow the output space: a positive value means the square (or
    position) pushed the model output up. For the sacrifice classifier that is
    probability; for the score probe it is the White-POV score, so a Black
    piece with a positive value is one that helps White.

    ``squares`` always describes the current (last) board. ``segment_squares``
    holds every position in the model's window, oldest first.
    """

    task: str
    method: str
    model_value: float
    baseline_value: float
    squares: tuple[float, ...]
    segment_squares: tuple[tuple[float, ...], ...]
    segment_totals: tuple[float, ...]
    piece_groups: Mapping[str, float] = field(default_factory=dict)
    piece_names: tuple[str | None, ...] = ()
    occlusion: str | None = None
    samples: int | None = None
    background: int | None = None
    seed: int | None = None

    def ranked(self, limit: int = 8) -> list[dict]:
        """Return the highest-impact squares, largest absolute value first."""
        order = np.argsort(-np.abs(np.asarray(self.squares)))[: max(0, limit)]
        return [
            {
                "square": chess.square_name(int(square)),
                "piece": self.piece_names[int(square)]
                if int(square) < len(self.piece_names)
                else None,
                "value": float(self.squares[int(square)]),
            }
            for square in order
        ]

    def to_dict(self, top: int = 8) -> dict:
        return {
            "task": self.task,
            "method": self.method,
            "model_value": self.model_value,
            "baseline_value": self.baseline_value,
            "squares": [float(value) for value in self.squares],
            "segment_squares": [
                [float(value) for value in segment] for segment in self.segment_squares
            ],
            "segment_totals": [float(value) for value in self.segment_totals],
            "piece_groups": {
                name: float(value) for name, value in self.piece_groups.items()
            },
            "top": self.ranked(top),
            "occlusion": self.occlusion,
            "samples": self.samples,
            "background": self.background,
            "seed": self.seed,
        }


def occlusion_deltas(
    score_batch: ScoreBatch,
    base_tokens: np.ndarray,
    positions: np.ndarray,
    replacement: int,
    batch_size: int = 256,
) -> tuple[float, np.ndarray]:
    """Leave-one-out deltas for square positions, batched into few forwards.

    ``positions`` is ``[n_segments, 64]`` (or any 2-D array) of token indices.
    ``deltas[i, j] = f(base) - f(base with token positions[i, j] replaced)``.
    """
    base = _as_batch_row(base_tokens)
    positions = np.asarray(positions, dtype=np.int64)
    if positions.ndim == 1:
        positions = positions[None, :]
    replacement = int(replacement)
    if batch_size < 1:
        raise ValueError("batch_size must be positive")

    base_value = float(score_batch(base)[0])
    flat = positions.ravel()
    rows = np.repeat(base, flat.size, axis=0)
    for index, position in enumerate(flat):
        rows[index, int(position)] = replacement
    values = _batched(score_batch, rows, batch_size)
    deltas = (base_value - values).reshape(positions.shape)
    return base_value, deltas


def kernel_shap(
    value_fn: ValueFunction,
    n_players: int,
    nsamples: int = DEFAULT_SHAP_SAMPLES,
    seed: int = 0,
) -> np.ndarray:
    """KernelSHAP values over ``n_players`` binary features.

    ``value_fn`` receives a ``[m, n_players]`` boolean coalition matrix and
    returns ``[m]`` outputs. The empty and full coalitions are always sampled;
    the remaining coalitions are weighted by the Shapley kernel and fit with
    weighted least squares. Values are finally shifted so they sum exactly to
    ``v(full) - v(empty)``.
    """
    if n_players < 1:
        raise ValueError("n_players must be positive")
    if nsamples < 2:
        raise ValueError("nsamples must be at least 2")

    rng = np.random.default_rng(seed)
    coalitions = rng.integers(0, 2, size=(nsamples, n_players)).astype(bool)
    coalitions[0] = False
    coalitions[1] = True
    values = np.asarray(value_fn(coalitions), dtype=np.float64).reshape(-1)
    if values.size != nsamples:
        raise ValueError("value_fn must return one value per coalition")

    empty = float(values[0])
    full = float(values[1])
    target = full - empty
    sizes = coalitions.sum(axis=1)
    interior = (sizes > 0) & (sizes < n_players)
    if not interior.any():
        return np.full(n_players, target / n_players, dtype=np.float64)

    design = coalitions[interior].astype(np.float64)
    weights = _shapley_kernel(n_players, sizes[interior])
    root_weights = np.sqrt(weights)
    response = values[interior] - empty
    coefficients, *_ = np.linalg.lstsq(
        design * root_weights[:, None],
        response * root_weights,
        rcond=None,
    )
    phi = coefficients.astype(np.float64)
    phi += (target - float(phi.sum())) / n_players
    return phi


def deletion_curve(
    score_batch: ScoreBatch,
    base_tokens: np.ndarray,
    positions: np.ndarray,
    order: Sequence[int],
    replacement: int = MASK,
    batch_size: int = 256,
) -> list[float]:
    """Model output after deleting the first ``k`` squares in ``order``.

    ``positions`` is a flat array of token indices and ``order`` indexes into
    it. The returned list has ``len(order) + 1`` entries, starting at the
    unmodified output.
    """
    base = _as_batch_row(base_tokens)
    positions = np.asarray(positions, dtype=np.int64).ravel()
    order = [int(index) for index in order]
    rows = [base[0].copy()]
    current = base[0].copy()
    for index in order:
        current[int(positions[index])] = int(replacement)
        rows.append(current.copy())
    values = _batched(score_batch, np.stack(rows), batch_size)
    return [float(value) for value in values]


def attribute_position(
    handle: ModelHandle,
    boards: Sequence[chess.Board],
    device: str | torch.device | None = None,
    method: str = "occlusion",
    occlusion: str = "mask",
    samples: int = DEFAULT_SHAP_SAMPLES,
    seed: int = 0,
    background: Sequence[chess.Board] | None = None,
) -> AttributionResult:
    """Explain one model output for the current board.

    Kings are never treated as removable: a kingless board is not a legal
    position, so removing one measures the model's reaction to garbage.
    ``occlusion`` selects the replacement token for the occlusion method.

    KernelSHAP uses a coalition game over non-king pieces. Without a
    ``background`` the absent state is an empty square; with a background of
    real positions, absent squares take that background's content, which keeps
    the sampled boards in distribution (interventional SHAP).
    """
    if method not in METHODS:
        raise ValueError(f"method must be one of {METHODS}, got {method!r}")
    if occlusion not in OCCLUSION_MODES:
        raise ValueError(
            f"occlusion must be one of {OCCLUSION_MODES}, got {occlusion!r}"
        )
    if not boards:
        raise ValueError("at least one board is required")

    batch = model_batch(handle, boards)

    def score_batch(rows: np.ndarray) -> np.ndarray:
        return forward_tokens(handle, batch, rows, device)

    model_value = float(score_batch(batch.tokens[None])[0])
    n_segments = batch.square_positions.shape[0]
    kings = _king_mask(batch)
    background_rows = _background_rows(handle, background, n_segments)

    if method == "occlusion":
        replacement = occlusion_replacement(occlusion)
        _, deltas = occlusion_deltas(
            score_batch, batch.tokens, batch.square_positions, replacement
        )
        deltas[kings] = 0.0
        segment_squares = tuple(tuple(float(value) for value in row) for row in deltas)
        baseline_rows = batch.tokens[None].copy()
        baseline_rows[0, batch.square_positions[~kings]] = replacement
        baseline_value = float(score_batch(baseline_rows)[0])
    else:
        segment_squares, baseline_value = _shap_squares(
            score_batch,
            batch,
            kings,
            background_rows,
            samples=samples,
            seed=seed,
            model_value=model_value,
        )

    segment_totals = tuple(float(sum(segment)) for segment in segment_squares)
    squares = segment_squares[-1]
    piece_groups, piece_names = _piece_breakdown(batch.boards[-1], squares)
    return AttributionResult(
        task=handle.info.task,
        method=method,
        model_value=model_value,
        baseline_value=baseline_value,
        squares=squares,
        segment_squares=segment_squares,
        segment_totals=segment_totals,
        piece_groups=piece_groups,
        piece_names=piece_names,
        occlusion=occlusion if method == "occlusion" else None,
        samples=samples if method == "shap" else None,
        background=len(background_rows) if background_rows is not None else None,
        seed=seed if method == "shap" else None,
    )


def _shap_squares(
    score_batch: ScoreBatch,
    batch: ModelBatch,
    kings: np.ndarray,
    background_rows: np.ndarray | None,
    samples: int,
    seed: int,
    model_value: float,
) -> tuple[tuple[tuple[float, ...], ...], float]:
    """Shapley values over non-king pieces, plus the game's baseline value."""
    players_mask = _player_mask(batch) & ~kings
    n_segments, n_squares = batch.square_positions.shape
    flat = np.zeros(n_segments * n_squares, dtype=np.float64)
    players = np.flatnonzero(players_mask.ravel())
    if players.size == 0:
        zeros = tuple(tuple(0.0 for _ in range(n_squares)) for _ in range(n_segments))
        return zeros, model_value

    positions = batch.square_positions.ravel()[players]

    def value_fn(coalitions: np.ndarray) -> np.ndarray:
        coalitions = np.asarray(coalitions, dtype=bool)
        if background_rows is None:
            rows = np.repeat(batch.tokens[None], coalitions.shape[0], axis=0)
            for index, present in enumerate(coalitions):
                rows[index, positions[~present]] = EMPTY
            return score_batch(rows)
        rows = []
        for present in coalitions:
            for background in background_rows:
                row = batch.tokens.copy()
                for index, position in enumerate(positions):
                    if not present[index]:
                        row[position] = _background_token(background[position])
                rows.append(row)
        values = score_batch(np.stack(rows)).reshape(
            coalitions.shape[0], len(background_rows)
        )
        return values.mean(axis=1)

    phi = kernel_shap(value_fn, players.size, nsamples=samples, seed=seed)
    flat[players] = phi
    shaped = flat.reshape(n_segments, n_squares)
    baseline = float(value_fn(np.zeros((1, players.size), dtype=bool))[0])
    return (
        tuple(tuple(float(value) for value in row) for row in shaped),
        baseline,
    )


def _king_mask(batch: ModelBatch) -> np.ndarray:
    mask = np.zeros(batch.square_positions.shape, dtype=bool)
    for segment, board in enumerate(batch.boards):
        for square in board.pieces(chess.KING, chess.WHITE) | board.pieces(
            chess.KING, chess.BLACK
        ):
            mask[segment, square] = True
    return mask


def _player_mask(batch: ModelBatch) -> np.ndarray:
    mask = np.zeros(batch.square_positions.shape, dtype=bool)
    for segment, board in enumerate(batch.boards):
        for square in board.piece_map():
            mask[segment, square] = True
    return mask


def _background_rows(
    handle: ModelHandle,
    background: Sequence[chess.Board] | None,
    n_segments: int,
) -> np.ndarray | None:
    if not background:
        return None
    rows = [model_batch(handle, [board] * n_segments).tokens for board in background]
    return np.stack(rows)


def _background_token(token: int) -> int:
    """Copy a background square unless it would duplicate a king."""
    token = int(token)
    if EMPTY <= token < TURN_WHITE and token not in _KING_TOKENS:
        return token
    return EMPTY


def _piece_breakdown(
    board: chess.Board,
    squares: Sequence[float],
) -> tuple[dict[str, float], tuple[str | None, ...]]:
    groups: dict[str, float] = {}
    names: list[str | None] = [None] * 64
    for square, piece in board.piece_map().items():
        color = "white" if piece.color == chess.WHITE else "black"
        name = f"{color} {chess.piece_name(piece.piece_type)}"
        names[square] = name
        groups[name] = groups.get(name, 0.0) + float(squares[square])
    return groups, tuple(names)


def _shapley_kernel(n_players: int, sizes: np.ndarray) -> np.ndarray:
    combinations = np.array(
        [math.comb(n_players, int(size)) for size in range(n_players + 1)],
        dtype=np.float64,
    )
    sizes = sizes.astype(np.int64)
    return (n_players - 1) / (combinations[sizes] * sizes * (n_players - sizes))


def _batched(
    score_batch: ScoreBatch,
    rows: np.ndarray,
    batch_size: int,
) -> np.ndarray:
    chunks = [
        score_batch(rows[start : start + batch_size])
        for start in range(0, len(rows), batch_size)
    ]
    return np.concatenate(chunks) if chunks else np.empty(0)


def _as_batch_row(tokens: np.ndarray) -> np.ndarray:
    row = np.asarray(tokens)
    if row.ndim == 1:
        row = row[None, :]
    if row.ndim != 2 or row.shape[0] != 1:
        raise ValueError("base_tokens must be a single token row")
    return row
