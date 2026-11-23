"""Shared fixtures for the position vector-index tests."""

from __future__ import annotations

import datetime as dt
from typing import Sequence

import chess
import numpy as np

from src.dataset.token_stream import PIECE_BASE, TURN_WHITE, TokenStreamEncoder
from src.search.codec import encode_position
from src.search.schema import PositionRow


class FakeEmbedder:
    """Deterministic, normalized embedder that needs no torch checkpoint."""

    def __init__(self, dim: int = 8, vocab_size: int | None = None) -> None:
        self.dim = dim
        self.vocab_size = vocab_size
        return

    def embed(self, segments: Sequence[np.ndarray]) -> np.ndarray:
        vectors = np.empty((len(segments), self.dim), dtype=np.float32)
        for index, segment in enumerate(segments):
            values = np.asarray(segment, dtype=np.int64)
            seed = int(np.sum(values * (np.arange(values.size) + 1))) % (2**32)
            vectors[index] = np.random.default_rng(seed).normal(size=self.dim)
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        return vectors / np.maximum(norms, 1e-8)


def game_boards() -> list[chess.Board]:
    """Return the starting position, after 1.e4, and after 1.e4 e5."""
    board = chess.Board()
    boards = [board.copy()]
    for uci in ("e2e4", "e7e5"):
        board.push(chess.Move.from_uci(uci))
        boards.append(board.copy())
    return boards


def game_segments() -> list[np.ndarray]:
    encoder = TokenStreamEncoder()
    return [encoder.encode(board) for board in game_boards()]


def piece_count(tokens: np.ndarray) -> int:
    values = np.asarray(tokens)
    return int(np.sum((values >= PIECE_BASE) & (values < TURN_WHITE)))


def make_position_row(
    *,
    vector: np.ndarray | None = None,
    game_id: int = 0x0123456789ABCDEF,
    ply: int = 1,
    white_elo: int = 2500,
    black_elo: int = 2500,
    utc_date: dt.date = dt.date(2024, 1, 1),
    year: int = 2024,
    result: str = "1-0",
    split: str = "train",
    tokens: np.ndarray | None = None,
    dim: int = 8,
) -> PositionRow:
    if tokens is None:
        tokens = TokenStreamEncoder().encode(chess.Board())
    values = np.asarray(tokens, dtype=np.int16)
    if vector is None:
        vector = np.zeros(dim, dtype=np.float32)
        vector[ply % dim] = 1.0
    return PositionRow(
        vector=np.asarray(vector, dtype=np.float32),
        packed_position=encode_position(values),
        game_id=game_id,
        ply=ply,
        white_elo=white_elo,
        black_elo=black_elo,
        utc_date=utc_date,
        year=year,
        result=result,
        split=split,
        n_pieces=piece_count(values),
    )
