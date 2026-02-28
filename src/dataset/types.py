from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Iterator, Protocol, runtime_checkable

import chess
import numpy as np

if TYPE_CHECKING:
    pass


@runtime_checkable
class PositionEncoderProtocol(Protocol):
    def encode(self, board: chess.Board) -> np.ndarray: ...
    def encode_batch(self, boards: list[chess.Board]) -> np.ndarray: ...
    def decode(self, data: np.ndarray) -> chess.Board: ...
    def decode_batch(self, data: np.ndarray) -> list[chess.Board]: ...


@dataclass
class GameMetadata:
    white_elo: int | None = None
    black_elo: int | None = None
    result: str | None = None
    opening: str | None = None
    event: str | None = None
    date: str | None = None

    def to_dict(self) -> dict:
        return {
            "white_elo": self.white_elo,
            "black_elo": self.black_elo,
            "result": self.result,
            "opening": self.opening,
            "event": self.event,
            "date": self.date,
        }


@dataclass
class PositionRecord:
    board: chess.Board
    ply: int
    metadata: GameMetadata
    move_sequence: list[chess.Move] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "fen": self.board.fen(),
            "ply": self.ply,
            "metadata": self.metadata.to_dict(),
            "move_sequence": [move.uci() for move in self.move_sequence],
        }


@dataclass
class GameRecord:
    positions: list[PositionRecord]
    metadata: GameMetadata

    def to_dict(self) -> dict:
        return {
            "positions": [p.to_dict() for p in self.positions],
            "metadata": self.metadata.to_dict(),
        }

    def __iter__(self) -> Iterator[PositionRecord]:
        return iter(self.positions)

    def __len__(self) -> int:
        return len(self.positions)


@dataclass
class EncodedBatch:
    data: np.ndarray
    encoding_format: str
    metadata: list[dict] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.data)


EncodingFormat = str

TOKEN_SEQUENCE: EncodingFormat = "token_sequence"
TENSOR: EncodingFormat = "tensor"
BITBOARD: EncodingFormat = "bitboard"

ENCODING_SHAPES: dict[EncodingFormat, tuple[int, ...]] = {
    TOKEN_SEQUENCE: (69,),
    TENSOR: (8, 8, 15),
    BITBOARD: (773,),
}
