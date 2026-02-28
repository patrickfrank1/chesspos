from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Callable

import chess
import numpy as np

from src.dataset.types import (
    ENCODING_SHAPES,
    EncodingFormat,
    TOKEN_SEQUENCE,
    TENSOR,
    BITBOARD,
)
from src.preprocessing.board_representation import (
    board_to_token_sequence,
    board_to_tensor,
    board_to_bitboard,
)


class PositionEncoder(ABC):
    @property
    @abstractmethod
    def encoding_format(self) -> EncodingFormat:
        pass

    @property
    @abstractmethod
    def output_shape(self) -> tuple[int, ...]:
        pass

    @abstractmethod
    def encode(self, board: chess.Board) -> np.ndarray:
        pass

    @abstractmethod
    def encode_batch(self, boards: list[chess.Board]) -> np.ndarray:
        pass


class TokenSequenceEncoder(PositionEncoder):
    @property
    def encoding_format(self) -> EncodingFormat:
        return TOKEN_SEQUENCE

    @property
    def output_shape(self) -> tuple[int, ...]:
        return ENCODING_SHAPES[TOKEN_SEQUENCE]

    def encode(self, board: chess.Board) -> np.ndarray:
        return board_to_token_sequence(board)

    def encode_batch(self, boards: list[chess.Board]) -> np.ndarray:
        return np.stack([self.encode(board) for board in boards])


class TensorEncoder(PositionEncoder):
    @property
    def encoding_format(self) -> EncodingFormat:
        return TENSOR

    @property
    def output_shape(self) -> tuple[int, ...]:
        return ENCODING_SHAPES[TENSOR]

    def encode(self, board: chess.Board) -> np.ndarray:
        return board_to_tensor(board)

    def encode_batch(self, boards: list[chess.Board]) -> np.ndarray:
        return np.stack([self.encode(board) for board in boards])


class BitboardEncoder(PositionEncoder):
    @property
    def encoding_format(self) -> EncodingFormat:
        return BITBOARD

    @property
    def output_shape(self) -> tuple[int, ...]:
        return ENCODING_SHAPES[BITBOARD]

    def encode(self, board: chess.Board) -> np.ndarray:
        return board_to_bitboard(board)

    def encode_batch(self, boards: list[chess.Board]) -> np.ndarray:
        return np.stack([self.encode(board) for board in boards])


_ENCODER_REGISTRY: dict[EncodingFormat, type[PositionEncoder]] = {
    TOKEN_SEQUENCE: TokenSequenceEncoder,
    TENSOR: TensorEncoder,
    BITBOARD: BitboardEncoder,
}


def get_encoder(encoding_format: EncodingFormat) -> PositionEncoder:
    if encoding_format not in _ENCODER_REGISTRY:
        raise ValueError(
            f"Unknown encoding format: {encoding_format}. "
            f"Available: {list(_ENCODER_REGISTRY.keys())}"
        )
    return _ENCODER_REGISTRY[encoding_format]()


def register_encoder(
    encoding_format: EncodingFormat, encoder_class: type[PositionEncoder]
) -> None:
    _ENCODER_REGISTRY[encoding_format] = encoder_class
