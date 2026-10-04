from __future__ import annotations

from abc import ABC, abstractmethod

import chess
import numpy as np

from src.dataset.types import TOKEN_STREAM, EncodingFormat


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

    @abstractmethod
    def decode(self, data: np.ndarray) -> chess.Board:
        pass

    @abstractmethod
    def decode_batch(self, data: np.ndarray) -> list[chess.Board]:
        pass


_ENCODER_REGISTRY: dict[EncodingFormat, type[PositionEncoder]] = {}


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


from src.dataset.token_stream import TokenStreamEncoder  # noqa: E402

register_encoder(TOKEN_STREAM, TokenStreamEncoder)
