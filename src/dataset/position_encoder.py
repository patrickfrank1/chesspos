from __future__ import annotations

from abc import ABC, abstractmethod

import chess
import numpy as np

from src.dataset.types import (
    ENCODING_SHAPES,
    BITBOARD,
    EncodingFormat,
    TENSOR,
    TOKEN_SEQUENCE,
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

    @abstractmethod
    def decode(self, data: np.ndarray) -> chess.Board:
        pass

    @abstractmethod
    def decode_batch(self, data: np.ndarray) -> list[chess.Board]:
        pass


class TokenSequenceEncoder(PositionEncoder):
    _PIECE_ENCODING = {
        "empty": 15,
        "mask": 16,
        "P": 17,
        "N": 18,
        "B": 19,
        "R": 20,
        "Q": 21,
        "K": 22,
        "p": 23,
        "n": 24,
        "b": 25,
        "r": 26,
        "q": 27,
        "k": 28,
        "turn_white": 29,
        "turn_black": 30,
        "no_castling": 31,
        "castling": 32,
    }
    _INVERSE_PIECE_ENCODING = {v: k for k, v in _PIECE_ENCODING.items()}

    @property
    def encoding_format(self) -> EncodingFormat:
        return TOKEN_SEQUENCE

    @property
    def output_shape(self) -> tuple[int, ...]:
        return ENCODING_SHAPES[TOKEN_SEQUENCE]

    def encode(self, board: chess.Board) -> np.ndarray:
        pieces = [board.piece_at(square) for square in chess.SQUARES]
        encoding = np.zeros((64 + 5), dtype=np.int8)
        for i, piece in enumerate(pieces):
            encoding[i] = self._PIECE_ENCODING[
                piece.symbol() if piece is not None else "empty"
            ]
        encoding[64] = self._PIECE_ENCODING[
            "turn_white" if board.turn else "turn_black"
        ]
        encoding[65] = self._PIECE_ENCODING[
            "castling"
            if board.has_kingside_castling_rights(chess.WHITE)
            else "no_castling"
        ]
        encoding[66] = self._PIECE_ENCODING[
            "castling"
            if board.has_queenside_castling_rights(chess.WHITE)
            else "no_castling"
        ]
        encoding[67] = self._PIECE_ENCODING[
            "castling"
            if board.has_kingside_castling_rights(chess.BLACK)
            else "no_castling"
        ]
        encoding[68] = self._PIECE_ENCODING[
            "castling"
            if board.has_queenside_castling_rights(chess.BLACK)
            else "no_castling"
        ]
        return encoding

    def encode_batch(self, boards: list[chess.Board]) -> np.ndarray:
        return np.stack([self.encode(board) for board in boards])

    @staticmethod
    def _piece_id_to_piece(id: int) -> chess.Piece | None:
        if 17 <= id <= 28:
            return chess.Piece.from_symbol(
                TokenSequenceEncoder._INVERSE_PIECE_ENCODING[id]
            )
        return None

    def decode(self, data: np.ndarray) -> chess.Board:
        assert data.shape == (69,)
        board = chess.Board()
        board.clear()
        piece_map = {
            i: self._piece_id_to_piece(piece_id)
            for i, piece_id in enumerate(data[:64])
            if self._piece_id_to_piece(piece_id) is not None
        }
        board.set_piece_map(piece_map)
        board.turn = data[64] == 29
        castling_rights = ""
        if data[65] == 32:
            castling_rights += "K"
        if data[66] == 32:
            castling_rights += "Q"
        if data[67] == 32:
            castling_rights += "k"
        if data[68] == 32:
            castling_rights += "q"
        board.set_castling_fen(castling_rights)
        return board

    def decode_batch(self, data: np.ndarray) -> list[chess.Board]:
        return [self.decode(d) for d in data]


class TensorEncoder(PositionEncoder):
    @property
    def encoding_format(self) -> EncodingFormat:
        return TENSOR

    @property
    def output_shape(self) -> tuple[int, ...]:
        return ENCODING_SHAPES[TENSOR]

    def encode(self, board: chess.Board) -> np.ndarray:
        embedding = np.zeros((8, 8, 15), dtype=bool)
        for color in [1, 0]:
            for i in range(1, 7):
                index = (1 - color) * 6 + i - 1
                bmp = np.zeros(shape=(64,)).astype(bool)
                for j in list(board.pieces(i, color)):
                    bmp[j] = True
                bmp = bmp.reshape((8, 8))
                embedding[:, :, index] = bmp
        embedding[0, 0, 12] = board.has_queenside_castling_rights(chess.WHITE)
        embedding[0, 7, 12] = board.has_kingside_castling_rights(chess.WHITE)
        embedding[7, 0, 12] = board.has_queenside_castling_rights(chess.BLACK)
        embedding[7, 7, 12] = board.has_kingside_castling_rights(chess.BLACK)
        en_passant = np.zeros((64,), dtype=bool)
        if board.has_legal_en_passant():
            en_passant[board.ep_square] = True
        en_passant = en_passant.reshape((8, 8))
        embedding[:, :, 13] = en_passant
        embedding[0, 0, 14] = board.turn
        return embedding

    def encode_batch(self, boards: list[chess.Board]) -> np.ndarray:
        return np.stack([self.encode(board) for board in boards])

    def decode(self, data: np.ndarray, threshold: float = 0.5) -> chess.Board:
        assert data.shape == (8, 8, 15), (
            f"decode encountered an input with invalid shape {data.shape}, "
            + "expected shape (8, 8, 15)"
        )
        data = np.where(data > threshold, 1, 0)
        board = chess.Board()
        board.clear()
        for color in [1, 0]:
            for i in range(1, 7):
                idx = (1 - color) * 6 + i - 1
                piece = chess.Piece(i, color)
                square_bitmask = data[:, :, idx].reshape((64,))
                squares = np.flatnonzero(square_bitmask)
                for square in squares:
                    board.set_piece_at(square, piece)
        castling_rights = ""
        if data[0, 0, 12]:
            castling_rights += "Q"
        if data[0, 7, 12]:
            castling_rights += "K"
        if data[7, 0, 12]:
            castling_rights += "q"
        if data[7, 7, 12]:
            castling_rights += "k"
        board.set_castling_fen(castling_rights)
        en_passant = data[:, :, 13].reshape((64,))
        if np.any(en_passant):
            board.ep_square = np.flatnonzero(en_passant)[0]
        board.turn = data[0, 0, 14]
        return board

    def decode_batch(self, data: np.ndarray) -> list[chess.Board]:
        return [self.decode(d) for d in data]


class BitboardEncoder(PositionEncoder):
    @property
    def encoding_format(self) -> EncodingFormat:
        return BITBOARD

    @property
    def output_shape(self) -> tuple[int, ...]:
        return ENCODING_SHAPES[BITBOARD]

    def encode(self, board: chess.Board) -> np.ndarray:
        embedding = np.array([], dtype=bool)
        for color in [1, 0]:
            for i in range(1, 7):
                bmp = np.zeros(shape=(64,)).astype(bool)
                for j in list(board.pieces(i, color)):
                    bmp[j] = True
                embedding = np.concatenate((embedding, bmp))
        additional = np.array(
            [
                bool(board.turn),
                board.has_kingside_castling_rights(chess.WHITE),
                board.has_queenside_castling_rights(chess.WHITE),
                board.has_kingside_castling_rights(chess.BLACK),
                board.has_queenside_castling_rights(chess.BLACK),
            ]
        )
        embedding = np.concatenate((embedding, additional))
        return embedding

    def encode_batch(self, boards: list[chess.Board]) -> np.ndarray:
        return np.stack([self.encode(board) for board in boards])

    def decode(self, data: np.ndarray) -> chess.Board:
        assert data.shape == (773,)
        board = chess.Board()
        board.clear()
        for color in [1, 0]:
            for i in range(1, 7):
                idx = (1 - color) * 6 + i - 1
                piece = chess.Piece(i, color)
                bitmask = data[idx * 64 : (idx + 1) * 64]
                squares = np.flatnonzero(bitmask)
                for square in squares:
                    board.set_piece_at(square, piece)
        board.turn = data[768]
        castling_rights = ""
        if data[769]:
            castling_rights += "Q"
        if data[770]:
            castling_rights += "K"
        if data[771]:
            castling_rights += "q"
        if data[772]:
            castling_rights += "k"
        board.set_castling_fen(castling_rights)
        return board

    def decode_batch(self, data: np.ndarray) -> list[chess.Board]:
        return [self.decode(d) for d in data]


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
