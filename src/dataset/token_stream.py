from __future__ import annotations

from dataclasses import dataclass

import chess
import numpy as np

from src.dataset.position_encoder import PositionEncoder
from src.dataset.types import TOKEN_STREAM

PAD = 0
MASK = 1
SEP = 2
CLS = 3

TURN_WHITE = 4
TURN_BLACK = 5

CASTLE_BASE = 6
EP_BASE = 22
PIECE_SQUARE_BASE = 38
VOCAB_SIZE = 806

_PIECE_INDEX = {
    chess.PAWN: 0,
    chess.KNIGHT: 1,
    chess.BISHOP: 2,
    chess.ROOK: 3,
    chess.QUEEN: 4,
    chess.KING: 5,
}
_PIECE_TYPES = {index: piece_type for piece_type, index in _PIECE_INDEX.items()}


def piece_square_id(piece: chess.Piece, square: int) -> int:
    piece_index = _PIECE_INDEX[piece.piece_type]
    if piece.color == chess.BLACK:
        piece_index += 6
    return PIECE_SQUARE_BASE + 64 * piece_index + square


def parse_piece_square(token: int) -> tuple[chess.Piece, int]:
    offset = token - PIECE_SQUARE_BASE
    if offset < 0 or offset >= VOCAB_SIZE - PIECE_SQUARE_BASE:
        raise ValueError(f"Token {token} is not a piece-square token")
    piece_index, square = divmod(offset, 64)
    color = chess.BLACK if piece_index >= 6 else chess.WHITE
    piece_type = _PIECE_TYPES[piece_index % 6]
    return chess.Piece(piece_type, color), square


def castle_token(board: chess.Board) -> int:
    mask = 0
    if board.has_kingside_castling_rights(chess.WHITE):
        mask |= 1
    if board.has_queenside_castling_rights(chess.WHITE):
        mask |= 2
    if board.has_kingside_castling_rights(chess.BLACK):
        mask |= 4
    if board.has_queenside_castling_rights(chess.BLACK):
        mask |= 8
    return CASTLE_BASE + mask


def parse_castle_token(token: int) -> str:
    mask = token - CASTLE_BASE
    if not 0 <= mask <= 15:
        raise ValueError(f"Token {token} is not a castling token")
    symbols = ""
    if mask & 1:
        symbols += "K"
    if mask & 2:
        symbols += "Q"
    if mask & 4:
        symbols += "k"
    if mask & 8:
        symbols += "q"
    return symbols


def ep_token(board: chess.Board) -> int | None:
    if not board.has_legal_en_passant():
        return None
    return ep_square_token(board.ep_square)


def ep_square_token(square: int) -> int:
    rank = chess.square_rank(square)
    if rank == 2:
        return EP_BASE + (square - chess.A3)
    if rank == 5:
        return EP_BASE + 8 + (square - chess.A6)
    raise ValueError(
        f"Square {chess.square_name(square)} is not a reachable en-passant target"
    )


def parse_ep_token(token: int) -> int:
    index = token - EP_BASE
    if index < 0 or index > 15:
        raise ValueError(f"Token {token} is not an en-passant token")
    return chess.A3 + index if index < 8 else chess.A6 + (index - 8)


@dataclass
class TokenStreamEncoder(PositionEncoder):
    encoding_format = TOKEN_STREAM
    output_shape = None

    def encode(self, board: chess.Board) -> np.ndarray:
        tokens = [TURN_WHITE if board.turn else TURN_BLACK, castle_token(board)]
        ep = ep_token(board)
        if ep is not None:
            tokens.append(ep)
        piece_map = board.piece_map()
        tokens.extend(
            piece_square_id(piece, square)
            for square, piece in sorted(piece_map.items())
        )
        return np.array(tokens, dtype=np.int16)

    def encode_batch(self, boards: list[chess.Board]) -> tuple[np.ndarray, np.ndarray]:
        segments = [self.encode(board) for board in boards]
        if not segments:
            return np.empty((0, 0), dtype=np.int16), np.empty(0, dtype=np.int32)
        lengths = np.array([len(segment) for segment in segments], dtype=np.int32)
        padded = np.full((len(segments), int(lengths.max())), PAD, dtype=np.int16)
        for i, segment in enumerate(segments):
            padded[i, : len(segment)] = segment
        return padded, lengths

    def decode(self, data: np.ndarray) -> chess.Board:
        tokens = [int(token) for token in np.asarray(data).ravel()]
        if len(tokens) < 2:
            raise ValueError(
                "A token stream segment needs at least a TURN and a CASTLE token"
            )

        turn = tokens[0]
        if turn not in (TURN_WHITE, TURN_BLACK):
            raise ValueError(f"Expected a TURN token, got {turn}")

        castle = tokens[1]
        if not CASTLE_BASE <= castle < CASTLE_BASE + 16:
            raise ValueError(f"Expected a CASTLE token, got {castle}")

        index = 2
        ep_square = None
        if len(tokens) > index and EP_BASE <= tokens[index] < EP_BASE + 16:
            ep_square = parse_ep_token(tokens[index])
            index += 1

        pieces: dict[int, chess.Piece] = {}
        for token in tokens[index:]:
            piece, square = parse_piece_square(token)
            if square in pieces:
                raise ValueError(
                    f"Duplicate piece-square token for {chess.square_name(square)}"
                )
            pieces[square] = piece

        board = chess.Board()
        board.clear()
        board.set_piece_map(pieces)
        board.turn = turn == TURN_WHITE
        board.set_castling_fen(parse_castle_token(castle))
        if ep_square is not None:
            board.ep_square = ep_square
        return board

    def decode_batch(
        self, data: np.ndarray, lengths: np.ndarray | None = None
    ) -> list[chess.Board]:
        boards = []
        for i, row in enumerate(np.asarray(data)):
            if lengths is not None:
                row = row[: int(lengths[i])]
            boards.append(self.decode(row))
        return boards


def pack_stream(segments: list[np.ndarray], add_cls: bool = True) -> np.ndarray:
    if not segments:
        raise ValueError("Cannot pack an empty list of segments")
    tokens = [CLS] if add_cls else []
    for segment in segments:
        if len(segment) == 0:
            raise ValueError("Cannot pack an empty segment")
        tokens.extend(int(token) for token in segment)
        tokens.append(SEP)
    return np.array(tokens, dtype=np.int16)
