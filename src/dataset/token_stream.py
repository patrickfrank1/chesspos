from __future__ import annotations

from dataclasses import dataclass

import chess
import numpy as np

PAD = 0
MASK = 1
SEP = 2
CLS = 3

EMPTY = 4
PIECE_BASE = 5

TURN_WHITE = 17
TURN_BLACK = 18

CASTLE_BASE = 19
EP_BASE = 35
EP_NONE = EP_BASE

N_SLOTS = 64
SEGMENT_LEN = N_SLOTS + 3
VOCAB_SIZE = 52

_PIECE_INDEX = {
    chess.PAWN: 0,
    chess.KNIGHT: 1,
    chess.BISHOP: 2,
    chess.ROOK: 3,
    chess.QUEEN: 4,
    chess.KING: 5,
}
_PIECE_TYPES = {index: piece_type for piece_type, index in _PIECE_INDEX.items()}


def piece_id(piece: chess.Piece) -> int:
    piece_index = _PIECE_INDEX[piece.piece_type]
    if piece.color == chess.BLACK:
        piece_index += 6
    return PIECE_BASE + piece_index


def parse_piece(token: int) -> chess.Piece:
    piece_index = token - PIECE_BASE
    if not 0 <= piece_index <= 11:
        raise ValueError(f"Token {token} is not a piece token")
    color = chess.BLACK if piece_index >= 6 else chess.WHITE
    piece_type = _PIECE_TYPES[piece_index % 6]
    return chess.Piece(piece_type, color)


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


def ep_token(board: chess.Board) -> int:
    if not board.has_legal_en_passant():
        return EP_NONE
    return ep_square_token(board.ep_square)


def ep_square_token(square: int) -> int:
    rank = chess.square_rank(square)
    if rank == 2:
        return EP_BASE + 1 + (square - chess.A3)
    if rank == 5:
        return EP_BASE + 1 + 8 + (square - chess.A6)
    raise ValueError(
        f"Square {chess.square_name(square)} is not a reachable en-passant target"
    )


def parse_ep_token(token: int) -> int | None:
    index = token - EP_BASE
    if not 0 <= index <= 16:
        raise ValueError(f"Token {token} is not an en-passant token")
    if index == 0:
        return None
    if index <= 8:
        return chess.A3 + (index - 1)
    return chess.A6 + (index - 9)


@dataclass
class TokenStreamEncoder:
    def encode(self, board: chess.Board) -> np.ndarray:
        tokens = np.empty(SEGMENT_LEN, dtype=np.int16)
        piece_map = board.piece_map()
        for square in range(N_SLOTS):
            piece = piece_map.get(square)
            tokens[square] = EMPTY if piece is None else piece_id(piece)
        tokens[N_SLOTS] = TURN_WHITE if board.turn else TURN_BLACK
        tokens[N_SLOTS + 1] = castle_token(board)
        tokens[N_SLOTS + 2] = ep_token(board)
        return tokens

    def encode_batch(self, boards: list[chess.Board]) -> tuple[np.ndarray, np.ndarray]:
        segments = [self.encode(board) for board in boards]
        if not segments:
            return np.empty((0, 0), dtype=np.int16), np.empty(0, dtype=np.int32)
        lengths = np.full(len(segments), SEGMENT_LEN, dtype=np.int32)
        return np.stack(segments), lengths

    def decode(self, data: np.ndarray) -> chess.Board:
        tokens = [int(token) for token in np.asarray(data).ravel()]
        if len(tokens) != SEGMENT_LEN:
            raise ValueError(
                f"A slot segment must have exactly {SEGMENT_LEN} tokens, "
                f"got {len(tokens)}"
            )

        pieces: dict[int, chess.Piece] = {}
        for square in range(N_SLOTS):
            token = tokens[square]
            if token == EMPTY:
                continue
            pieces[square] = parse_piece(token)

        turn = tokens[N_SLOTS]
        if turn not in (TURN_WHITE, TURN_BLACK):
            raise ValueError(f"Expected a TURN token, got {turn}")

        castle = tokens[N_SLOTS + 1]
        if not CASTLE_BASE <= castle < CASTLE_BASE + 16:
            raise ValueError(f"Expected a CASTLE token, got {castle}")

        ep_square = parse_ep_token(tokens[N_SLOTS + 2])

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
