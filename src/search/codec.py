"""Compact 34-byte encoding of one chess position.

The token stream stores 64 squares plus turn, castling and en-passant state as
an ``int16`` segment. That is 134 bytes per position, too wide for a billion-row
index. This codec packs the same information into 34 bytes:

* 32 bytes: one 4-bit nibble per square. ``0`` is empty, ``1..6`` are white
  pawn..king, ``7..12`` are black pawn..king.
* 1 byte: turn (bit 0), castling rights (bits 1-4), en-passant-present (bit 5).
* 1 byte: en-passant target square (0..63).

Only the three state tokens are stored in addition to the board, so the codec
is lossless with respect to the token stream. Move counters are intentionally
not represented: the token stream is a position, not a game state.
"""

from __future__ import annotations

from typing import Sequence

import chess
import numpy as np

from src.dataset.token_stream import (
    CASTLE_BASE,
    EMPTY,
    EP_BASE,
    N_SLOTS,
    PIECE_BASE,
    SEGMENT_LEN,
    TURN_BLACK,
    TURN_WHITE,
    TokenStreamEncoder,
    ep_square_token,
    parse_ep_token,
)

POSITION_BYTES = 34
_FLAGS_EP = 0x20
_ENCODER = TokenStreamEncoder()


def encode_position(tokens: Sequence[int] | np.ndarray) -> bytes:
    """Pack a 67-token position segment into 34 bytes."""
    values = np.asarray(tokens, dtype=np.int16).ravel()
    if values.size != SEGMENT_LEN:
        raise ValueError(
            f"A position segment must contain {SEGMENT_LEN} tokens, got {values.size}"
        )

    packed = bytearray(POSITION_BYTES)
    for square in range(N_SLOTS):
        token = int(values[square])
        if token == EMPTY:
            code = 0
        elif PIECE_BASE <= token <= PIECE_BASE + 11:
            code = token - PIECE_BASE + 1
        else:
            raise ValueError(f"Token {token} is not an empty or piece token")
        if square % 2 == 0:
            packed[square // 2] = code << 4
        else:
            packed[square // 2] |= code

    turn = int(values[N_SLOTS])
    if turn not in (TURN_WHITE, TURN_BLACK):
        raise ValueError(f"Expected a TURN token, got {turn}")
    castle = int(values[N_SLOTS + 1]) - CASTLE_BASE
    if not 0 <= castle <= 15:
        raise ValueError("Expected a CASTLE token")
    ep = int(values[N_SLOTS + 2])
    ep_square = parse_ep_token(ep)

    flags = (1 if turn == TURN_WHITE else 0) | (castle << 1)
    if ep_square is not None:
        flags |= _FLAGS_EP
        packed[33] = ep_square
    packed[32] = flags
    return bytes(packed)


def decode_position(packed: bytes | bytearray | memoryview) -> np.ndarray:
    """Unpack 34 bytes into a 67-token position segment."""
    payload = bytes(packed)
    if len(payload) != POSITION_BYTES:
        raise ValueError(
            f"A packed position must be {POSITION_BYTES} bytes, got {len(payload)}"
        )

    tokens = np.empty(SEGMENT_LEN, dtype=np.int16)
    for square in range(N_SLOTS):
        value = payload[square // 2]
        code = (value >> 4) if square % 2 == 0 else (value & 0x0F)
        if code > 12:
            raise ValueError(f"Square {square} has invalid piece code {code}")
        tokens[square] = EMPTY if code == 0 else PIECE_BASE + code - 1

    flags = payload[32]
    if flags & 0xC0:
        raise ValueError("Packed position has unknown flag bits")
    tokens[N_SLOTS] = TURN_WHITE if flags & 0x01 else TURN_BLACK

    castle = (flags >> 1) & 0x0F
    tokens[N_SLOTS + 1] = CASTLE_BASE + castle

    if flags & _FLAGS_EP:
        ep_square = payload[33]
        if ep_square >= 64:
            raise ValueError("Packed position has an invalid en-passant square")
        tokens[N_SLOTS + 2] = ep_square_token(ep_square)
    else:
        tokens[N_SLOTS + 2] = EP_BASE
    return tokens


def encode_board(board: chess.Board) -> bytes:
    """Pack a python-chess board into 34 bytes."""
    return encode_position(_ENCODER.encode(board))


def decode_board(packed: bytes | bytearray | memoryview) -> chess.Board:
    """Unpack 34 bytes into a python-chess board."""
    return _ENCODER.decode(decode_position(packed))
