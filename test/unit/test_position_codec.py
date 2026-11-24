import random

import chess
import numpy as np
import pytest

from bdd import given, then, when
from src.dataset.token_stream import TokenStreamEncoder
from src.search.codec import (
    POSITION_BYTES,
    decode_board,
    decode_position,
    encode_board,
    encode_position,
)


@pytest.fixture
def encoder() -> TokenStreamEncoder:
    return TokenStreamEncoder()


def test_starting_position_roundtrip(encoder):
    with given("the starting board"):
        board = chess.Board()
        tokens = encoder.encode(board)

    with when("the position is packed and unpacked"):
        packed = encode_position(tokens)
        decoded_tokens = decode_position(packed)

    with then("the payload is 34 bytes and the tokens match"):
        assert len(packed) == POSITION_BYTES == 34
        np.testing.assert_array_equal(decoded_tokens, tokens)

    with then("the decoded board is the original position"):
        assert decode_board(packed).epd() == board.epd()
    return


def test_random_game_positions_roundtrip(encoder):
    with given("a random legal game"):
        rng = random.Random(7)
        board = chess.Board()
        boards = [board.copy()]
        for _ in range(80):
            if board.is_game_over():
                break
            board.push(rng.choice(list(board.legal_moves)))
            boards.append(board.copy())

    with when("each position is packed and decoded"):
        decoded = [decode_board(encode_board(position)) for position in boards]

    with then("every decoded position equals its source position"):
        assert [position.epd() for position in decoded] == [
            position.epd() for position in boards
        ]
    return


def test_castling_turn_and_en_passant_are_preserved(encoder):
    with given("positions with castling rights and a legal en-passant capture"):
        castling = chess.Board("r3k2r/8/8/8/8/8/8/R3K2R b KQkq - 0 1")
        en_passant = chess.Board(
            "rnbqkbnr/ppp1p1pp/8/3pPp2/8/8/PPPP1PPP/RNBQKBNR w KQkq f6 0 3"
        )
        assert castling.has_kingside_castling_rights(chess.WHITE)
        assert en_passant.has_legal_en_passant()

    with when("each position is packed and decoded"):
        decoded = [
            decode_board(encode_position(encoder.encode(board)))
            for board in (castling, en_passant)
        ]

    with then("turn, castling and en-passant survive the roundtrip"):
        assert decoded[0].epd() == castling.epd()
        assert decoded[1].epd() == en_passant.epd()
    return


def test_invalid_payload_length_is_rejected():
    with given("a payload that is not 34 bytes"):
        payload = b"\x00" * (POSITION_BYTES - 1)

    with when("the payload is decoded"):
        with pytest.raises(ValueError):
            decode_position(payload)

    with then("no board is produced"):
        pass
    return
