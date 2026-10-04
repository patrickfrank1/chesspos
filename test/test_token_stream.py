import random

import chess
import numpy as np
import pytest

from src.dataset.position_encoder import get_encoder
from src.dataset.token_stream import (
    CASTLE_BASE,
    CLS,
    EP_BASE,
    MASK,
    PAD,
    SEP,
    TURN_BLACK,
    TURN_WHITE,
    VOCAB_SIZE,
    TokenStreamEncoder,
    castle_token,
    pack_stream,
    piece_square_id,
)
from src.dataset.types import TOKEN_STREAM


@pytest.fixture
def encoder() -> TokenStreamEncoder:
    return TokenStreamEncoder()


def random_game_boards(rng: random.Random, max_plies: int = 60) -> list[chess.Board]:
    board = chess.Board()
    boards = [board.copy()]
    for _ in range(max_plies):
        moves = list(board.legal_moves)
        if not moves:
            break
        board.push(rng.choice(moves))
        boards.append(board.copy())
    return boards


class TestVocabulary:
    def test_registered_in_registry(self):
        assert isinstance(get_encoder(TOKEN_STREAM), TokenStreamEncoder)

    def test_special_token_values(self):
        assert (PAD, MASK, SEP, CLS) == (0, 1, 2, 3)
        assert (TURN_WHITE, TURN_BLACK) == (4, 5)
        assert CASTLE_BASE == 6
        assert EP_BASE == 22
        assert piece_square_id(chess.Piece(chess.PAWN, chess.WHITE), 0) == 38
        assert (
            piece_square_id(chess.Piece(chess.KING, chess.BLACK), 63) == VOCAB_SIZE - 1
        )

    def test_dtype_and_value_range(self, encoder):
        rng = random.Random(42)
        for board in random_game_boards(rng):
            segment = encoder.encode(board)
            assert segment.dtype == np.int16
            assert np.all(segment >= 0)
            assert np.all(segment < VOCAB_SIZE)


class TestRoundTrip:
    def test_random_games(self, encoder):
        rng = random.Random(1234)
        plies = 0
        for _ in range(5):
            for board in random_game_boards(rng):
                decoded = encoder.decode(encoder.encode(board))
                assert decoded.epd() == board.epd()
                plies += 1
        assert plies >= 100

    def test_promotions(self, encoder):
        for promotion in [chess.QUEEN, chess.ROOK, chess.BISHOP, chess.KNIGHT]:
            board = chess.Board("4k3/P7/8/8/8/8/8/4K3 w - - 0 1")
            board.push(chess.Move(chess.A7, chess.A8, promotion=promotion))
            decoded = encoder.decode(encoder.encode(board))
            assert decoded.epd() == board.epd()

        for promotion in [chess.QUEEN, chess.ROOK, chess.BISHOP, chess.KNIGHT]:
            board = chess.Board("4k3/8/8/8/8/8/p7/4K3 b - - 0 1")
            board.push(chess.Move(chess.A2, chess.A1, promotion=promotion))
            decoded = encoder.decode(encoder.encode(board))
            assert decoded.epd() == board.epd()

    def test_en_passant_capture(self, encoder):
        board = chess.Board("4k3/8/8/8/3pP3/8/8/4K3 b - e3 0 1")
        assert board.has_legal_en_passant()
        decoded = encoder.decode(encoder.encode(board))
        assert decoded.epd() == board.epd()
        assert decoded.ep_square == board.ep_square

    def test_turn(self, encoder):
        white_to_move = encoder.encode(chess.Board())
        black_to_move = encoder.encode(
            chess.Board("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR b KQkq - 0 1")
        )
        assert white_to_move[0] == TURN_WHITE
        assert black_to_move[0] == TURN_BLACK


class TestEnPassantToken:
    def test_legal_ep_emits_token(self, encoder):
        board = chess.Board("4k3/8/8/8/3pP3/8/8/4K3 b - e3 0 1")
        segment = encoder.encode(board)
        assert segment[2] == EP_BASE + (chess.E3 - chess.A3)

    def test_pseudo_legal_ep_emits_no_token(self, encoder):
        board = chess.Board()
        board.push(chess.Move(chess.E2, chess.E4))
        assert not board.has_legal_en_passant()
        segment = encoder.encode(board)
        assert len(segment) == 2 + 32
        assert np.all(segment[:2] != EP_BASE)


class TestCanonicalOrder:
    def test_deterministic_and_sorted(self, encoder):
        rng = random.Random(7)
        for board in random_game_boards(rng):
            first = encoder.encode(board)
            second = encoder.encode(board)
            np.testing.assert_array_equal(first, second)
            piece_tokens = first[first >= 38]
            squares = (piece_tokens.astype(int) - 38) % 64
            assert np.all(np.diff(squares) > 0)


class TestCastling:
    @pytest.mark.parametrize("mask", range(16))
    def test_all_masks_round_trip(self, encoder, mask):
        board = chess.Board("r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1")
        fen = ""
        if mask & 1:
            fen += "K"
        if mask & 2:
            fen += "Q"
        if mask & 4:
            fen += "k"
        if mask & 8:
            fen += "q"
        board.set_castling_fen(fen)
        assert castle_token(board) == CASTLE_BASE + mask
        decoded = encoder.decode(encoder.encode(board))
        assert decoded.epd() == board.epd()


class TestDecodeValidation:
    def test_rejects_empty_segment(self, encoder):
        with pytest.raises(ValueError):
            encoder.decode(np.array([], dtype=np.int16))

    def test_rejects_missing_castle_token(self, encoder):
        with pytest.raises(ValueError, match="CASTLE"):
            encoder.decode(np.array([TURN_WHITE], dtype=np.int16))

    def test_rejects_unknown_token(self, encoder):
        segment = encoder.encode(chess.Board())
        segment_with_sep = np.concatenate([segment[:2], [SEP], segment[2:]])
        with pytest.raises(ValueError):
            encoder.decode(segment_with_sep)

    def test_rejects_duplicate_square(self, encoder):
        tokens = [
            TURN_WHITE,
            CASTLE_BASE,
            piece_square_id(chess.Piece(chess.PAWN, chess.WHITE), chess.E2),
            piece_square_id(chess.Piece(chess.KNIGHT, chess.WHITE), chess.E2),
        ]
        with pytest.raises(ValueError, match="Duplicate"):
            encoder.decode(np.array(tokens, dtype=np.int16))

    def test_rejects_padding(self, encoder):
        segment = encoder.encode(chess.Board())
        padded = np.concatenate([segment, [PAD]])
        with pytest.raises(ValueError):
            encoder.decode(padded)


class TestBatch:
    def test_encode_batch_pads_to_max(self, encoder):
        boards = [
            chess.Board(),
            chess.Board("4k3/8/8/8/8/8/8/4K3 w - - 0 1"),
        ]
        encoded, lengths = encoder.encode_batch(boards)
        assert encoded.dtype == np.int16
        assert encoded.shape == (2, 34)
        np.testing.assert_array_equal(lengths, [34, 4])
        assert np.all(encoded[1, 4:] == PAD)

    def test_encode_batch_empty(self, encoder):
        encoded, lengths = encoder.encode_batch([])
        assert encoded.shape == (0, 0)
        assert lengths.shape == (0,)

    def test_decode_batch_with_lengths(self, encoder):
        rng = random.Random(99)
        boards = random_game_boards(rng)[:10]
        encoded, lengths = encoder.encode_batch(boards)
        decoded = encoder.decode_batch(encoded, lengths)
        for original, rebuilt in zip(boards, decoded):
            assert rebuilt.epd() == original.epd()


class TestPackStream:
    def test_structure_with_cls(self, encoder):
        segments = [encoder.encode(chess.Board()), encoder.encode(chess.Board())]
        stream = pack_stream(segments, add_cls=True)
        assert stream[0] == CLS
        assert stream[35] == SEP
        assert stream[-1] == SEP
        assert len(stream) == 1 + 34 + 1 + 34 + 1

    def test_structure_without_cls(self, encoder):
        segments = [encoder.encode(chess.Board())]
        stream = pack_stream(segments, add_cls=False)
        assert stream[0] != CLS
        assert stream[-1] == SEP

    def test_rejects_empty_input(self):
        with pytest.raises(ValueError):
            pack_stream([])

    def test_decode_of_packed_stream_rejected(self, encoder):
        stream = pack_stream([encoder.encode(chess.Board())], add_cls=True)
        with pytest.raises(ValueError):
            encoder.decode(stream)
