import random

import chess
import numpy as np
import pytest

from bdd import given, then, when
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
    def test_special_token_values(self):
        with given("the token vocabulary and two pieces on fixed squares"):
            pawn = chess.Piece(chess.PAWN, chess.WHITE)
            king = chess.Piece(chess.KING, chess.BLACK)

        with when("piece-square ids are derived for those pieces"):
            pawn_id = piece_square_id(pawn, 0)
            king_id = piece_square_id(king, 63)

        with then("special tokens and ranges have their fixed ids"):
            assert (PAD, MASK, SEP, CLS) == (0, 1, 2, 3)
            assert (TURN_WHITE, TURN_BLACK) == (4, 5)
            assert CASTLE_BASE == 6
            assert EP_BASE == 22
            assert pawn_id == 38
            assert king_id == VOCAB_SIZE - 1
        return

    def test_dtype_and_value_range(self, encoder):
        with given("an encoder and boards from random games"):
            rng = random.Random(42)
            boards = random_game_boards(rng)

        with when("each board is encoded"):
            segments = [encoder.encode(board) for board in boards]

        with then("segments are int16 with values inside the vocabulary range"):
            for segment in segments:
                assert segment.dtype == np.int16
                assert np.all(segment >= 0)
                assert np.all(segment < VOCAB_SIZE)
        return


class TestRoundTrip:
    def test_random_games(self, encoder):
        with given("an encoder and boards from random games"):
            rng = random.Random(1234)
            boards = [board for _ in range(5) for board in random_game_boards(rng)]

        with when("each board is encoded and decoded"):
            decoded = [encoder.decode(encoder.encode(board)) for board in boards]

        with then("every decoded board matches the original"):
            for original, rebuilt in zip(boards, decoded):
                assert rebuilt.epd() == original.epd()
            assert len(boards) >= 100
        return

    def test_promotions(self, encoder):
        with given("boards one move before a promotion on each wing"):
            promotions = [chess.QUEEN, chess.ROOK, chess.BISHOP, chess.KNIGHT]
            white_boards = []
            black_boards = []
            for promotion in promotions:
                white_board = chess.Board("4k3/P7/8/8/8/8/8/4K3 w - - 0 1")
                white_board.push(chess.Move(chess.A7, chess.A8, promotion=promotion))
                white_boards.append(white_board)
                black_board = chess.Board("4k3/8/8/8/8/8/p7/4K3 b - - 0 1")
                black_board.push(chess.Move(chess.A2, chess.A1, promotion=promotion))
                black_boards.append(black_board)

        with when("each promoted position is encoded and decoded"):
            decoded = [
                encoder.decode(encoder.encode(board))
                for board in white_boards + black_boards
            ]

        with then("every decoded board matches the original"):
            for original, rebuilt in zip(white_boards + black_boards, decoded):
                assert rebuilt.epd() == original.epd()
        return

    def test_en_passant_capture(self, encoder):
        with given("a board with a legal en passant capture"):
            board = chess.Board("4k3/8/8/8/3pP3/8/8/4K3 b - e3 0 1")

        with when("the board is encoded and decoded"):
            decoded = encoder.decode(encoder.encode(board))

        with then("the decoded board matches the original including the ep square"):
            assert board.has_legal_en_passant()
            assert decoded.epd() == board.epd()
            assert decoded.ep_square == board.ep_square
        return

    def test_turn(self, encoder):
        with given("boards with white and black to move"):
            white_board = chess.Board()
            black_board = chess.Board(
                "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR b KQkq - 0 1"
            )

        with when("each board is encoded"):
            white_to_move = encoder.encode(white_board)
            black_to_move = encoder.encode(black_board)

        with then("the first token encodes the side to move"):
            assert white_to_move[0] == TURN_WHITE
            assert black_to_move[0] == TURN_BLACK
        return


class TestEnPassantToken:
    def test_legal_ep_emits_token(self, encoder):
        with given("a board with a legal en passant capture"):
            board = chess.Board("4k3/8/8/8/3pP3/8/8/4K3 b - e3 0 1")

        with when("the board is encoded"):
            segment = encoder.encode(board)

        with then("the en passant target token is emitted"):
            assert segment[2] == EP_BASE + (chess.E3 - chess.A3)
        return

    def test_pseudo_legal_ep_emits_no_token(self, encoder):
        with given("a board whose en passant square is only pseudo-legal"):
            board = chess.Board()
            board.push(chess.Move(chess.E2, chess.E4))

        with when("the board is encoded"):
            segment = encoder.encode(board)

        with then("no en passant token is emitted"):
            assert not board.has_legal_en_passant()
            assert len(segment) == 2 + 32
            assert np.all(segment[:2] != EP_BASE)
        return


class TestCanonicalOrder:
    def test_deterministic_and_sorted(self, encoder):
        with given("an encoder and boards from a random game"):
            rng = random.Random(7)
            boards = random_game_boards(rng)

        with when("each board is encoded twice"):
            encodings = [
                (encoder.encode(board), encoder.encode(board)) for board in boards
            ]

        with then("encodings are identical and piece tokens are sorted by square"):
            for first, second in encodings:
                np.testing.assert_array_equal(first, second)
                piece_tokens = first[first >= 38]
                squares = (piece_tokens.astype(int) - 38) % 64
                assert np.all(np.diff(squares) > 0)
        return


class TestCastling:
    @pytest.mark.parametrize("mask", range(16))
    def test_all_masks_round_trip(self, encoder, mask):
        with given("a board with the castling rights selected by the mask"):
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

        with when("the castle token is derived and the board is round-tripped"):
            token = castle_token(board)
            decoded = encoder.decode(encoder.encode(board))

        with then("the token encodes the rights and the decoded board matches"):
            assert token == CASTLE_BASE + mask
            assert decoded.epd() == board.epd()
        return


class TestDecodeValidation:
    def test_rejects_empty_segment(self, encoder):
        with given("an empty token segment"):
            segment = np.array([], dtype=np.int16)

        with when("the segment is decoded"):
            decoded = None
            with pytest.raises(ValueError) as excinfo:
                decoded = encoder.decode(segment)

        with then("a ValueError is raised and nothing is decoded"):
            assert decoded is None
            assert isinstance(excinfo.value, ValueError)
        return

    def test_rejects_missing_castle_token(self, encoder):
        with given("a segment without a castle token"):
            segment = np.array([TURN_WHITE], dtype=np.int16)

        with when("the segment is decoded"):
            error = None
            with pytest.raises(ValueError) as excinfo:
                error = encoder.decode(segment)

        with then("a ValueError mentioning CASTLE is raised"):
            assert error is None
            assert "CASTLE" in str(excinfo.value)
        return

    def test_rejects_unknown_token(self, encoder):
        with given("a valid segment with a stray SEP inserted"):
            segment = encoder.encode(chess.Board())
            segment_with_sep = np.concatenate([segment[:2], [SEP], segment[2:]])

        with when("the segment is decoded"):
            decoded = None
            with pytest.raises(ValueError):
                decoded = encoder.decode(segment_with_sep)

        with then("a ValueError is raised and nothing is decoded"):
            assert decoded is None
        return

    def test_rejects_duplicate_square(self, encoder):
        with given("a hand-built segment with two pieces on the same square"):
            tokens = [
                TURN_WHITE,
                CASTLE_BASE,
                piece_square_id(chess.Piece(chess.PAWN, chess.WHITE), chess.E2),
                piece_square_id(chess.Piece(chess.KNIGHT, chess.WHITE), chess.E2),
            ]

        with when("the segment is decoded"):
            error = None
            with pytest.raises(ValueError) as excinfo:
                error = encoder.decode(np.array(tokens, dtype=np.int16))

        with then("a ValueError mentioning Duplicate is raised"):
            assert error is None
            assert "Duplicate" in str(excinfo.value)
        return

    def test_rejects_padding(self, encoder):
        with given("a valid segment padded with a PAD token"):
            segment = encoder.encode(chess.Board())
            padded = np.concatenate([segment, [PAD]])

        with when("the segment is decoded"):
            decoded = None
            with pytest.raises(ValueError):
                decoded = encoder.decode(padded)

        with then("a ValueError is raised and nothing is decoded"):
            assert decoded is None
        return


class TestBatch:
    def test_encode_batch_pads_to_max(self, encoder):
        with given("boards of different segment lengths"):
            boards = [
                chess.Board(),
                chess.Board("4k3/8/8/8/8/8/8/4K3 w - - 0 1"),
            ]

        with when("the boards are batch-encoded"):
            encoded, lengths = encoder.encode_batch(boards)

        with then("the batch is padded to the max length with lengths reported"):
            assert encoded.dtype == np.int16
            assert encoded.shape == (2, 34)
            np.testing.assert_array_equal(lengths, [34, 4])
            assert np.all(encoded[1, 4:] == PAD)
        return

    def test_encode_batch_empty(self, encoder):
        with when("an empty list of boards is batch-encoded"):
            encoded, lengths = encoder.encode_batch([])

        with then("empty arrays are returned"):
            assert encoded.shape == (0, 0)
            assert lengths.shape == (0,)
        return

    def test_decode_batch_with_lengths(self, encoder):
        with given("boards from a random game"):
            rng = random.Random(99)
            boards = random_game_boards(rng)[:10]

        with when("the boards are batch-encoded and decoded with lengths"):
            encoded, lengths = encoder.encode_batch(boards)
            decoded = encoder.decode_batch(encoded, lengths)

        with then("every decoded board matches the original"):
            for original, rebuilt in zip(boards, decoded):
                assert rebuilt.epd() == original.epd()
        return


class TestPackStream:
    def test_structure_with_cls(self, encoder):
        with given("two encoded board segments"):
            segments = [encoder.encode(chess.Board()), encoder.encode(chess.Board())]

        with when("the segments are packed with a leading CLS"):
            stream = pack_stream(segments, add_cls=True)

        with then("CLS and SEP tokens frame the stream at the expected offsets"):
            assert stream[0] == CLS
            assert stream[35] == SEP
            assert stream[-1] == SEP
            assert len(stream) == 1 + 34 + 1 + 34 + 1
        return

    def test_structure_without_cls(self, encoder):
        with given("one encoded board segment"):
            segments = [encoder.encode(chess.Board())]

        with when("the segment is packed without a leading CLS"):
            stream = pack_stream(segments, add_cls=False)

        with then("the stream starts with a content token and ends with SEP"):
            assert stream[0] != CLS
            assert stream[-1] == SEP
        return

    def test_rejects_empty_input(self):
        with when("an empty list of segments is packed"):
            stream = None
            with pytest.raises(ValueError):
                stream = pack_stream([])

        with then("a ValueError is raised and nothing is packed"):
            assert stream is None
        return

    def test_decode_of_packed_stream_rejected(self, encoder):
        with given("a packed stream of multiple segments"):
            stream = pack_stream([encoder.encode(chess.Board())], add_cls=True)

        with when("the stream is decoded as a single segment"):
            decoded = None
            with pytest.raises(ValueError):
                decoded = encoder.decode(stream)

        with then("a ValueError is raised and nothing is decoded"):
            assert decoded is None
        return
