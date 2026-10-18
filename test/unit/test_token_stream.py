import random

import chess
import numpy as np
import pytest

from bdd import given, then, when
from src.dataset.token_stream import (
    CASTLE_BASE,
    CLS,
    EMPTY,
    EP_BASE,
    EP_NONE,
    MASK,
    N_SLOTS,
    PAD,
    PIECE_BASE,
    SEGMENT_LEN,
    SEP,
    TURN_BLACK,
    TURN_WHITE,
    VOCAB_SIZE,
    TokenStreamEncoder,
    castle_token,
    ep_square_token,
    ep_token,
    pack_stream,
    parse_ep_token,
    piece_id,
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
        with given("the slot vocabulary and two pieces"):
            pawn = chess.Piece(chess.PAWN, chess.WHITE)
            king = chess.Piece(chess.KING, chess.BLACK)

        with when("content ids are derived for those pieces"):
            pawn_id_value = piece_id(pawn)
            king_id_value = piece_id(king)

        with then("special tokens and ranges have their fixed ids"):
            assert (PAD, MASK, SEP, CLS) == (0, 1, 2, 3)
            assert EMPTY == 4
            assert pawn_id_value == PIECE_BASE
            assert king_id_value == PIECE_BASE + 11
            assert (TURN_WHITE, TURN_BLACK) == (17, 18)
            assert CASTLE_BASE == 19
            assert EP_NONE == 35
            assert VOCAB_SIZE == 52
            assert parse_ep_token(EP_NONE) is None
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

    def test_fixed_segment_length(self, encoder):
        with given("boards from random games with wildly different piece counts"):
            rng = random.Random(3)
            boards = random_game_boards(rng, max_plies=80)

        with when("each board is encoded"):
            segments = [encoder.encode(board) for board in boards]

        with then("every segment has exactly SEGMENT_LEN tokens"):
            for segment in segments:
                assert len(segment) == SEGMENT_LEN
                assert SEGMENT_LEN == N_SLOTS + 3
        return

    def test_slots_only_hold_content_tokens(self, encoder):
        with given("boards from random games"):
            rng = random.Random(11)
            boards = random_game_boards(rng)

        with when("the slot region of each encoding is inspected"):
            slot_values = [
                set(np.unique(encoder.encode(board)[:N_SLOTS])) for board in boards
            ]

        with then("slots contain only EMPTY or piece ids, never scalars or MASK"):
            allowed = {EMPTY} | set(range(PIECE_BASE, PIECE_BASE + 12))
            for values in slot_values:
                assert values <= allowed
                assert MASK not in values
        return


class TestSlotLayout:
    def test_start_position_slots(self, encoder):
        with given("the standard starting position"):
            board = chess.Board()

        with when("the board is encoded"):
            segment = encoder.encode(board)

        with then("slots map squares a1..h8 to content in board order"):
            assert segment[chess.A1] == piece_id(chess.Piece(chess.ROOK, chess.WHITE))
            assert segment[chess.E1] == piece_id(chess.Piece(chess.KING, chess.WHITE))
            assert segment[chess.E2] == piece_id(chess.Piece(chess.PAWN, chess.WHITE))
            assert segment[chess.E4] == EMPTY
            assert segment[chess.E7] == piece_id(chess.Piece(chess.PAWN, chess.BLACK))
            assert segment[chess.E8] == piece_id(chess.Piece(chess.KING, chess.BLACK))
            assert segment[chess.H8] == piece_id(chess.Piece(chess.ROOK, chess.BLACK))
        return

    def test_after_e4_slots_and_scalars(self, encoder):
        with given("the position after 1.e4"):
            board = chess.Board()
            board.push(chess.Move(chess.E2, chess.E4))

        with when("the board is encoded"):
            segment = encoder.encode(board)

        with then("e2 is empty, e4 holds a white pawn, and the scalars follow"):
            assert segment[chess.E2] == EMPTY
            assert segment[chess.E4] == piece_id(chess.Piece(chess.PAWN, chess.WHITE))
            assert segment[N_SLOTS] == TURN_BLACK
            assert segment[N_SLOTS + 1] == CASTLE_BASE + 15
            assert segment[N_SLOTS + 2] == EP_NONE
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

        with then("the turn slot encodes the side to move"):
            assert white_to_move[N_SLOTS] == TURN_WHITE
            assert black_to_move[N_SLOTS] == TURN_BLACK
        return


class TestEnPassantSlot:
    def test_legal_ep_emits_target_square(self, encoder):
        with given("a board with a legal en passant capture"):
            board = chess.Board("4k3/8/8/8/3pP3/8/8/4K3 b - e3 0 1")

        with when("the board is encoded"):
            segment = encoder.encode(board)

        with then("the ep slot holds the en passant target token"):
            assert segment[N_SLOTS + 2] == ep_square_token(chess.E3)
            assert ep_square_token(chess.E3) == EP_BASE + 1 + (chess.E3 - chess.A3)
        return

    def test_pseudo_legal_ep_emits_none(self, encoder):
        with given("a board whose en passant square is only pseudo-legal"):
            board = chess.Board()
            board.push(chess.Move(chess.E2, chess.E4))

        with when("the board is encoded"):
            segment = encoder.encode(board)

        with then("the ep slot holds EP_NONE and the segment stays fixed-length"):
            assert not board.has_legal_en_passant()
            assert ep_token(board) == EP_NONE
            assert segment[N_SLOTS + 2] == EP_NONE
            assert len(segment) == SEGMENT_LEN
        return

    def test_ep_none_parses_to_none(self):
        with given("the EP_NONE token and two ep target tokens"):
            none_token = EP_NONE
            e3_token = ep_square_token(chess.E3)
            a6_token = ep_square_token(chess.A6)

        with when("each token is parsed"):
            parsed = [
                parse_ep_token(none_token),
                parse_ep_token(e3_token),
                parse_ep_token(a6_token),
            ]

        with then("EP_NONE yields None and targets yield their squares"):
            assert parsed == [None, chess.E3, chess.A6]
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

    def test_sixty_random_games_deterministic(self, encoder):
        with given("an encoder and boards from a random game"):
            rng = random.Random(7)
            boards = random_game_boards(rng)

        with when("each board is encoded twice and decoded once"):
            encodings = [
                (encoder.encode(board), encoder.encode(board)) for board in boards
            ]
            decoded = [encoder.decode(first) for first, _ in encodings]

        with then("encodings are identical and roundtrips match"):
            for first, second in encodings:
                np.testing.assert_array_equal(first, second)
            for board, rebuilt in zip(boards, decoded):
                assert rebuilt.epd() == board.epd()
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

    def test_rejects_wrong_length(self, encoder):
        with given("a segment missing its final ep slot"):
            segment = encoder.encode(chess.Board())[:-1]

        with when("the segment is decoded"):
            decoded = None
            with pytest.raises(ValueError) as excinfo:
                decoded = encoder.decode(segment)

        with then("a ValueError mentioning the expected length is raised"):
            assert decoded is None
            assert str(SEGMENT_LEN) in str(excinfo.value)
        return

    def test_rejects_unknown_slot_token(self, encoder):
        with given("a valid segment with a scalar token injected into a slot"):
            segment = encoder.encode(chess.Board())
            segment[N_SLOTS - 1] = castle_token(chess.Board())

        with when("the segment is decoded"):
            decoded = None
            with pytest.raises(ValueError):
                decoded = encoder.decode(segment)

        with then("a ValueError is raised and nothing is decoded"):
            assert decoded is None
        return

    def test_rejects_unknown_scalar_token(self, encoder):
        with given("a valid segment with MASK written into the turn slot"):
            segment = encoder.encode(chess.Board())
            segment[N_SLOTS] = MASK

        with when("the segment is decoded"):
            decoded = None
            with pytest.raises(ValueError):
                decoded = encoder.decode(segment)

        with then("a ValueError is raised and nothing is decoded"):
            assert decoded is None
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
    def test_encode_batch_fixed_shape(self, encoder):
        with given("boards of different piece counts"):
            boards = [
                chess.Board(),
                chess.Board("4k3/8/8/8/8/8/8/4K3 w - - 0 1"),
            ]

        with when("the boards are batch-encoded"):
            encoded, lengths = encoder.encode_batch(boards)

        with then("the batch has the fixed segment shape and all lengths match"):
            assert encoded.dtype == np.int16
            assert encoded.shape == (2, SEGMENT_LEN)
            np.testing.assert_array_equal(lengths, [SEGMENT_LEN, SEGMENT_LEN])
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
            assert stream[SEGMENT_LEN + 1] == SEP
            assert stream[-1] == SEP
            assert len(stream) == 1 + 2 * (SEGMENT_LEN + 1)
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
