import chess
import numpy as np
import pytest

from src.dataset.encoder import (
    BitboardEncoder,
    PositionEncoder,
    TensorEncoder,
    TokenSequenceEncoder,
    get_encoder,
    register_encoder,
)
from src.dataset.types import BITBOARD, TENSOR, TOKEN_SEQUENCE


class TestTokenSequenceEncoder:
    def test_encoding_format(self):
        encoder = TokenSequenceEncoder()
        assert encoder.encoding_format == TOKEN_SEQUENCE

    def test_output_shape(self):
        encoder = TokenSequenceEncoder()
        assert encoder.output_shape == (69,)

    def test_encode_returns_correct_shape(self):
        encoder = TokenSequenceEncoder()
        board = chess.Board()
        encoded = encoder.encode(board)
        assert encoded.shape == (69,)
        assert encoded.dtype == np.int8

    def test_encode_batch_returns_correct_shape(self):
        encoder = TokenSequenceEncoder()
        boards = [chess.Board(), chess.Board()]
        encoded = encoder.encode_batch(boards)
        assert encoded.shape == (2, 69)
        assert encoded.dtype == np.int8

    def test_encode_different_positions(self):
        encoder = TokenSequenceEncoder()
        board1 = chess.Board()
        board2 = chess.Board()
        board2.push_san("e4")

        encoded1 = encoder.encode(board1)
        encoded2 = encoder.encode(board2)

        assert not np.array_equal(encoded1, encoded2)

    def test_decode_reconstructs_board(self):
        encoder = TokenSequenceEncoder()
        board = chess.Board()
        encoded = encoder.encode(board)
        decoded = encoder.decode(encoded)
        assert decoded.fen() == board.fen()

    def test_decode_batch_reconstructs_boards(self):
        encoder = TokenSequenceEncoder()
        boards = [
            chess.Board(),
            chess.Board("rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1"),
        ]
        encoded = encoder.encode_batch(boards)
        decoded = encoder.decode_batch(encoded)
        assert decoded[0].fen() == boards[0].fen()
        assert decoded[1].fen() == boards[1].fen()


class TestTensorEncoder:
    def test_encoding_format(self):
        encoder = TensorEncoder()
        assert encoder.encoding_format == TENSOR

    def test_output_shape(self):
        encoder = TensorEncoder()
        assert encoder.output_shape == (8, 8, 15)

    def test_encode_returns_correct_shape(self):
        encoder = TensorEncoder()
        board = chess.Board()
        encoded = encoder.encode(board)
        assert encoded.shape == (8, 8, 15)
        assert encoded.dtype == bool

    def test_encode_batch_returns_correct_shape(self):
        encoder = TensorEncoder()
        boards = [chess.Board(), chess.Board()]
        encoded = encoder.encode_batch(boards)
        assert encoded.shape == (2, 8, 8, 15)
        assert encoded.dtype == bool

    def test_decode_reconstructs_board(self):
        encoder = TensorEncoder()
        board = chess.Board()
        encoded = encoder.encode(board)
        decoded = encoder.decode(encoded)
        assert decoded.fen() == board.fen()

    def test_decode_batch_reconstructs_boards(self):
        encoder = TensorEncoder()
        boards = [
            chess.Board(),
            chess.Board("rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1"),
        ]
        encoded = encoder.encode_batch(boards)
        decoded = encoder.decode_batch(encoded)
        assert decoded[0].fen() == boards[0].fen()
        assert decoded[1].fen() == boards[1].fen()


class TestBitboardEncoder:
    def test_encoding_format(self):
        encoder = BitboardEncoder()
        assert encoder.encoding_format == BITBOARD

    def test_output_shape(self):
        encoder = BitboardEncoder()
        assert encoder.output_shape == (773,)

    def test_encode_returns_correct_shape(self):
        encoder = BitboardEncoder()
        board = chess.Board()
        encoded = encoder.encode(board)
        assert encoded.shape == (773,)
        assert encoded.dtype == bool

    def test_decode_reconstructs_board(self):
        encoder = BitboardEncoder()
        board = chess.Board()
        encoded = encoder.encode(board)
        decoded = encoder.decode(encoded)
        assert decoded.fen() == board.fen()

    def test_decode_batch_reconstructs_boards(self):
        encoder = BitboardEncoder()
        boards = [
            chess.Board(),
            chess.Board("rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1"),
        ]
        encoded = encoder.encode_batch(boards)
        decoded = encoder.decode_batch(encoded)
        assert decoded[0].fen() == boards[0].fen()
        assert decoded[1].fen() == boards[1].fen()


class TestGetEncoder:
    def test_get_token_sequence_encoder(self):
        encoder = get_encoder(TOKEN_SEQUENCE)
        assert isinstance(encoder, TokenSequenceEncoder)

    def test_get_tensor_encoder(self):
        encoder = get_encoder(TENSOR)
        assert isinstance(encoder, TensorEncoder)

    def test_get_bitboard_encoder(self):
        encoder = get_encoder(BITBOARD)
        assert isinstance(encoder, BitboardEncoder)

    def test_get_unknown_encoder_raises_error(self):
        with pytest.raises(ValueError, match="Unknown encoding format"):
            get_encoder("unknown_format")


class TestRegisterEncoder:
    def test_register_custom_encoder(self):
        class CustomEncoder(PositionEncoder):
            @property
            def encoding_format(self):
                return "custom"

            @property
            def output_shape(self):
                return (10,)

            def encode(self, board):
                return np.zeros(10, dtype=np.int8)

            def encode_batch(self, boards):
                return np.zeros((len(boards), 10), dtype=np.int8)

            def decode(self, data):
                return chess.Board()

            def decode_batch(self, data):
                return [chess.Board() for _ in data]

        register_encoder("custom", CustomEncoder)
        encoder = get_encoder("custom")
        assert isinstance(encoder, CustomEncoder)
