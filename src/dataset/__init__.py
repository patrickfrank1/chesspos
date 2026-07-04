import chess
import numpy as np

from src.dataset.config import DatasetConfig, EncoderConfig, PreprocessingConfig
from src.dataset.position_encoder import (
    BitboardEncoder,
    PositionEncoder,
    TensorEncoder,
    TokenSequenceEncoder,
    get_encoder,
    register_encoder,
)
from src.dataset.pgn_processor import GameRecord, PGNProcessor
from src.dataset.huggingface_client import HuggingFaceClient
from src.dataset.etl import ChessPositionDataset
from src.dataset.data_loader import TrainingDataGenerator

_token_encoder = TokenSequenceEncoder()
_tensor_encoder = TensorEncoder()
_bitboard_encoder = BitboardEncoder()


def board_to_token_sequence(board: chess.Board) -> np.ndarray:
    return _token_encoder.encode(board)


def token_sequence_to_board(data: np.ndarray) -> chess.Board:
    return _token_encoder.decode(data)


def board_to_tensor(board: chess.Board) -> np.ndarray:
    return _tensor_encoder.encode(board)


def tensor_to_board(data: np.ndarray) -> chess.Board:
    return _tensor_encoder.decode(data)


def board_to_bitboard(board: chess.Board) -> np.ndarray:
    return _bitboard_encoder.encode(board)


def bitboard_to_board(data: np.ndarray) -> chess.Board:
    return _bitboard_encoder.decode(data)


__all__ = [
    "BitboardEncoder",
    "ChessPositionDataset",
    "DatasetConfig",
    "EncoderConfig",
    "GameRecord",
    "HuggingFaceClient",
    "PGNProcessor",
    "PositionEncoder",
    "PreprocessingConfig",
    "TensorEncoder",
    "TokenSequenceEncoder",
    "TrainingDataGenerator",
    "bitboard_to_board",
    "board_to_bitboard",
    "board_to_tensor",
    "board_to_token_sequence",
    "get_encoder",
    "register_encoder",
    "tensor_to_board",
    "token_sequence_to_board",
]
