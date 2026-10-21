from typing import TYPE_CHECKING

from src.dataset.config import DatasetConfig, PreprocessingConfig
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
    pack_stream,
)

if TYPE_CHECKING:
    from src.dataset.etl import ChessPositionDataset
    from src.dataset.huggingface_client import HuggingFaceClient
    from src.dataset.pgn_processor import GameRecord, PGNProcessor

_LAZY_IMPORTS = {
    "ChessPositionDataset": ("src.dataset.etl", "ChessPositionDataset"),
    "GameRecord": ("src.dataset.pgn_processor", "GameRecord"),
    "HuggingFaceClient": ("src.dataset.huggingface_client", "HuggingFaceClient"),
    "PGNProcessor": ("src.dataset.pgn_processor", "PGNProcessor"),
}


def __getattr__(name: str):
    if name in _LAZY_IMPORTS:
        import importlib

        module_name, attribute = _LAZY_IMPORTS[name]
        return getattr(importlib.import_module(module_name), attribute)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "CASTLE_BASE",
    "CLS",
    "ChessPositionDataset",
    "DatasetConfig",
    "EMPTY",
    "EP_BASE",
    "EP_NONE",
    "GameRecord",
    "HuggingFaceClient",
    "MASK",
    "N_SLOTS",
    "PAD",
    "PGNProcessor",
    "PIECE_BASE",
    "PreprocessingConfig",
    "SEGMENT_LEN",
    "SEP",
    "TURN_BLACK",
    "TURN_WHITE",
    "TokenStreamEncoder",
    "VOCAB_SIZE",
    "pack_stream",
]
