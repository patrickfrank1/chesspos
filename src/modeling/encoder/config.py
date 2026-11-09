from __future__ import annotations

from dataclasses import dataclass

from src.dataset.token_stream import SEGMENT_LEN, VOCAB_SIZE


@dataclass
class EncoderConfig:
    """Token encoder hyperparameters shared by Perceiver variants."""

    vocab_size: int = VOCAB_SIZE
    d_model: int = 512
    n_heads: int = 16
    n_layers: int = 8
    d_ff: int = 2048
    dropout: float = 0.1
    max_seq_len: int = 2048
    max_segments: int = 128
    segment_len: int = SEGMENT_LEN

    def __post_init__(self) -> None:
        if self.d_model % self.n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")
        if self.n_layers < 0:
            raise ValueError("n_layers must be non-negative")
        if self.max_seq_len <= 0:
            raise ValueError("max_seq_len must be positive")
        if self.max_segments <= 0:
            raise ValueError("max_segments must be positive")
        if self.segment_len <= 0:
            raise ValueError("segment_len must be positive")
        return
