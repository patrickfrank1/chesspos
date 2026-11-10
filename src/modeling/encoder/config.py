from __future__ import annotations

from dataclasses import dataclass

from src.dataset.token_stream import SEGMENT_LEN, VOCAB_SIZE, resolve_max_segments


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
    max_segments: int | None = None
    segment_len: int = SEGMENT_LEN

    def __post_init__(self) -> None:
        if self.d_model % self.n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")
        if self.n_layers < 0:
            raise ValueError("n_layers must be non-negative")
        self.max_segments = resolve_max_segments(
            self.max_seq_len, self.max_segments, self.segment_len
        )
        return
