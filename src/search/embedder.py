"""Checkpoint-backed position embedder."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Sequence

import numpy as np
import torch

from src.evaluation.core.embeddings import embed_tokens
from src.modeling.frozen import embedding_dim
from src.run.train_puzzle_classifier import load_frozen_encoder

FEATURE = "native_positions"


def file_sha256(path: str | Path, chunk_size: int = 1 << 20) -> str:
    """Return the hexadecimal SHA-256 digest of a file."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


class CheckpointEmbedder:
    """Embed positions with a frozen checkpoint and normalize the vectors."""

    def __init__(
        self,
        checkpoint: str | Path,
        *,
        device: str | None = None,
        batch_size: int = 256,
        bf16: bool | None = None,
    ) -> None:
        self.checkpoint = str(checkpoint)
        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.batch_size = batch_size
        self.bf16 = bf16
        self.model = load_frozen_encoder(self.checkpoint, self.device)
        self.dim = embedding_dim(self.model, FEATURE)
        self.vocab_size = int(getattr(self.model.config, "vocab_size", 0)) or None
        return

    def embed(self, segments: Sequence[np.ndarray]) -> np.ndarray:
        segments = list(segments)
        if not segments:
            return np.zeros((0, self.dim), dtype=np.float32)
        vectors = embed_tokens(
            self.model,
            segments,
            self.device,
            batch_size=self.batch_size,
            bf16=self.bf16,
        )
        return vectors.numpy().astype(np.float32)
