from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch

from src.evaluation.eval_suite import embed_segments
from src.evaluation.observability.samples import PositionSample
from src.modeling.transformer import MaskedStreamTransformer


def embed_samples(
    model: MaskedStreamTransformer,
    samples: Sequence[PositionSample],
    device: torch.device | str,
    *,
    batch_size: int = 256,
    bf16: bool | None = None,
) -> np.ndarray:
    """Embed position samples and return L2-normalized float32 vectors."""
    device = torch.device(device) if isinstance(device, str) else device
    segments = [np.asarray(sample.tokens, dtype=np.int16) for sample in samples]
    if not segments:
        return np.zeros((0, 0), dtype=np.float32)
    vectors = embed_segments(model, segments, device, batch_size=batch_size, bf16=bf16)
    return vectors.cpu().numpy().astype(np.float32)


def save_samples(path: str | Path, samples: Sequence[PositionSample]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    table = pa.table(
        {
            "sample_id": pa.array([sample.sample_id for sample in samples], pa.int64()),
            "source": pa.array([sample.source for sample in samples], pa.string()),
            "fen": pa.array([sample.fen for sample in samples], pa.string()),
            "game_id": pa.array([sample.game_id for sample in samples], pa.string()),
            "puzzle_id": pa.array(
                [sample.puzzle_id for sample in samples], pa.string()
            ),
            "ply": pa.array([sample.ply for sample in samples], pa.int32()),
            "rating": pa.array([sample.rating for sample in samples], pa.int32()),
            "themes": pa.array(
                [list(sample.themes) for sample in samples],
                pa.list_(pa.string()),
            ),
            "tokens": pa.array(
                [list(sample.tokens) for sample in samples],
                pa.list_(pa.int16()),
            ),
            "surrogates": pa.array(
                [json.dumps(sample.surrogates, sort_keys=True) for sample in samples],
                pa.string(),
            ),
        }
    )
    pq.write_table(table, path)
    return path


def save_embeddings(path: str | Path, embeddings: np.ndarray) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, np.asarray(embeddings, dtype=np.float32))
    return path
