from __future__ import annotations

import random
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from src.dataset.puzzle_variants import split_stream
from src.dataset.token_stream import TokenStreamEncoder

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_PUZZLE_DIR = REPO_ROOT / "data/processed/puzzles"


@dataclass(frozen=True)
class PuzzleSample:
    sample_id: str
    puzzle_id: str
    position_index: int
    n_positions: int
    rating: int
    sacrifice: int
    fen: str
    label: str

    def to_dict(self) -> dict:
        return {
            "sample_id": self.sample_id,
            "puzzle_id": self.puzzle_id,
            "position_index": self.position_index,
            "n_positions": self.n_positions,
            "rating": self.rating,
            "sacrifice": self.sacrifice,
            "fen": self.fen,
            "label": self.label,
        }


@lru_cache(maxsize=8)
def load_puzzle_samples(
    data_dir: str | Path | None = None,
    max_puzzles: int = 60,
    seed: int = 17,
    shard_limit: int = 5,
) -> list[PuzzleSample]:
    """Sample puzzle positions from the parquet shards.

    The sample includes the position before the puzzle solution and each
    position after a solution move. Positives and negatives are sampled
    separately so the dropdown is not dominated by the 7.5% sacrifice base
    rate.
    """
    root = Path(data_dir) if data_dir is not None else DEFAULT_PUZZLE_DIR
    if max_puzzles < 1:
        raise ValueError("max_puzzles must be positive")
    if not root.exists():
        return []
    files = sorted(root.rglob("*.parquet"))[: max(1, shard_limit)]
    if not files:
        return []

    rng = random.Random(seed)
    positives: list[dict] = []
    negatives: list[dict] = []
    per_shard = max(10, max_puzzles)
    for path in files:
        table = pq.read_table(
            path,
            columns=[
                "puzzle_id",
                "previous_position",
                "packed",
                "n_positions",
                "rating",
                "sacrifice",
            ],
        )
        labels = table.column("sacrifice").to_numpy(zero_copy_only=False)
        for value, bucket in ((1, positives), (0, negatives)):
            indices = np.flatnonzero(labels == value)
            if indices.size == 0:
                continue
            take = min(per_shard, indices.size)
            chosen = rng.sample(indices.tolist(), take)
            bucket.extend(table.take(chosen).to_pylist())
        if len(positives) >= per_shard and len(negatives) >= per_shard:
            break

    half = max(1, max_puzzles // 2)
    chosen_rows = rng.sample(positives, min(half, len(positives)))
    chosen_rows += rng.sample(negatives, min(half, len(negatives)))
    rng.shuffle(chosen_rows)

    encoder = TokenStreamEncoder()
    samples: dict[str, PuzzleSample] = {}
    for row in chosen_rows:
        packed = row.get("packed")
        if not packed:
            continue
        puzzle_id = str(row.get("puzzle_id") or "")
        rating = int(row.get("rating") or 0)
        sacrifice = int(row.get("sacrifice") or 0)
        segments = split_stream(np.asarray(packed, dtype=np.int16))
        n_positions = len(segments)
        positions: list[tuple[int, np.ndarray]] = []
        previous = row.get("previous_position")
        if previous:
            positions.append((0, np.asarray(previous, dtype=np.int16)))
        for index, segment in enumerate(segments, start=1):
            positions.append((index, np.asarray(segment, dtype=np.int16)))

        for position_index, segment in positions:
            try:
                board = encoder.decode(segment)
            except (ValueError, TypeError):
                continue
            sample_id = f"{puzzle_id}:{position_index}"
            if sample_id in samples:
                continue
            position_label = (
                "start"
                if position_index == 0
                else f"move {position_index}/{n_positions}"
            )
            theme_label = "sacrifice" if sacrifice else "no sacrifice"
            samples[sample_id] = PuzzleSample(
                sample_id=sample_id,
                puzzle_id=puzzle_id,
                position_index=position_index,
                n_positions=n_positions,
                rating=rating,
                sacrifice=sacrifice,
                fen=board.fen(),
                label=(f"{puzzle_id} · {position_label} · {rating} · {theme_label}"),
            )

    return sorted(
        samples.values(),
        key=lambda sample: (-sample.sacrifice, sample.puzzle_id, sample.position_index),
    )
