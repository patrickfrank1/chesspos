from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch
from torch.utils.data import Dataset

from src.dataset.token_stream import CLS, PAD, SEP
from src.modeling.masking import IGNORE_INDEX, MaskingConfig, mask_stream


@dataclass
class WindowDatasetConfig:
    data_dir: str
    max_seq_len: int = 2048
    max_segments: int = 128
    p_single: float = 0.15
    p_few: float = 0.25
    few_min: int = 2
    few_max: int = 12
    masking: MaskingConfig = field(default_factory=MaskingConfig)
    seed: int | None = None
    apply_masking: bool = True


class WindowDataset(Dataset):
    def __init__(self, config: WindowDatasetConfig):
        self.config = config
        self.games: list[list[np.ndarray]] = []
        for path in sorted(Path(config.data_dir).glob("*.parquet")):
            self.games.extend(_load_game_segments(path))

    def __len__(self) -> int:
        return len(self.games)

    def __getitem__(self, index: int) -> dict[str, np.ndarray]:
        config = self.config
        seed = None if config.seed is None else config.seed + index
        rng = np.random.default_rng(seed)
        segments = self._sample_segments(self.games[index], rng)
        tokens = _pack(segments)
        if config.apply_masking:
            masked, targets = mask_stream(tokens, config.masking, rng)
        else:
            masked = tokens
            targets = np.full(tokens.shape, IGNORE_INDEX, dtype=np.int16)
        return {"tokens": masked, "targets": targets}

    def _sample_segments(
        self, segments: list[np.ndarray], rng: np.random.Generator
    ) -> list[np.ndarray]:
        config = self.config
        n = len(segments)
        u = rng.random()
        if u < config.p_single:
            start = int(rng.integers(n))
            return [segments[start]]
        if u < config.p_single + config.p_few:
            k = int(rng.integers(config.few_min, min(config.few_max, n) + 1))
            start = int(rng.integers(n - k + 1))
            return segments[start : start + k]
        return self._sample_full(segments, rng)

    def _sample_full(
        self, segments: list[np.ndarray], rng: np.random.Generator
    ) -> list[np.ndarray]:
        budget = self.config.max_seq_len - 1
        lengths = np.fromiter((len(s) for s in segments), dtype=np.int64)
        total = int(lengths.sum() + len(segments))
        if total <= budget and len(segments) <= self.config.max_segments:
            return list(segments)
        start = int(rng.integers(len(segments)))
        chosen: list[np.ndarray] = []
        used = 0
        for i in range(start, len(segments)):
            cost = int(lengths[i]) + 1
            if used + cost > budget or len(chosen) >= self.config.max_segments:
                break
            chosen.append(segments[i])
            used += cost
        if not chosen:
            chosen = [segments[start][: budget - 1]]
        return chosen


def _load_game_segments(path: Path) -> list[list[np.ndarray]]:
    table = pq.read_table(path, columns=["packed"])
    column = table.column("packed").combine_chunks()
    if len(column) == 0:
        return []
    offsets = np.asarray(column.offsets)
    values = column.flatten().to_numpy(zero_copy_only=False).astype(np.int16)
    games = []
    for i in range(len(column)):
        stream = values[offsets[i] : offsets[i + 1]]
        segments = _split_segments(stream)
        if segments:
            games.append(segments)
    return games


def _split_segments(stream: np.ndarray) -> list[np.ndarray]:
    sep_positions = np.flatnonzero(stream == SEP)
    segments = []
    start = 1 if stream.size and stream[0] == CLS else 0
    for sep in sep_positions:
        if start < sep:
            segments.append(stream[start:sep])
        start = sep + 1
    if start < stream.size:
        segments.append(stream[start:])
    return segments


def _pack(segments: list[np.ndarray]) -> np.ndarray:
    total = 1 + sum(len(s) + 1 for s in segments)
    tokens = np.empty(total, dtype=np.int16)
    tokens[0] = CLS
    position = 1
    for segment in segments:
        tokens[position : position + len(segment)] = segment
        position += len(segment)
        tokens[position] = SEP
        position += 1
    return tokens


def collate_windows(batch: list[dict[str, np.ndarray]]) -> dict[str, torch.Tensor]:
    lengths = [len(sample["tokens"]) for sample in batch]
    max_len = max(lengths)
    n = len(batch)
    tokens = np.full((n, max_len), PAD, dtype=np.int16)
    targets = np.full((n, max_len), IGNORE_INDEX, dtype=np.int16)
    for i, sample in enumerate(batch):
        tokens[i, : len(sample["tokens"])] = sample["tokens"]
        targets[i, : len(sample["targets"])] = sample["targets"]
    segment_ids = np.cumsum(tokens == SEP, axis=1, dtype=np.int64)
    return {
        "tokens": torch.from_numpy(tokens.astype(np.int64)),
        "targets": torch.from_numpy(targets.astype(np.int64)),
        "segment_ids": torch.from_numpy(segment_ids),
        "padding_mask": torch.from_numpy(tokens != PAD),
    }
