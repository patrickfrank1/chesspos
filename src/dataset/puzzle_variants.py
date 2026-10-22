from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Sequence

import numpy as np
import pyarrow.parquet as pq
import torch
from torch.utils.data import Dataset

from src.dataset.token_stream import CLS, PAD, SEP

_MASK64 = (1 << 64) - 1
_COPRIMES: dict[int, np.ndarray] = {}


@dataclass
class PuzzleRecord:
    puzzle_id: str
    segments: list[np.ndarray]
    label: int

    def __post_init__(self):
        if not self.segments:
            raise ValueError(f"Puzzle {self.puzzle_id} has no positions")
        if self.label not in (0, 1):
            raise ValueError(f"Puzzle {self.puzzle_id} has a non-binary label")
        return


@dataclass
class EpochPlan:
    puzzle_index: np.ndarray
    start: np.ndarray
    end: np.ndarray
    label: np.ndarray

    def __len__(self) -> int:
        return int(self.label.shape[0])


def variant_count(n_positions: int) -> int:
    """Number of contiguous subsequences of a sequence of n positions."""
    return n_positions * (n_positions + 1) // 2


def iter_variants(n_positions: int) -> Iterator[tuple[int, int]]:
    for start in range(n_positions):
        for end in range(start, n_positions):
            yield start, end


def split_for_puzzle(puzzle_id: str, val_ratio: float = 0.1) -> str:
    if not 0 < val_ratio < 1:
        raise ValueError("val_ratio must be between 0 and 1")
    digest = hashlib.sha1(puzzle_id.encode("utf-8")).digest()
    uniform = int.from_bytes(digest[:8], "big") / 2**64
    return "val" if uniform < val_ratio else "train"


def _coprimes(count: int) -> np.ndarray:
    if count <= 1:
        return np.array([1], dtype=np.int64)
    cached = _COPRIMES.get(count)
    if cached is None:
        cached = np.array(
            [value for value in range(1, count) if np.gcd(value, count) == 1],
            dtype=np.int64,
        )
        _COPRIMES[count] = cached
    return cached


def _hash64(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.uint64)
    with np.errstate(over="ignore"):
        values = values + np.uint64(0x9E3779B97F4A7C15)
        values = (values ^ (values >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
        values = (values ^ (values >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
        values = values ^ (values >> np.uint64(31))
    return values


def _mix_seed(
    seed: int, puzzle_indices: np.ndarray, cycles: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    with np.errstate(over="ignore"):
        base = (
            np.uint64(seed & _MASK64)
            ^ (puzzle_indices.astype(np.uint64) * np.uint64(0x9E3779B97F4A7C15))
            ^ (cycles.astype(np.uint64) * np.uint64(0xD1B54A32D192ED03))
        )
    first = _hash64(base)
    second = _hash64(base ^ np.uint64(0xA0761D6478BD642F))
    return first, second


def _local_ids_to_span(
    local_ids: np.ndarray, lengths: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Map local variant ids (start-major order) to inclusive start/end spans."""
    local_ids = np.asarray(local_ids, dtype=np.int64)
    lengths = np.asarray(lengths, dtype=np.int64)
    starts = np.zeros(local_ids.shape[0], dtype=np.int64)
    if local_ids.size:
        max_start = int(lengths.max())
        for start in range(max_start):
            valid = start < lengths
            offsets = start * lengths - start * (start - 1) // 2
            starts[valid & (local_ids >= offsets)] = start
    bases = starts * lengths - starts * (starts - 1) // 2
    ends = starts + (local_ids - bases)
    return starts.astype(np.int16), ends.astype(np.int16)


class RollingVariantPlanner:
    """Builds unique contiguous-subsequence samples across epochs.

    Positives draw from a global permutation of their variant space. Negatives
    use each puzzle once per epoch and advance a per-puzzle cursor through a
    seeded permutation of that puzzle's variants. No configuration repeats
    until the relevant variant space has been exhausted.
    """

    def __init__(
        self,
        records: Sequence[PuzzleRecord],
        positive_fraction: float = 0.5,
        seed: int = 0,
    ):
        if not 0 < positive_fraction < 1:
            raise ValueError("positive_fraction must be between 0 and 1")

        self.records = list(records)
        self.positive_fraction = positive_fraction
        self.seed = seed

        self.positive_indices = np.array(
            [i for i, record in enumerate(self.records) if record.label == 1],
            dtype=np.int32,
        )
        self.negative_indices = np.array(
            [i for i, record in enumerate(self.records) if record.label == 0],
            dtype=np.int32,
        )
        if self.negative_indices.size == 0:
            raise ValueError("RollingVariantPlanner requires at least one negative")

        self.positive_lengths = np.array(
            [len(self.records[i].segments) for i in self.positive_indices],
            dtype=np.int64,
        )
        self.negative_lengths = np.array(
            [len(self.records[i].segments) for i in self.negative_indices],
            dtype=np.int64,
        )
        self.positive_counts = np.array(
            [variant_count(int(length)) for length in self.positive_lengths],
            dtype=np.int64,
        )
        self.negative_counts = np.array(
            [variant_count(int(length)) for length in self.negative_lengths],
            dtype=np.int64,
        )
        self.positive_offsets = np.concatenate(
            [np.zeros(1, dtype=np.int64), np.cumsum(self.positive_counts)]
        )
        self.total_positive_variants = int(self.positive_offsets[-1])

        self.n_neg = int(self.negative_indices.size)
        self.n_pos = int(
            round(self.n_neg * positive_fraction / (1.0 - positive_fraction))
        )
        if self.n_pos > self.total_positive_variants:
            raise ValueError(
                "positive_fraction requests more positive samples than the "
                f"variant space holds ({self.n_pos} > {self.total_positive_variants})"
            )

        self._positive_cycle = 0
        self._positive_cursor = 0
        self._positive_permutation = self._make_positive_permutation(0)
        self._negative_cursors = np.zeros(self.n_neg, dtype=np.int32)
        self._negative_cycles = np.zeros(self.n_neg, dtype=np.int32)
        self._epoch = 0
        return

    def _make_positive_permutation(self, cycle: int) -> np.ndarray:
        rng = np.random.default_rng(self.seed + cycle * 0x9E3779B1)
        return rng.permutation(self.total_positive_variants).astype(np.int32)

    def _take_positive_ids(self, count: int) -> np.ndarray:
        ids = np.empty(count, dtype=np.int64)
        filled = 0
        while filled < count:
            if self._positive_cursor >= self.total_positive_variants:
                self._positive_cycle += 1
                self._positive_cursor = 0
                self._positive_permutation = self._make_positive_permutation(
                    self._positive_cycle
                )
            remaining = self.total_positive_variants - self._positive_cursor
            take = min(remaining, count - filled)
            ids[filled : filled + take] = self._positive_permutation[
                self._positive_cursor : self._positive_cursor + take
            ]
            filled += take
            self._positive_cursor += take
        if self._positive_cursor >= self.total_positive_variants:
            self._positive_cycle += 1
            self._positive_cursor = 0
            self._positive_permutation = self._make_positive_permutation(
                self._positive_cycle
            )
        return ids

    def _positive_triplets(
        self, ids: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        positions = np.searchsorted(self.positive_offsets, ids, side="right") - 1
        local_ids = ids - self.positive_offsets[positions]
        lengths = self.positive_lengths[positions]
        starts, ends = _local_ids_to_span(local_ids, lengths)
        return self.positive_indices[positions], starts, ends

    def _negative_local_ids(self) -> np.ndarray:
        first, second = _mix_seed(
            self.seed, self.negative_indices, self._negative_cycles
        )
        local_ids = np.zeros(self.n_neg, dtype=np.int64)
        for count in np.unique(self.negative_counts):
            mask = self.negative_counts == count
            if count == 1:
                local_ids[mask] = 0
                continue
            coprimes = _coprimes(int(count))
            multiplier = coprimes[(first[mask] % len(coprimes)).astype(np.int64)]
            offset = (second[mask] % count).astype(np.int64)
            local_ids[mask] = (
                multiplier * self._negative_cursors[mask] + offset
            ) % count

        self._negative_cursors += 1
        wrapped = self._negative_cursors >= self.negative_counts
        self._negative_cursors[wrapped] = 0
        self._negative_cycles[wrapped] += 1
        return local_ids

    def next_epoch(self) -> EpochPlan:
        positive_ids = self._take_positive_ids(self.n_pos)
        positive_index, positive_start, positive_end = self._positive_triplets(
            positive_ids
        )

        negative_local = self._negative_local_ids()
        negative_start, negative_end = _local_ids_to_span(
            negative_local, self.negative_lengths
        )

        puzzle_index = np.concatenate([positive_index, self.negative_indices]).astype(
            np.int32
        )
        start = np.concatenate([positive_start, negative_start]).astype(np.int16)
        end = np.concatenate([positive_end, negative_end]).astype(np.int16)
        label = np.concatenate(
            [
                np.ones(positive_index.size, dtype=np.int8),
                np.zeros(self.n_neg, dtype=np.int8),
            ]
        )

        rng = np.random.default_rng(self.seed + self._epoch * 0x85EBCA6B)
        order = rng.permutation(label.size)
        self._epoch += 1
        return EpochPlan(
            puzzle_index=puzzle_index[order],
            start=start[order],
            end=end[order],
            label=label[order],
        )

    def state_dict(self) -> dict:
        return {
            "positive_cycle": self._positive_cycle,
            "positive_cursor": self._positive_cursor,
            "negative_cursors": self._negative_cursors.copy(),
            "negative_cycles": self._negative_cycles.copy(),
            "epoch": self._epoch,
        }

    def load_state_dict(self, state: dict) -> None:
        self._positive_cycle = int(state["positive_cycle"])
        self._positive_cursor = int(state["positive_cursor"])
        self._positive_permutation = self._make_positive_permutation(
            self._positive_cycle
        )
        self._negative_cursors = np.asarray(
            state["negative_cursors"], dtype=np.int32
        ).copy()
        self._negative_cycles = np.asarray(
            state["negative_cycles"], dtype=np.int32
        ).copy()
        self._epoch = int(state["epoch"])
        return


def build_epoch_plan(
    records: Sequence[PuzzleRecord],
    positive_fraction: float = 0.5,
    seed: int = 0,
) -> EpochPlan:
    return RollingVariantPlanner(
        records, positive_fraction=positive_fraction, seed=seed
    ).next_epoch()


def pack_segments(segments: Sequence[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    if not segments:
        raise ValueError("Cannot pack an empty sequence")
    total = 1 + sum(len(segment) + 1 for segment in segments)
    tokens = np.empty(total, dtype=np.int16)
    tokens[0] = CLS
    position = 1
    for segment in segments:
        segment = np.asarray(segment, dtype=np.int16)
        tokens[position : position + len(segment)] = segment
        position += len(segment)
        tokens[position] = SEP
        position += 1
    segment_ids = np.cumsum(tokens == SEP, dtype=np.int64)
    return tokens, segment_ids


class PuzzleVariantDataset(Dataset):
    def __init__(self, records: Sequence[PuzzleRecord], plan: EpochPlan):
        self.records = list(records)
        self.plan = plan
        return

    def __len__(self) -> int:
        return len(self.plan)

    def __getitem__(self, index: int) -> dict:
        record = self.records[int(self.plan.puzzle_index[index])]
        start = int(self.plan.start[index])
        end = int(self.plan.end[index])
        tokens, segment_ids = pack_segments(record.segments[start : end + 1])
        return {
            "tokens": tokens,
            "segment_ids": segment_ids,
            "label": int(self.plan.label[index]),
            "puzzle_id": record.puzzle_id,
        }


def collate_variants(batch: list[dict]) -> dict:
    lengths = [len(sample["tokens"]) for sample in batch]
    max_len = max(lengths)
    tokens = np.full((len(batch), max_len), PAD, dtype=np.int16)
    segment_ids = np.zeros((len(batch), max_len), dtype=np.int64)
    padding_mask = np.zeros((len(batch), max_len), dtype=bool)
    labels = np.zeros(len(batch), dtype=np.float32)
    for i, sample in enumerate(batch):
        length = len(sample["tokens"])
        tokens[i, :length] = sample["tokens"]
        segment_ids[i, :length] = sample["segment_ids"]
        padding_mask[i, :length] = True
        labels[i] = sample["label"]
    return {
        "tokens": torch.from_numpy(tokens.astype(np.int64)),
        "segment_ids": torch.from_numpy(segment_ids),
        "padding_mask": torch.from_numpy(padding_mask),
        "labels": torch.from_numpy(labels),
        "puzzle_ids": [sample["puzzle_id"] for sample in batch],
    }


def _split_stream(stream: np.ndarray) -> list[np.ndarray]:
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


def load_puzzle_records(
    data_dir: str | Path, theme: str, limit: int | None = None
) -> list[PuzzleRecord]:
    records: list[PuzzleRecord] = []
    for path in sorted(Path(data_dir).rglob("*.parquet")):
        table = pq.read_table(path, columns=["puzzle_id", "packed", theme])
        if theme not in table.schema.names:
            raise ValueError(f"Column {theme!r} not found in {path}")
        puzzle_ids = table.column("puzzle_id").to_pylist()
        labels = table.column(theme).to_pylist()
        packed = table.column("packed").combine_chunks()
        offsets = np.asarray(packed.offsets)
        values = packed.flatten().to_numpy(zero_copy_only=False).astype(np.int16)
        for i, (puzzle_id, label) in enumerate(zip(puzzle_ids, labels)):
            segments = _split_stream(values[offsets[i] : offsets[i + 1]])
            if not segments:
                continue
            records.append(
                PuzzleRecord(
                    puzzle_id=str(puzzle_id),
                    segments=segments,
                    label=int(label),
                )
            )
            if limit is not None and len(records) >= limit:
                return records
    return records
