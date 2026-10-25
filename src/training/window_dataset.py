from __future__ import annotations

import json
import os
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from queue import Queue
from threading import Thread

import numpy as np
import pyarrow.parquet as pq
import torch
from torch.utils.data import Dataset, IterableDataset

from src.dataset.token_stream import CLS, PAD, SEP, VOCAB_SIZE
from src.modeling.masking import IGNORE_INDEX, MaskingConfig, mask_stream


@dataclass
class WindowDatasetConfig:
    data_dir: str = ""
    games: list[list[np.ndarray]] | None = None
    max_seq_len: int = 2048
    max_segments: int = 128
    p_single: float = 0.15
    p_few: float = 0.25
    few_min: int = 2
    few_max: int = 12
    masking: MaskingConfig = field(default_factory=MaskingConfig)
    seed: int | None = None
    apply_masking: bool = True
    force_mode: str | None = None
    balanced_modes: bool = True
    unigram_weights: np.ndarray | None = None
    shuffle_buffer: int = 0
    prefetch_shards: int = 0
    resume_step: int = 0
    batch_size: int = 1
    num_workers: int = 1


def worker_skip_windows(
    resume_step: int, batch_size: int, num_workers: int, worker_id: int
) -> int:
    """Windows this worker already consumed within the first ``resume_step`` steps.

    A DataLoader with ``num_workers`` serves batches round-robin: batch ``i``
    comes from worker ``i % num_workers``. After ``resume_step`` steps the
    worker has produced ``ceil((resume_step - worker_id) / num_workers)``
    batches, i.e. that many times ``batch_size`` windows. With a single
    process (``num_workers <= 1``) the skip covers every window consumed so
    far. An exact stream position further requires the same ``num_workers``
    as the original run. The resumed loader restarts its round-robin at
    worker 0, so batch order may rotate within one worker cycle; the window
    multiset is preserved exactly.
    """
    if resume_step <= 0 or worker_id >= resume_step:
        return 0
    if num_workers <= 1:
        return resume_step * batch_size
    batches = (resume_step - worker_id - 1) // num_workers + 1
    return batches * batch_size


def build_mode_pattern(masking: MaskingConfig, period: int = 20) -> list[str]:
    """Deterministic per-index mode pattern approximating the mode probabilities."""
    probs = {
        "random": masking.p_random,
        "board": masking.p_board,
        "span": masking.p_span,
        "rect": masking.p_rect,
    }
    total = sum(probs.values())
    if total <= 0:
        return ["random"] * period
    scaled = {mode: p / total * period for mode, p in probs.items()}
    counts = {mode: int(value) for mode, value in scaled.items()}
    order = sorted(scaled, key=lambda mode: scaled[mode] - counts[mode], reverse=True)
    fill = 0
    while sum(counts.values()) < period:
        counts[order[fill % len(order)]] += 1
        fill += 1
    pattern: list[str] = []
    for mode in ("random", "board", "span", "rect"):
        pattern.extend([mode] * counts[mode])
    return pattern


def unigram_counts(games: list[list[np.ndarray]]) -> np.ndarray:
    counts = np.zeros(VOCAB_SIZE, dtype=np.int64)
    for game in games:
        for segment in game:
            counts += np.bincount(segment, minlength=VOCAB_SIZE)
    return counts


def inverse_frequency_weights(
    counts: np.ndarray, alpha: float = 1.0, cap: float = 100.0
) -> np.ndarray:
    counts = np.asarray(counts, dtype=np.float64)
    p = (counts + alpha) / (counts.sum() + alpha * counts.size)
    w = 1.0 / p
    w /= w.min()
    return np.minimum(w, cap).astype(np.float32)


class WindowDataset(Dataset):
    def __init__(self, config: WindowDatasetConfig):
        self.config = config
        self.mode_pattern = build_mode_pattern(config.masking)
        if config.games is not None:
            self.games = config.games
            return
        self.games: list[list[np.ndarray]] = []
        for path in sorted(Path(config.data_dir).rglob("*.parquet")):
            self.games.extend(_load_game_segments(path))
        return

    def __len__(self) -> int:
        return len(self.games)

    def __getitem__(self, index: int) -> dict[str, np.ndarray]:
        config = self.config
        seed = None if config.seed is None else config.seed + index
        rng = np.random.default_rng(seed)
        mode = config.force_mode
        if mode is None and config.balanced_modes:
            mode = self.mode_pattern[index % len(self.mode_pattern)]
        return _build_window(self.games[index], config, rng, mode)


def _sample_segments(
    segments: list[np.ndarray],
    rng: np.random.Generator,
    config: WindowDatasetConfig,
) -> list[np.ndarray]:
    n = len(segments)
    u = rng.random()
    if u < config.p_single or n < config.few_min:
        start = int(rng.integers(n))
        return [segments[start]]
    if u < config.p_single + config.p_few:
        k = int(rng.integers(config.few_min, min(config.few_max, n) + 1))
        return _sample_few(segments, rng, config, k)
    return _sample_full(segments, rng, config)


def _sample_few(
    segments: list[np.ndarray],
    rng: np.random.Generator,
    config: WindowDatasetConfig,
    k: int,
) -> list[np.ndarray]:
    """Sample up to k consecutive segments without exceeding the token budget."""
    budget = config.max_seq_len - 1
    start = int(rng.integers(len(segments) - k + 1))
    chosen: list[np.ndarray] = []
    used = 0
    for i in range(start, min(start + k, len(segments))):
        cost = int(len(segments[i])) + 1
        if chosen and used + cost > budget:
            break
        chosen.append(segments[i])
        used += cost
    if not chosen:
        chosen = [segments[start][: budget - 1]]
    return chosen


def _sample_full(
    segments: list[np.ndarray],
    rng: np.random.Generator,
    config: WindowDatasetConfig,
) -> list[np.ndarray]:
    budget = config.max_seq_len - 1
    lengths = np.fromiter((len(s) for s in segments), dtype=np.int64)
    total = int(lengths.sum() + len(segments))
    if total <= budget and len(segments) <= config.max_segments:
        return list(segments)
    start = int(rng.integers(len(segments)))
    chosen: list[np.ndarray] = []
    used = 0
    for i in range(start, len(segments)):
        cost = int(lengths[i]) + 1
        if used + cost > budget or len(chosen) >= config.max_segments:
            break
        chosen.append(segments[i])
        used += cost
    if not chosen:
        chosen = [segments[start][: budget - 1]]
    return chosen


def _sample_long(
    segments: list[np.ndarray],
    rng: np.random.Generator,
    config: WindowDatasetConfig,
    min_segments: int,
) -> list[np.ndarray]:
    """Sample a full-budget window guaranteed to hold at least min_segments."""
    budget = config.max_seq_len - 1
    lengths = np.fromiter((len(s) for s in segments), dtype=np.int64)
    total = int(lengths.sum() + len(segments))
    if total <= budget and len(segments) <= config.max_segments:
        return list(segments)
    max_start = len(segments) - min_segments
    if max_start < 0:
        return list(segments)
    start = int(rng.integers(max_start + 1))
    chosen: list[np.ndarray] = []
    used = 0
    for i in range(start, len(segments)):
        cost = int(lengths[i]) + 1
        if used + cost > budget or len(chosen) >= config.max_segments:
            break
        chosen.append(segments[i])
        used += cost
    return chosen


def _build_window(
    segments: list[np.ndarray],
    config: WindowDatasetConfig,
    rng: np.random.Generator,
    mode: str | None,
) -> dict[str, np.ndarray]:
    """Sample, pack, and mask one window from a game's segments."""
    if mode in ("board", "span"):
        if len(segments) >= config.masking.min_context_segments:
            segments = _sample_long(
                segments, rng, config, config.masking.min_context_segments
            )
        else:
            mode = "random"
            segments = _sample_segments(segments, rng, config)
    else:
        segments = _sample_segments(segments, rng, config)
    tokens = _pack(segments)
    if config.apply_masking:
        masked, targets, weights = mask_stream(tokens, config.masking, rng, mode)
        if config.unigram_weights is not None:
            hit = targets != IGNORE_INDEX
            weights[hit] *= config.unigram_weights[targets[hit]]
    else:
        masked = tokens
        targets = np.full(tokens.shape, IGNORE_INDEX, dtype=np.int16)
        weights = np.zeros(tokens.shape, dtype=np.float32)
    return {"tokens": masked, "targets": targets, "weights": weights}


def _load_game_segments(
    path: Path, max_workers: int | None = None
) -> list[list[np.ndarray]]:
    """Load games from a parquet file, or recursively from a directory.

    Directories (e.g. a hub snapshot's ``{split}/{archive_id}/`` layout) are
    searched recursively for ``*.parquet`` shards; other files such as
    ``.meta.json`` sidecars are ignored. Shards are loaded in parallel with a
    thread pool (parquet decoding releases the GIL); ``max_workers=None``
    uses all CPUs, and results keep shard order so loading stays
    deterministic.
    """
    if path.is_dir():
        shards = sorted(path.rglob("*.parquet"))
        if not shards:
            return []
        workers = min(len(shards), max_workers or os.cpu_count() or 1)
        games: list[list[np.ndarray]] = []
        with ThreadPoolExecutor(max_workers=workers) as pool:
            for shard_games in pool.map(_load_game_segments, shards):
                games.extend(shard_games)
        return games
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


def _shard_counts(shard: Path) -> np.ndarray:
    """Per-shard token histogram with SEP and leading-CLS framing removed."""
    counts = np.zeros(VOCAB_SIZE, dtype=np.int64)
    table = pq.read_table(shard, columns=["packed"])
    column = table.column("packed").combine_chunks()
    if len(column) == 0:
        return counts
    offsets = np.asarray(column.offsets)
    values = column.flatten().to_numpy(zero_copy_only=False).astype(np.int16)
    counts += np.bincount(values, minlength=VOCAB_SIZE)
    counts[SEP] -= int((values == SEP).sum())
    counts[CLS] -= int((values[offsets[:-1]] == CLS).sum())
    return counts


def streaming_unigram_counts(
    data_dir: str | Path,
    log=None,
    log_every: int = 25,
    max_workers: int | None = None,
) -> np.ndarray:
    """Compute token counts over a split without materializing games.

    Shards whose ``.meta.json`` sidecar carries a ``unigram_counts``
    histogram (recorded at ETL time) are summed directly without touching
    the parquet files. Remaining shards are counted with a parallel parquet
    pass (mirroring :func:`unigram_counts` semantics: SEP framing tokens and
    each stream's leading CLS excluded). ``log`` is an optional callable
    receiving human-readable progress messages.
    """
    log = log or (lambda message: None)
    shards = sorted(Path(data_dir).rglob("*.parquet"))
    log(f"unigram pass: {len(shards)} shards in {data_dir}")
    counts = np.zeros(VOCAB_SIZE, dtype=np.int64)
    missing: list[Path] = []
    for shard in shards:
        sidecar = _sidecar_counts(shard)
        if sidecar is None:
            missing.append(shard)
        else:
            counts += sidecar
    if not missing:
        log(f"unigram pass: done, {int(counts.sum())} tokens counted (from sidecars)")
        return counts
    log(f"unigram pass: {len(missing)} shards without sidecar histograms")
    workers = max_workers or min(len(missing), os.cpu_count() or 1)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for index, shard_count in enumerate(pool.map(_shard_counts, missing), start=1):
            counts += shard_count
            if index == 1 or index % log_every == 0 or index == len(missing):
                log(f"unigram pass: shard {index}/{len(missing)} done")
    log(f"unigram pass: done, {int(counts.sum())} tokens counted")
    return counts


def _sidecar_counts(shard: Path) -> np.ndarray | None:
    """Per-shard histogram from the ``.meta.json`` sidecar, if it has one."""
    sidecar = shard.with_suffix(".meta.json")
    if not sidecar.exists():
        return None
    try:
        payload = json.loads(sidecar.read_text(encoding="utf-8"))
        counts = payload["stats"]["unigram_counts"]
    except (OSError, ValueError, KeyError, TypeError):
        return None
    if not isinstance(counts, list) or len(counts) != VOCAB_SIZE:
        return None
    return np.asarray(counts, dtype=np.int64)


class StreamingWindowDataset(IterableDataset):
    """Stream training windows shard by shard without preloading the corpus.

    Each epoch shuffles the shard order, loads one shard at a time, and yields
    windows sampled with the same logic as :class:`WindowDataset`. An optional
    shuffle buffer decorrelates consecutive games across shard boundaries.
    With ``num_workers > 0`` each DataLoader worker claims a disjoint round-
    robin subset of the shards, so workers never yield the same game.
    """

    def __init__(self, config: WindowDatasetConfig):
        self.config = config
        self.mode_pattern = build_mode_pattern(config.masking)
        if config.games is not None:
            raise ValueError("streaming dataset requires data_dir, not preloaded games")
        if not config.data_dir:
            raise ValueError("streaming dataset requires data_dir")
        self.shards = sorted(Path(config.data_dir).rglob("*.parquet"))
        self.shard_sizes = [
            pq.ParquetFile(shard).metadata.num_rows for shard in self.shards
        ]
        return

    def __len__(self) -> int:
        return int(sum(self.shard_sizes))

    def _worker_shard_indices(self) -> list[int]:
        """Return this worker's round-robin subset of shard indices."""
        info = torch.utils.data.get_worker_info()
        all_indices = list(range(len(self.shards)))
        if info is None:
            return all_indices
        return all_indices[info.id :: info.num_workers]

    def __iter__(self):
        info = torch.utils.data.get_worker_info()
        worker_id = 0 if info is None else info.id
        skip = worker_skip_windows(
            self.config.resume_step,
            self.config.batch_size,
            self.config.num_workers,
            worker_id,
        )
        stream = self._iter_windows(worker_id, self._worker_shard_indices())
        for _ in range(skip):
            next(stream, None)
        yield from stream
        return

    def _iter_windows(self, worker_id: int, worker_indices: list[int]):
        rng = np.random.default_rng(
            None if self.config.seed is None else self.config.seed + worker_id
        )
        mode_counter = 0
        buffer: list[list[np.ndarray]] = []
        epoch = 0
        while True:
            order = rng.permutation(worker_indices)
            shard_iter = (
                self._prefetched_shards(order)
                if self.config.prefetch_shards > 0
                else (
                    (index, _load_game_segments(self.shards[index])) for index in order
                )
            )
            for _, games in shard_iter:
                for segments in games:
                    if self.config.shuffle_buffer > 1:
                        if len(buffer) >= self.config.shuffle_buffer:
                            game = buffer.pop(int(rng.integers(len(buffer))))
                            yield self._window(game, rng, mode_counter)
                            mode_counter += 1
                        buffer.append(segments)
                    else:
                        yield self._window(segments, rng, mode_counter)
                        mode_counter += 1
            while buffer:
                game = buffer.pop(int(rng.integers(len(buffer))))
                yield self._window(game, rng, mode_counter)
                mode_counter += 1
            epoch += 1
        return

    def _prefetched_shards(self, order: np.ndarray):
        """Yield ``(shard_index, games)`` while preloading ahead in a thread pool.

        A bounded queue holds up to ``prefetch_shards`` decoded shards per
        worker, decoupling parquet IO + decode from window building so the
        consumer never waits on the next shard load. The bound caps the
        cache's memory footprint.
        """
        queue: Queue = Queue(maxsize=self.config.prefetch_shards)
        end = object()

        def producer() -> None:
            try:
                for index in order:
                    queue.put((index, _load_game_segments(self.shards[index])))
                queue.put(end)
            except Exception as error:  # surface producer failures to consumer
                queue.put(error)
            return

        thread = Thread(target=producer, daemon=True)
        thread.start()
        while True:
            item = queue.get()
            if item is end:
                break
            if isinstance(item, Exception):
                raise item
            yield item
        thread.join()
        return

    def _window(
        self,
        segments: list[np.ndarray],
        rng: np.random.Generator,
        mode_counter: int,
    ) -> dict[str, np.ndarray]:
        config = self.config
        mode = config.force_mode
        if mode is None and config.balanced_modes:
            mode = self.mode_pattern[mode_counter % len(self.mode_pattern)]
        return _build_window(segments, config, rng, mode)


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
    weights = np.zeros((n, max_len), dtype=np.float32)
    for i, sample in enumerate(batch):
        tokens[i, : len(sample["tokens"])] = sample["tokens"]
        targets[i, : len(sample["targets"])] = sample["targets"]
        sample_weights = sample.get("weights")
        if sample_weights is not None:
            weights[i, : len(sample_weights)] = sample_weights
    segment_ids = np.cumsum(tokens == SEP, axis=1, dtype=np.int64)
    return {
        "tokens": torch.from_numpy(tokens.astype(np.int64)),
        "targets": torch.from_numpy(targets.astype(np.int64)),
        "weights": torch.from_numpy(weights),
        "segment_ids": torch.from_numpy(segment_ids),
        "padding_mask": torch.from_numpy(tokens != PAD),
    }
