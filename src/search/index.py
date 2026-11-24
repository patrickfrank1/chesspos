"""Build a position index from packed game shards."""

from __future__ import annotations

import datetime as dt
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Protocol, Sequence

import numpy as np
import pyarrow.parquet as pq

from src.dataset.puzzle_variants import split_stream
from src.dataset.token_stream import PIECE_BASE, TURN_WHITE
from src.search.codec import encode_position
from src.search.schema import (
    RESULT_TO_CODE,
    UNKNOWN_DATE,
    PositionRow,
    game_id_to_int,
)
from src.search.store import PositionStore

_YEAR = re.compile(r"(?<!\d)((?:19|20)\d{2})-\d{2}(?!\d)")
_REQUIRED_COLUMNS = {
    "packed",
    "game_id",
    "white_elo",
    "black_elo",
    "result",
}


class PositionEmbedder(Protocol):
    """Embeds raw position segments into normalized vectors."""

    dim: int

    def embed(self, segments: Sequence[np.ndarray]) -> np.ndarray: ...


@dataclass(frozen=True)
class GameRow:
    """One game as stored in the dataset parquet shards."""

    packed: np.ndarray
    game_id: str
    white_elo: int
    black_elo: int
    result: str
    split: str
    utc_date: dt.date | None = None
    year: int = 0


@dataclass(frozen=True)
class IndexStats:
    """Counts produced by a build."""

    games: int
    positions: int
    skipped: int
    files: int
    dated_positions: int
    partitions: dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "games": self.games,
            "positions": self.positions,
            "skipped": self.skipped,
            "files": self.files,
            "dated_positions": self.dated_positions,
            "partitions": self.partitions,
        }


def _clamp(value: int, bits: int) -> int:
    return max(0, min(int(value), 2**bits - 1))


def _normalize(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    if np.any(norms <= 0):
        raise ValueError("position vectors must be non-zero")
    return vectors / norms


def piece_count(tokens: np.ndarray) -> int:
    """Count non-empty squares in a position segment."""
    values = np.asarray(tokens)
    return int(np.sum((values >= PIECE_BASE) & (values < TURN_WHITE)))


def index_game_row(row: GameRow, embedder: PositionEmbedder) -> list[PositionRow]:
    """Expand one packed game into one index row per position."""
    segments = split_stream(np.asarray(row.packed, dtype=np.int16))
    if not segments:
        return []
    vocab_size = getattr(embedder, "vocab_size", None)
    if vocab_size is not None:
        for segment in segments:
            largest = int(np.max(segment)) if len(segment) else 0
            if largest >= vocab_size:
                raise ValueError(
                    f"position token {largest} exceeds the embedder vocabulary "
                    f"of {vocab_size}; this shard was probably encoded with a "
                    "different tokenizer"
                )
    vectors = np.asarray(embedder.embed(segments), dtype=np.float32)
    if vectors.shape != (len(segments), embedder.dim):
        raise ValueError(
            "embedder returned shape "
            f"{vectors.shape}, expected {(len(segments), embedder.dim)}"
        )
    vectors = _normalize(vectors)

    date = row.utc_date or UNKNOWN_DATE
    if date != UNKNOWN_DATE:
        year = date.year
    elif row.year:
        year = int(row.year)
    else:
        year = 0
    game_id = game_id_to_int(row.game_id)
    white_elo = _clamp(row.white_elo, 16)
    black_elo = _clamp(row.black_elo, 16)
    result = row.result if row.result in RESULT_TO_CODE else ""

    positions = []
    for ply, (segment, vector) in enumerate(zip(segments, vectors), start=1):
        tokens = np.asarray(segment, dtype=np.int16)
        positions.append(
            PositionRow(
                vector=vector,
                packed_position=encode_position(tokens),
                game_id=game_id,
                ply=_clamp(ply, 16),
                white_elo=white_elo,
                black_elo=black_elo,
                utc_date=date,
                year=_clamp(year, 16),
                result=result,
                split=row.split,
                n_pieces=piece_count(tokens),
            )
        )
    return positions


def path_year(path: str | Path) -> int:
    """Best-effort source year from an archive path.

    Archive directories are named ``<name>_YYYY-MM``; the pattern requires the
    month suffix so hexadecimal snapshot hashes cannot be mistaken for years.
    The last match wins when a cache path contains several dated components.
    """
    matches = _YEAR.findall(str(path))
    return int(matches[-1]) if matches else 0


def path_split(path: str | Path) -> str | None:
    """Return ``train``/``test`` when the path contains that split directory."""
    parts = {part.lower() for part in Path(path).parts}
    if "train" in parts:
        return "train"
    if "test" in parts:
        return "test"
    return None


def path_partition(path: str | Path, partition_by: str, year: int) -> str:
    """Derive a partition label from a shard path."""
    source_path = Path(path)
    if partition_by == "year":
        return f"year={int(year)}"
    if partition_by == "source":
        parent = source_path.parent
        if parent.name in {"train", "test"}:
            parent = parent.parent
        source = parent.name or source_path.stem
        return f"source={source}"
    raise ValueError(f"Unsupported partition_by {partition_by!r}")


def _parse_date(value) -> dt.date | None:
    if value is None:
        return None
    if isinstance(value, dt.datetime):
        return value.date()
    if isinstance(value, dt.date):
        return value
    text = str(value).strip()
    if not text:
        return None
    try:
        return dt.date.fromisoformat(text)
    except ValueError:
        return None


def _packed_column(table) -> tuple[np.ndarray, np.ndarray]:
    column = table.column("packed").combine_chunks()
    offsets = np.asarray(column.offsets, dtype=np.int64)
    values = column.flatten().to_numpy(zero_copy_only=False).astype(np.int16)
    return offsets, values


def build_position_index(
    *,
    store: PositionStore,
    embedder: PositionEmbedder,
    data_dir: str | Path,
    batch_games: int = 32,
    limit_games: int | None = None,
    source_year: int | None = None,
    default_split: str = "train",
    partition_by: str | None = None,
    progress: Callable[[str], None] | None = None,
) -> IndexStats:
    """Index every packed game under ``data_dir`` into ``store``.

    Existing shards do not carry a per-game ``utc_date`` or ``split``. The split
    is taken from the path (or ``default_split``), the year from the path (or
    ``source_year``), and missing dates use :data:`UNKNOWN_DATE`.

    With ``partition_by="year"`` or ``"source"``, rows are batched per shard's
    partition and written through a :class:`PartitionedPositionStore`.
    """
    if batch_games <= 0:
        raise ValueError("batch_games must be positive")
    if partition_by not in (None, "year", "source"):
        raise ValueError("partition_by must be one of None, 'year', 'source'")
    paths = sorted(Path(data_dir).rglob("*.parquet"))
    if not paths:
        raise ValueError(f"No parquet shards found under {data_dir}")

    games = positions = skipped = dated_positions = 0
    files = 0
    stop = False
    batches: dict[str | None, list[PositionRow]] = {}
    batch_counts: dict[str | None, int] = {}
    partition_positions: dict[str, int] = {}

    def flush(key: str | None) -> None:
        rows = batches.get(key)
        if not rows:
            return
        if key is None:
            store.add(rows, games=batch_counts.get(key, 0))
        else:
            store.add(
                rows,
                partition=key,
                games=batch_counts.get(key, 0),
            )
        batches[key] = []
        batch_counts[key] = 0
        return

    for path in paths:
        if stop:
            break
        files += 1
        table = pq.read_table(path)
        missing = _REQUIRED_COLUMNS - set(table.column_names)
        if missing:
            raise ValueError(f"{path} is missing required columns: {sorted(missing)}")
        offsets, values = _packed_column(table)
        ids = table.column("game_id").to_pylist()
        white = table.column("white_elo").to_pylist()
        black = table.column("black_elo").to_pylist()
        results = table.column("result").to_pylist()
        dates = (
            table.column("utc_date").to_pylist()
            if "utc_date" in table.column_names
            else [None] * len(ids)
        )
        split_values = (
            table.column("split").to_pylist()
            if "split" in table.column_names
            else [None] * len(ids)
        )
        file_split = path_split(path) or default_split
        file_year = source_year if source_year is not None else path_year(path)
        key = (
            None
            if partition_by is None
            else path_partition(path, partition_by, file_year)
        )

        for row_index, game_id in enumerate(ids):
            if limit_games is not None and games >= limit_games:
                stop = True
                break
            packed = values[offsets[row_index] : offsets[row_index + 1]]
            split_value = split_values[row_index]
            split = split_value if split_value in {"train", "test"} else file_split
            row = GameRow(
                packed=packed,
                game_id=str(game_id),
                white_elo=int(white[row_index] or 0),
                black_elo=int(black[row_index] or 0),
                result=str(results[row_index] or ""),
                split=split,
                utc_date=_parse_date(dates[row_index]),
                year=file_year,
            )
            new_rows = index_game_row(row, embedder)
            if not new_rows:
                skipped += 1
                continue
            batches.setdefault(key, []).extend(new_rows)
            games += 1
            positions += len(new_rows)
            dated_positions += sum(
                1 for position in new_rows if position.utc_date != UNKNOWN_DATE
            )
            if key is not None:
                partition_positions[key] = partition_positions.get(key, 0) + len(
                    new_rows
                )
            batch_counts[key] = batch_counts.get(key, 0) + 1
            if batch_counts[key] >= batch_games:
                flush(key)
            if progress is not None and games % 1000 == 0:
                progress(f"indexed {games} games / {positions} positions")
    for key in list(batches):
        flush(key)

    return IndexStats(
        games=games,
        positions=positions,
        skipped=skipped,
        files=files,
        dated_positions=dated_positions,
        partitions=partition_positions,
    )
