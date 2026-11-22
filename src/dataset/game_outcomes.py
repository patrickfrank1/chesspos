from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import pyarrow.parquet as pq
import torch
from torch.utils.data import Dataset

from src.dataset.puzzle_variants import pack_segments, split_stream
from src.dataset.token_stream import (
    CASTLE_BASE,
    EMPTY,
    EP_BASE,
    N_SLOTS,
    PIECE_BASE,
    PAD,
    SEGMENT_LEN,
    TURN_BLACK,
    TURN_WHITE,
)

WHITE_WIN = "1-0"
BLACK_WIN = "0-1"
DRAW = "1/2-1/2"

_WHITE_SCORES = {WHITE_WIN: 1.0, BLACK_WIN: -1.0, DRAW: 0.0}
_PIECE_VALUES = np.array([1, 3, 3, 5, 9, 0], dtype=np.int64)
_MIRROR_SQUARES = np.arange(N_SLOTS) ^ 56
_EP_TOKENS = (EP_BASE, EP_BASE + 16)


def score_from_result(result: str) -> float:
    """Map a lichess result string to a White-perspective score in [-1, 1]."""
    try:
        return _WHITE_SCORES[result]
    except KeyError as error:
        raise ValueError(f"unknown game result {result!r}") from error


@dataclass(frozen=True)
class GameOutcome:
    """One game's packed positions plus the realised result."""

    game_id: str
    segments: list[np.ndarray]
    result: str
    white_elo: int = 0
    black_elo: int = 0
    source: str = ""

    def __post_init__(self) -> None:
        if not self.segments:
            raise ValueError("a game must contain at least one position")
        score_from_result(self.result)
        return

    @property
    def score(self) -> float:
        return score_from_result(self.result)

    def __len__(self) -> int:
        return len(self.segments)


def decode_packed(packed: Sequence[int] | np.ndarray) -> list[np.ndarray]:
    """Decode a CLS-prefixed packed stream into 67-token position segments."""
    stream = np.asarray(packed, dtype=np.int16)
    segments = split_stream(stream)
    if not segments:
        raise ValueError("packed stream contains no position segments")
    return segments


def flip_segment(segment: np.ndarray) -> np.ndarray:
    """Mirror a segment's colours and ranks (``Board.mirror()`` in token form)."""
    segment = np.asarray(segment, dtype=np.int16)
    if segment.shape != (SEGMENT_LEN,):
        raise ValueError(
            f"a segment must have {SEGMENT_LEN} tokens, got {segment.shape}"
        )

    squares = segment[:N_SLOTS].copy()
    pieces = (squares >= PIECE_BASE) & (squares < PIECE_BASE + 12)
    squares[pieces] += np.where(squares[pieces] < PIECE_BASE + 6, 6, -6)
    if not np.isin(squares[~pieces], (EMPTY,)).all():
        raise ValueError("segment contains a non-piece, non-empty square token")

    flipped = np.empty_like(segment)
    flipped[_MIRROR_SQUARES] = squares

    turn = int(segment[N_SLOTS])
    if turn not in (TURN_WHITE, TURN_BLACK):
        raise ValueError(f"expected a turn token, got {turn}")
    flipped[N_SLOTS] = TURN_BLACK if turn == TURN_WHITE else TURN_WHITE

    castle = int(segment[N_SLOTS + 1]) - CASTLE_BASE
    if not 0 <= castle < 16:
        raise ValueError(f"expected a castling token, got {segment[N_SLOTS + 1]}")
    flipped[N_SLOTS + 1] = CASTLE_BASE + (((castle & 3) << 2) | ((castle >> 2) & 3))

    ep = int(segment[N_SLOTS + 2]) - EP_BASE
    if ep == 0:
        flipped[N_SLOTS + 2] = EP_BASE
    elif 1 <= ep <= 8:
        flipped[N_SLOTS + 2] = EP_BASE + ep + 8
    elif 9 <= ep <= 16:
        flipped[N_SLOTS + 2] = EP_BASE + ep - 8
    else:
        raise ValueError(f"expected an en-passant token, got {segment[N_SLOTS + 2]}")
    return flipped


def material_balance(segment: np.ndarray) -> int:
    """Piece value from White's perspective (pawn 1, N/B 3, R 5, Q 9)."""
    squares = np.asarray(segment, dtype=np.int16)[:N_SLOTS]
    index = squares.astype(np.int64) - PIECE_BASE
    pieces = (index >= 0) & (index < 12)
    values = _PIECE_VALUES[index % 6]
    values = np.where(index < 6, values, -values)
    return int(values[pieces].sum())


def state_features(segment: np.ndarray) -> np.ndarray:
    """Explicit state bits for recipes that pool content only."""
    turn = 1.0 if int(segment[N_SLOTS]) == TURN_WHITE else 0.0
    castle = (int(segment[N_SLOTS + 1]) - CASTLE_BASE) / 15.0
    ep = 0.0 if int(segment[N_SLOTS + 2]) == EP_BASE else 1.0
    return np.array([turn, castle, ep], dtype=np.float32)


def split_for_game(game_id: str, val_ratio: float = 0.1) -> str:
    """Deterministic game-level split, independent of iteration order."""
    if not 0 < val_ratio < 1:
        raise ValueError(f"val_ratio must be in (0, 1), got {val_ratio}")
    digest = hashlib.sha1(game_id.encode("utf-8")).digest()
    uniform = int.from_bytes(digest[:8], "big") / 2**64
    return "val" if uniform < val_ratio else "train"


@dataclass(frozen=True)
class ScorePlan:
    """One row per sampled (game, ply) position."""

    game_index: np.ndarray
    ply: np.ndarray

    def __post_init__(self) -> None:
        if self.game_index.shape != self.ply.shape:
            raise ValueError("game_index and ply must have the same shape")
        if self.ply.size and int(self.ply.min()) < 1:
            raise ValueError("plies are 1-based and must be positive")
        return

    def __len__(self) -> int:
        return int(self.game_index.size)


def build_score_plan(
    records: Sequence[GameOutcome],
    positions_per_game: int | None = 16,
    max_ply: int = 150,
    seed: int = 17,
) -> ScorePlan:
    """Sample plies per game; each sample is one (game, ply) position."""
    if positions_per_game is not None and positions_per_game < 1:
        raise ValueError("positions_per_game must be positive or None")
    if max_ply < 1:
        raise ValueError("max_ply must be positive")

    rng = np.random.default_rng(seed)
    game_index: list[np.ndarray] = []
    ply: list[np.ndarray] = []
    for index, record in enumerate(records):
        n_positions = min(len(record.segments), max_ply)
        if n_positions <= 0:
            continue
        candidates = np.arange(1, n_positions + 1, dtype=np.int16)
        if positions_per_game is None or positions_per_game >= n_positions:
            chosen = candidates
        else:
            chosen = np.sort(
                rng.choice(candidates, size=positions_per_game, replace=False)
            )
        game_index.append(np.full(chosen.size, index, dtype=np.int32))
        ply.append(chosen.astype(np.int16))
    if not game_index:
        empty = np.empty(0, dtype=np.int32)
        return ScorePlan(game_index=empty, ply=empty.astype(np.int16))
    return ScorePlan(
        game_index=np.concatenate(game_index),
        ply=np.concatenate(ply),
    )


class ScoreDataset(Dataset):
    """Single-position (or short window) samples with a White-POV score target."""

    def __init__(
        self,
        records: Sequence[GameOutcome],
        plan: ScorePlan,
        window: int = 1,
        include_state_features: bool = False,
    ):
        if window < 1:
            raise ValueError(f"window must be at least 1, got {window}")
        self.records = list(records)
        self.plan = plan
        self.window = window
        self.include_state_features = include_state_features
        return

    def __len__(self) -> int:
        return len(self.plan)

    def __getitem__(self, index: int) -> dict:
        record = self.records[int(self.plan.game_index[index])]
        ply = int(self.plan.ply[index])
        if not 1 <= ply <= len(record.segments):
            raise IndexError(
                f"ply {ply} is outside game {record.game_id!r} "
                f"with {len(record.segments)} positions"
            )
        start = max(0, ply - self.window)
        segments = record.segments[start:ply]
        tokens, segment_ids = pack_segments(segments)
        item = {
            "tokens": tokens,
            "segment_ids": segment_ids,
            "padding_mask": np.ones(tokens.shape[0], dtype=bool),
            "target": np.float32(record.score),
            "ply": ply,
            "game_id": record.game_id,
            "source": record.source,
            "material": material_balance(segments[-1]),
        }
        if self.include_state_features:
            item["features"] = state_features(segments[-1])
        return item


def collate_score_samples(batch: list[dict]) -> dict:
    lengths = [len(sample["tokens"]) for sample in batch]
    max_len = max(lengths)
    tokens = np.full((len(batch), max_len), PAD, dtype=np.int16)
    segment_ids = np.zeros((len(batch), max_len), dtype=np.int64)
    padding_mask = np.zeros((len(batch), max_len), dtype=bool)
    for i, sample in enumerate(batch):
        length = len(sample["tokens"])
        tokens[i, :length] = sample["tokens"]
        segment_ids[i, :length] = sample["segment_ids"]
        padding_mask[i, :length] = True
    collated = {
        "tokens": torch.from_numpy(tokens.astype(np.int64)),
        "segment_ids": torch.from_numpy(segment_ids),
        "padding_mask": torch.from_numpy(padding_mask),
        "targets": torch.from_numpy(
            np.array([sample["target"] for sample in batch], dtype=np.float32)
        ),
        "plies": torch.tensor([sample["ply"] for sample in batch], dtype=torch.long),
        "game_ids": [sample["game_id"] for sample in batch],
        "sources": [sample.get("source", "") for sample in batch],
    }
    if "material" in batch[0]:
        collated["materials"] = torch.tensor(
            [sample["material"] for sample in batch], dtype=torch.long
        )
    if "features" in batch[0]:
        collated["features"] = torch.from_numpy(
            np.stack([sample["features"] for sample in batch]).astype(np.float32)
        )
    return collated


def read_game_shard(path: Path, limit: int | None = None) -> list[GameOutcome]:
    source = path.parent.name
    table = pq.read_table(
        path,
        columns=["packed", "game_id", "white_elo", "black_elo", "result"],
    )
    rows = table.to_pylist()
    records: list[GameOutcome] = []
    for row in rows:
        if limit is not None and len(records) >= limit:
            break
        result = row.get("result") or ""
        if result not in _WHITE_SCORES:
            continue
        try:
            segments = decode_packed(row["packed"])
        except ValueError:
            continue
        records.append(
            GameOutcome(
                game_id=str(row.get("game_id") or ""),
                segments=segments,
                result=result,
                white_elo=int(row.get("white_elo") or 0),
                black_elo=int(row.get("black_elo") or 0),
                source=source,
            )
        )
    return records


def load_game_records(
    data_dir: str | Path, limit: int | None = None
) -> list[GameOutcome]:
    """Load game shards from a parquet file or a directory of shards."""
    path = Path(data_dir)
    if limit is not None and limit < 1:
        raise ValueError("limit must be positive or None")
    paths = sorted(path.rglob("*.parquet")) if path.is_dir() else [path]
    if not paths:
        raise ValueError(f"no parquet shards found in {data_dir}")
    records: list[GameOutcome] = []
    for shard in paths:
        remaining = None if limit is None else limit - len(records)
        if remaining is not None and remaining <= 0:
            break
        records.extend(read_game_shard(shard, limit=remaining))
    return records
