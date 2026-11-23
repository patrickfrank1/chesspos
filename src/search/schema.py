"""Arrow schema and value objects for the position vector index."""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pyarrow as pa

from src.search.codec import POSITION_BYTES

RESULT_UNKNOWN = 0
RESULT_WHITE = 1
RESULT_BLACK = 2
RESULT_DRAW = 3

RESULT_TO_CODE = {
    "": RESULT_UNKNOWN,
    "*": RESULT_UNKNOWN,
    "1-0": RESULT_WHITE,
    "0-1": RESULT_BLACK,
    "1/2-1/2": RESULT_DRAW,
}
CODE_TO_RESULT = {
    RESULT_UNKNOWN: "*",
    RESULT_WHITE: "1-0",
    RESULT_BLACK: "0-1",
    RESULT_DRAW: "1/2-1/2",
}

SPLIT_TO_CODE = {"train": 0, "test": 1}
CODE_TO_SPLIT = {0: "train", 1: "test"}

UNKNOWN_DATE = dt.date(1970, 1, 1)


def result_to_code(value: str | None) -> int:
    """Map a PGN result to its compact code."""
    return RESULT_TO_CODE.get((value or "").strip(), RESULT_UNKNOWN)


def code_to_result(value: int) -> str:
    """Map a compact result code back to a PGN result."""
    try:
        return CODE_TO_RESULT[int(value)]
    except KeyError as error:
        raise ValueError(f"Unknown result code {value!r}") from error


def split_to_code(value: str | None) -> int:
    """Map a split name to its compact code."""
    return SPLIT_TO_CODE.get((value or "train").strip().lower(), 0)


def code_to_split(value: int) -> str:
    """Map a compact split code back to a split name."""
    try:
        return CODE_TO_SPLIT[int(value)]
    except KeyError as error:
        raise ValueError(f"Unknown split code {value!r}") from error


def game_id_to_int(value: str) -> int:
    """Convert a 16-hex-character game id to ``uint64``."""
    text = str(value).strip().lower()
    if len(text) != 16:
        raise ValueError("game_id must contain exactly 16 hexadecimal characters")
    try:
        return int(text, 16)
    except ValueError as error:
        raise ValueError(
            "game_id must contain exactly 16 hexadecimal characters"
        ) from error


def game_id_to_hex(value: int) -> str:
    """Format a ``uint64`` game id as 16 lowercase hexadecimal characters."""
    number = int(value)
    if number < 0 or number >= 2**64:
        raise ValueError("game_id must fit in an unsigned 64-bit integer")
    return f"{number:016x}"


def _uint(value: int, bits: int, name: str) -> int:
    number = int(value)
    limit = 2**bits
    if number < 0 or number >= limit:
        raise ValueError(f"{name} must fit in uint{bits}")
    return number


@dataclass(frozen=True)
class PositionRow:
    """One game-position row ready for the vector store."""

    vector: np.ndarray
    packed_position: bytes
    game_id: int
    ply: int
    white_elo: int
    black_elo: int
    utc_date: dt.date
    year: int
    result: str
    split: str
    n_pieces: int

    def __post_init__(self) -> None:
        if len(bytes(self.packed_position)) != POSITION_BYTES:
            raise ValueError(f"packed_position must be {POSITION_BYTES} bytes")
        _uint(self.game_id, 64, "game_id")
        _uint(self.ply, 16, "ply")
        _uint(self.white_elo, 16, "white_elo")
        _uint(self.black_elo, 16, "black_elo")
        _uint(self.year, 16, "year")
        _uint(self.n_pieces, 8, "n_pieces")
        if self.result not in RESULT_TO_CODE:
            raise ValueError(f"Unknown result {self.result!r}")
        if self.split not in SPLIT_TO_CODE:
            raise ValueError(f"Unknown split {self.split!r}")
        return


def position_schema(dim: int) -> pa.Schema:
    """Return the compact Arrow schema for a position index."""
    if dim <= 0:
        raise ValueError("dim must be positive")
    return pa.schema(
        [
            pa.field("vector", pa.list_(pa.float16(), dim), nullable=False),
            pa.field("packed_position", pa.binary(POSITION_BYTES), nullable=False),
            pa.field("game_id", pa.uint64(), nullable=False),
            pa.field("ply", pa.uint16(), nullable=False),
            pa.field("white_elo", pa.uint16(), nullable=False),
            pa.field("black_elo", pa.uint16(), nullable=False),
            pa.field("utc_date", pa.date32(), nullable=False),
            pa.field("year", pa.uint16(), nullable=False),
            pa.field("result", pa.uint8(), nullable=False),
            pa.field("split", pa.uint8(), nullable=False),
            pa.field("n_pieces", pa.uint8(), nullable=False),
        ]
    )


def rows_to_table(rows: Sequence[PositionRow], dim: int) -> pa.Table:
    """Convert position rows into an Arrow table with the index schema."""
    vectors = []
    for row in rows:
        vector = np.asarray(row.vector, dtype=np.float16).ravel()
        if vector.size != dim:
            raise ValueError(f"vector has {vector.size} dimensions, expected {dim}")
        vectors.append(vector)
    return pa.table(
        {
            "vector": pa.array(vectors, type=pa.list_(pa.float16(), dim)),
            "packed_position": pa.array(
                [bytes(row.packed_position) for row in rows],
                type=pa.binary(POSITION_BYTES),
            ),
            "game_id": pa.array([row.game_id for row in rows], type=pa.uint64()),
            "ply": pa.array([row.ply for row in rows], type=pa.uint16()),
            "white_elo": pa.array([row.white_elo for row in rows], type=pa.uint16()),
            "black_elo": pa.array([row.black_elo for row in rows], type=pa.uint16()),
            "utc_date": pa.array([row.utc_date for row in rows], type=pa.date32()),
            "year": pa.array([row.year for row in rows], type=pa.uint16()),
            "result": pa.array(
                [result_to_code(row.result) for row in rows], type=pa.uint8()
            ),
            "split": pa.array(
                [split_to_code(row.split) for row in rows], type=pa.uint8()
            ),
            "n_pieces": pa.array([row.n_pieces for row in rows], type=pa.uint8()),
        },
        schema=position_schema(dim),
    )


@dataclass(frozen=True)
class IndexManifest:
    """Provenance and coverage metadata stored next to an index."""

    index_version: int
    checkpoint: str
    checkpoint_sha256: str
    feature: str
    dim: int
    dtype: str
    schema_version: int
    rows: int
    games: int
    date_coverage: float
    lancedb_version: str
    created_at: str

    def to_dict(self) -> dict:
        return {
            "index_version": self.index_version,
            "checkpoint": self.checkpoint,
            "checkpoint_sha256": self.checkpoint_sha256,
            "feature": self.feature,
            "dim": self.dim,
            "dtype": self.dtype,
            "schema_version": self.schema_version,
            "rows": self.rows,
            "games": self.games,
            "date_coverage": self.date_coverage,
            "lancedb_version": self.lancedb_version,
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, values: dict) -> IndexManifest:
        return cls(**values)
