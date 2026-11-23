"""Vector-store interfaces shared by LanceDB and the test oracle."""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from typing import Protocol, Sequence

import numpy as np

from src.search.schema import PositionRow, result_to_code, split_to_code


@dataclass(frozen=True)
class PositionFilter:
    """Metadata filters applied before the nearest-neighbour search.

    ``min_elo``/``max_elo`` constrain both players, matching the common
    "both players are at least this strong" interpretation.
    """

    min_elo: int | None = None
    max_elo: int | None = None
    min_year: int | None = None
    max_year: int | None = None
    min_date: dt.date | None = None
    max_date: dt.date | None = None
    result: str | None = None
    split: str | None = None
    min_ply: int | None = None
    max_ply: int | None = None
    min_pieces: int | None = None
    max_pieces: int | None = None
    exclude_game_id: int | None = None

    def to_sql(self) -> str | None:
        """Render the filter as a LanceDB SQL predicate."""
        clauses: list[str] = []
        if self.min_elo is not None:
            clauses.append(
                f"white_elo >= {int(self.min_elo)} AND black_elo >= {int(self.min_elo)}"
            )
        if self.max_elo is not None:
            clauses.append(
                f"white_elo <= {int(self.max_elo)} AND black_elo <= {int(self.max_elo)}"
            )
        if self.min_year is not None:
            clauses.append(f"year >= {int(self.min_year)}")
        if self.max_year is not None:
            clauses.append(f"year <= {int(self.max_year)}")
        if self.min_date is not None:
            clauses.append(f"utc_date >= date '{self.min_date.isoformat()}'")
        if self.max_date is not None:
            clauses.append(f"utc_date <= date '{self.max_date.isoformat()}'")
        if self.result is not None:
            clauses.append(f"result = {result_to_code(self.result)}")
        if self.split is not None:
            clauses.append(f"split = {split_to_code(self.split)}")
        if self.min_ply is not None:
            clauses.append(f"ply >= {int(self.min_ply)}")
        if self.max_ply is not None:
            clauses.append(f"ply <= {int(self.max_ply)}")
        if self.min_pieces is not None:
            clauses.append(f"n_pieces >= {int(self.min_pieces)}")
        if self.max_pieces is not None:
            clauses.append(f"n_pieces <= {int(self.max_pieces)}")
        if self.exclude_game_id is not None:
            clauses.append(f"game_id != {int(self.exclude_game_id)}")
        return " AND ".join(clauses) or None

    def matches(self, row: PositionRow) -> bool:
        """Apply the same semantics in Python for the brute-force oracle."""
        if self.min_elo is not None and (
            row.white_elo < self.min_elo or row.black_elo < self.min_elo
        ):
            return False
        if self.max_elo is not None and (
            row.white_elo > self.max_elo or row.black_elo > self.max_elo
        ):
            return False
        if self.min_year is not None and row.year < self.min_year:
            return False
        if self.max_year is not None and row.year > self.max_year:
            return False
        if self.min_date is not None and row.utc_date < self.min_date:
            return False
        if self.max_date is not None and row.utc_date > self.max_date:
            return False
        if self.result is not None and result_to_code(row.result) != result_to_code(
            self.result
        ):
            return False
        if self.split is not None and split_to_code(row.split) != split_to_code(
            self.split
        ):
            return False
        if self.min_ply is not None and row.ply < self.min_ply:
            return False
        if self.max_ply is not None and row.ply > self.max_ply:
            return False
        if self.min_pieces is not None and row.n_pieces < self.min_pieces:
            return False
        if self.max_pieces is not None and row.n_pieces > self.max_pieces:
            return False
        if self.exclude_game_id is not None and row.game_id == self.exclude_game_id:
            return False
        return True


@dataclass(frozen=True)
class PositionHit:
    """One nearest-neighbour result."""

    game_id: int
    ply: int
    packed_position: bytes
    distance: float
    white_elo: int
    black_elo: int
    utc_date: dt.date
    year: int
    result: str
    split: str
    n_pieces: int

    @property
    def score(self) -> float:
        """Cosine similarity derived from the cosine distance."""
        return 1.0 - self.distance


class PositionStore(Protocol):
    """Minimal interface implemented by LanceDB and the memory oracle."""

    dim: int

    def add(
        self, rows: Sequence[PositionRow], *, partition: str | None = None
    ) -> int: ...

    def count(self) -> int: ...

    def search(
        self,
        vector: np.ndarray,
        *,
        limit: int,
        filter: PositionFilter | None = None,
        where: str | None = None,
        nprobes: int | None = None,
        refine_factor: int | None = None,
        exact: bool = False,
        partitions: Sequence[str] | None = None,
    ) -> list[PositionHit]: ...

    def create_index(
        self,
        *,
        index_type: str = "ivf_pq",
        num_partitions: int | None = None,
        num_sub_vectors: int | None = None,
    ) -> None: ...

    def optimize(self) -> None: ...

    def close(self) -> None: ...
