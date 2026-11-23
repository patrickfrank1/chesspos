"""Brute-force position store used as a test oracle and small-data backend."""

from __future__ import annotations

from typing import Sequence

import numpy as np

from src.search.schema import PositionRow
from src.search.store import PositionFilter, PositionHit


class MemoryPositionStore:
    """In-memory cosine search with the same semantics as LanceDB.

    This is intentionally simple: it keeps every row and scans them. It is the
    reference implementation for tests and for measuring ANN recall.
    """

    def __init__(self, dim: int) -> None:
        if dim <= 0:
            raise ValueError("dim must be positive")
        self.dim = dim
        self.rows: list[PositionRow] = []
        self._vectors = np.zeros((0, dim), dtype=np.float32)
        return

    def add(
        self,
        rows: Sequence[PositionRow],
        *,
        partition: str | None = None,
        games: int = 0,
    ) -> int:
        del partition, games
        if not rows:
            return 0
        vectors = np.empty((len(rows), self.dim), dtype=np.float32)
        for index, row in enumerate(rows):
            vector = np.asarray(row.vector, dtype=np.float32).ravel()
            if vector.size != self.dim:
                raise ValueError(
                    f"vector has {vector.size} dimensions, expected {self.dim}"
                )
            norm = float(np.linalg.norm(vector))
            if norm <= 0:
                raise ValueError("position vectors must be non-zero")
            vectors[index] = vector / norm
        self.rows.extend(rows)
        self._vectors = np.vstack([self._vectors, vectors])
        return len(rows)

    def count(self) -> int:
        return len(self.rows)

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
    ) -> list[PositionHit]:
        del nprobes, refine_factor, exact, partitions
        if where is not None:
            raise ValueError("MemoryPositionStore only accepts PositionFilter")
        if limit <= 0:
            return []
        if not self.rows:
            return []
        query = np.asarray(vector, dtype=np.float32).ravel()
        if query.size != self.dim:
            raise ValueError(f"query has {query.size} dimensions, expected {self.dim}")
        norm = float(np.linalg.norm(query))
        if norm <= 0:
            raise ValueError("query vector must be non-zero")
        query = query / norm

        indices = np.fromiter(
            (
                index
                for index, row in enumerate(self.rows)
                if filter is None or filter.matches(row)
            ),
            dtype=np.int64,
        )
        if indices.size == 0:
            return []
        distances = 1.0 - self._vectors[indices] @ query
        order = np.argsort(distances, kind="stable")[: min(limit, indices.size)]
        hits = []
        for slot in order:
            row = self.rows[int(indices[int(slot)])]
            hits.append(
                PositionHit(
                    game_id=row.game_id,
                    ply=row.ply,
                    packed_position=row.packed_position,
                    distance=float(distances[int(slot)]),
                    white_elo=row.white_elo,
                    black_elo=row.black_elo,
                    utc_date=row.utc_date,
                    year=row.year,
                    result=row.result,
                    split=row.split,
                    n_pieces=row.n_pieces,
                )
            )
        return hits

    def create_index(
        self,
        *,
        index_type: str = "ivf_pq",
        num_partitions: int | None = None,
        num_sub_vectors: int | None = None,
    ) -> None:
        del index_type, num_partitions, num_sub_vectors
        return

    def optimize(self) -> None:
        return

    def close(self) -> None:
        return

    def byte_size(self) -> int:
        return sum(
            row.vector.nbytes + len(row.packed_position) + 64 for row in self.rows
        )
