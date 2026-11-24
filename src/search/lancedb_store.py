"""LanceDB-backed position store."""

from __future__ import annotations

import datetime as dt
import json
from pathlib import Path
from typing import Sequence

import lancedb
import numpy as np

from src.search.schema import (
    IndexManifest,
    PositionRow,
    code_to_result,
    code_to_split,
    position_schema,
    rows_to_table,
)
from src.search.store import PositionFilter, PositionHit

DEFAULT_TABLE = "positions"


class LancePositionStore:
    """A single LanceDB table holding one row per game-position."""

    def __init__(
        self,
        path: str | Path,
        *,
        dim: int,
        create: bool = False,
        overwrite: bool = False,
        table_name: str = DEFAULT_TABLE,
    ) -> None:
        if dim <= 0:
            raise ValueError("dim must be positive")
        self.path = Path(path)
        self.dim = dim
        self.table_name = table_name
        self._db = lancedb.connect(str(self.path))
        if create:
            schema = position_schema(dim)
            empty = rows_to_table([], dim)
            mode = "overwrite" if overwrite else "create"
            self._table = self._db.create_table(
                table_name,
                data=empty,
                schema=schema,
                mode=mode,
            )
        else:
            self._table = self._db.open_table(table_name)
        return

    @property
    def table(self):
        return self._table

    def add(
        self,
        rows: Sequence[PositionRow],
        *,
        partition: str | None = None,
        games: int = 0,
    ) -> int:
        del games
        if partition is not None:
            raise ValueError("LancePositionStore does not support partitions")
        if not rows:
            return 0
        self._table.add(rows_to_table(rows, self.dim))
        return len(rows)

    def count(self) -> int:
        return int(self._table.count_rows())

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
        del partitions
        if limit <= 0:
            return []
        query = np.asarray(vector, dtype=np.float16).ravel()
        if query.size != self.dim:
            raise ValueError(f"query has {query.size} dimensions, expected {self.dim}")
        if where is None and filter is not None:
            where = filter.to_sql()

        builder = self._table.search(query)
        if where:
            builder = builder.where(where, prefilter=True)
        builder = builder.select(
            [
                "game_id",
                "ply",
                "packed_position",
                "white_elo",
                "black_elo",
                "utc_date",
                "year",
                "result",
                "split",
                "n_pieces",
                "_distance",
            ]
        ).limit(limit)
        if nprobes is not None:
            builder = builder.nprobes(nprobes)
        if refine_factor is not None:
            builder = builder.refine_factor(refine_factor)
        if exact:
            builder = builder.bypass_vector_index()

        hits = []
        for row in builder.to_list():
            hits.append(
                PositionHit(
                    game_id=int(row["game_id"]),
                    ply=int(row["ply"]),
                    packed_position=bytes(row["packed_position"]),
                    distance=float(row.get("_distance", 0.0)),
                    white_elo=int(row["white_elo"]),
                    black_elo=int(row["black_elo"]),
                    utc_date=_as_date(row["utc_date"]),
                    year=int(row["year"]),
                    result=code_to_result(int(row["result"])),
                    split=code_to_split(int(row["split"])),
                    n_pieces=int(row["n_pieces"]),
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
        from lancedb.index import IvfFlat, IvfHnswSq, IvfPq

        kind = index_type.lower()
        if kind == "ivf_pq":
            config = IvfPq(
                distance_type="cosine",
                num_partitions=num_partitions,
                num_sub_vectors=num_sub_vectors,
            )
        elif kind == "ivf_hnsw_sq":
            config = IvfHnswSq(distance_type="cosine", num_partitions=num_partitions)
        elif kind == "ivf_flat":
            config = IvfFlat(distance_type="cosine", num_partitions=num_partitions)
        else:
            raise ValueError(f"Unsupported index type {index_type!r}")
        self._table.create_index("vector", config=config, replace=True)
        return

    def create_scalar_indexes(self) -> None:
        from lancedb.index import BTree, Bitmap

        self._table.create_index("utc_date", config=BTree(), replace=True)
        self._table.create_index("game_id", config=BTree(), replace=True)
        self._table.create_index("white_elo", config=BTree(), replace=True)
        self._table.create_index("black_elo", config=BTree(), replace=True)
        self._table.create_index("ply", config=BTree(), replace=True)
        self._table.create_index("year", config=Bitmap(), replace=True)
        self._table.create_index("result", config=Bitmap(), replace=True)
        self._table.create_index("split", config=Bitmap(), replace=True)
        self._table.create_index("n_pieces", config=Bitmap(), replace=True)
        return

    def optimize(self) -> None:
        self._table.optimize()
        return

    def close(self) -> None:
        return

    def byte_size(self) -> int:
        return sum(
            path.stat().st_size for path in self.path.rglob("*") if path.is_file()
        )

    @property
    def manifest_path(self) -> Path:
        return self.path / f"{self.table_name}.manifest.json"

    def write_manifest(self, manifest: IndexManifest) -> Path:
        path = self.manifest_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(manifest.to_dict(), indent=2, sort_keys=True))
        return path

    def read_manifest(self) -> IndexManifest | None:
        path = self.manifest_path
        if not path.exists():
            return None
        return IndexManifest.from_dict(json.loads(path.read_text()))


def _as_date(value) -> dt.date:
    if isinstance(value, dt.datetime):
        return value.date()
    if isinstance(value, dt.date):
        return value
    if isinstance(value, int):
        return dt.date(1970, 1, 1) + dt.timedelta(days=value)
    if isinstance(value, str) and value:
        return dt.date.fromisoformat(value)
    return dt.date(1970, 1, 1)
