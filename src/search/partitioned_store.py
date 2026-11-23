"""A partitioned LanceDB store with fan-out search."""

from __future__ import annotations

import json
import re
import shutil
from pathlib import Path
from typing import Sequence

import numpy as np

from src.search.lancedb_store import LancePositionStore
from src.search.schema import IndexManifest, PositionRow
from src.search.store import PositionFilter, PositionHit

ROSTER_NAME = "partitions.json"
_NAME = re.compile(r"[^A-Za-z0-9._=-]+")


def partition_name(value: str) -> str:
    """Sanitise a partition label for use as a directory name."""
    name = _NAME.sub("_", str(value)).strip("._")
    if not name:
        raise ValueError("partition name cannot be empty")
    return name


class PartitionedPositionStore:
    """One LanceDB table per partition, queried with a client-side fan-out."""

    def __init__(
        self,
        root: str | Path,
        *,
        dim: int,
        create: bool = False,
        overwrite: bool = False,
        table_name: str = "positions",
    ) -> None:
        if dim <= 0:
            raise ValueError("dim must be positive")
        self.root = Path(root)
        self.dim = dim
        self.table_name = table_name
        self.root.mkdir(parents=True, exist_ok=True)
        self._stores: dict[str, LancePositionStore] = {}
        self._partitions: dict[str, dict] = {}
        self.roster_path = self.root / ROSTER_NAME
        if overwrite:
            self._clear()
        if self.roster_path.exists():
            self._load_roster()
        elif create:
            self._write_roster()
        return

    def _clear(self) -> None:
        for child in self.root.iterdir():
            if child.is_dir():
                shutil.rmtree(child)
            elif child.name in {ROSTER_NAME, f"{self.table_name}.manifest.json"}:
                child.unlink()
        return

    def _load_roster(self) -> None:
        payload = json.loads(self.roster_path.read_text())
        if int(payload.get("dim", self.dim)) != self.dim:
            raise ValueError(
                "partition roster dimension does not match the embedder "
                f"({payload.get('dim')} != {self.dim})"
            )
        self._partitions = dict(payload.get("partitions", {}))
        return

    def _write_roster(self) -> None:
        self.roster_path.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "dim": self.dim,
                    "table_name": self.table_name,
                    "partitions": self._partitions,
                },
                indent=2,
                sort_keys=True,
            )
        )
        return

    def _store(self, name: str) -> LancePositionStore:
        if name in self._stores:
            return self._stores[name]
        directory = self.root / name
        table_exists = (directory / f"{self.table_name}.lance").exists()
        store = LancePositionStore(
            directory,
            dim=self.dim,
            create=not table_exists,
            overwrite=False,
            table_name=self.table_name,
        )
        self._stores[name] = store
        self._partitions.setdefault(name, {"rows": 0})
        return store

    @property
    def partitions(self) -> dict[str, dict]:
        return self._partitions

    def add(
        self,
        rows: Sequence[PositionRow],
        *,
        partition: str | None = None,
        games: int = 0,
    ) -> int:
        if partition is None:
            raise ValueError("a partitioned store requires a partition name")
        name = partition_name(partition)
        added = self._store(name).add(rows)
        info = self._partitions[name]
        info["rows"] = int(info.get("rows", 0)) + added
        info["games"] = int(info.get("games", 0)) + int(games)
        self._write_roster()
        return added

    def count(self) -> int:
        return sum(int(info.get("rows", 0)) for info in self._partitions.values())

    def games_count(self) -> int:
        return sum(int(info.get("games", 0)) for info in self._partitions.values())

    def _selected(self, filter: PositionFilter | None) -> list[str]:
        names = sorted(self._partitions)
        if not names:
            return []
        if all(name.startswith("year=") for name in names):
            selected = names
            if filter is not None and filter.min_year is not None:
                selected = [
                    name
                    for name in selected
                    if int(name.split("=", 1)[1]) >= filter.min_year
                ]
            if filter is not None and filter.max_year is not None:
                selected = [
                    name
                    for name in selected
                    if int(name.split("=", 1)[1]) <= filter.max_year
                ]
            return selected
        return names

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
        if limit <= 0:
            return []
        selected = (
            [partition_name(name) for name in partitions]
            if partitions is not None
            else self._selected(filter)
        )
        hits: list[PositionHit] = []
        for name in selected:
            if name not in self._partitions:
                continue
            hits.extend(
                self._store(name).search(
                    vector,
                    limit=limit,
                    filter=filter,
                    where=where,
                    nprobes=nprobes,
                    refine_factor=refine_factor,
                    exact=exact,
                )
            )
        hits.sort(key=lambda hit: (hit.distance, hit.game_id, hit.ply))
        return hits[:limit]

    def create_index(
        self,
        *,
        index_type: str = "ivf_pq",
        num_partitions: int | None = None,
        num_sub_vectors: int | None = None,
    ) -> None:
        for name in sorted(self._partitions):
            self._store(name).create_index(
                index_type=index_type,
                num_partitions=num_partitions,
                num_sub_vectors=num_sub_vectors,
            )
        return

    def create_scalar_indexes(self) -> None:
        for name in sorted(self._partitions):
            self._store(name).create_scalar_indexes()
        return

    def optimize(self) -> None:
        for name in sorted(self._partitions):
            self._store(name).optimize()
        return

    def close(self) -> None:
        for store in self._stores.values():
            store.close()
        return

    def byte_size(self) -> int:
        return sum(
            path.stat().st_size for path in self.root.rglob("*") if path.is_file()
        )

    @property
    def manifest_path(self) -> Path:
        return self.root / f"{self.table_name}.manifest.json"

    def write_manifest(self, manifest: IndexManifest) -> Path:
        path = self.manifest_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(manifest.to_dict(), indent=2, sort_keys=True))
        return path

    def read_manifest(self) -> IndexManifest | None:
        if not self.manifest_path.exists():
            return None
        return IndexManifest.from_dict(json.loads(self.manifest_path.read_text()))
