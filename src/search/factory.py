"""Open either a single-table or partitioned position index."""

from __future__ import annotations

from pathlib import Path

from src.search.lancedb_store import LancePositionStore
from src.search.partitioned_store import ROSTER_NAME, PartitionedPositionStore


def open_position_store(
    path: str | Path,
    *,
    dim: int,
    table_name: str = "positions",
) -> LancePositionStore | PartitionedPositionStore:
    """Open the store layout recorded under ``path``."""
    root = Path(path)
    if (root / ROSTER_NAME).exists():
        return PartitionedPositionStore(
            root, dim=dim, create=False, table_name=table_name
        )
    return LancePositionStore(root, dim=dim, create=False, table_name=table_name)
