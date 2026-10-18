"""Per-batch metadata sidecars.

Every parquet shard is accompanied by a ``.meta.json`` file describing
where its games came from, what it contains, and with which pipeline
configuration it was produced. The sidecars are the source of truth for
idempotent resume: a batch whose recorded ``file_sha256`` matches the
locally produced one is considered already uploaded.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timezone

import pyarrow as pa

from src.dataset.config import GameSubsampling, TimeControlFilter

SCHEMA_VERSION = 3
PIPELINE_VERSION = "3"


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


def config_fingerprint(
    subsampling: GameSubsampling,
    time_control_filter: TimeControlFilter,
) -> str:
    """Deterministic sha1 of the game-selection configuration."""
    payload = json.dumps(
        {
            "subsampling": subsampling.to_dict(),
            "time_control_filter": time_control_filter.to_dict(),
        },
        sort_keys=True,
    )
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()


@dataclass
class BatchMeta:
    """Metadata for one parquet batch file."""

    schema_version: int
    provider: str
    archive_id: str
    url: str | None
    split: str
    batch_number: int
    chunk_index: int
    seq: int
    file_name: str
    num_games: int
    num_positions: int
    duplicates_dropped: int
    file_sha256: str
    date_min: str | None
    date_max: str | None
    config: dict
    config_fingerprint: str
    pipeline_version: str
    created_at: str

    @property
    def batch_stem(self) -> str:
        return f"batch_{self.batch_number:04d}"

    def repo_parquet_path(self) -> str:
        return f"{self.split}/{self.archive_id}/{self.batch_stem}.parquet"

    def repo_meta_path(self) -> str:
        return f"{self.split}/{self.archive_id}/{self.batch_stem}.meta.json"

    def to_json(self) -> str:
        return json.dumps(
            {
                "schema_version": self.schema_version,
                "source": {
                    "provider": self.provider,
                    "archive_id": self.archive_id,
                    "url": self.url,
                },
                "batch": {
                    "split": self.split,
                    "batch_number": self.batch_number,
                    "chunk_index": self.chunk_index,
                    "seq": self.seq,
                    "file_name": self.file_name,
                },
                "stats": {
                    "num_games": self.num_games,
                    "num_positions": self.num_positions,
                    "duplicates_dropped": self.duplicates_dropped,
                    "file_sha256": self.file_sha256,
                },
                "range": {
                    "date_min": self.date_min,
                    "date_max": self.date_max,
                },
                "config": self.config,
                "provenance": {
                    "config_fingerprint": self.config_fingerprint,
                    "pipeline_version": self.pipeline_version,
                    "created_at": self.created_at,
                },
            }
        )

    @classmethod
    def from_json(cls, json_str: str) -> "BatchMeta":
        data = json.loads(json_str)
        source = data["source"]
        batch = data["batch"]
        stats = data["stats"]
        range_ = data["range"]
        config = data.get("config", {})
        provenance = data["provenance"]
        return cls(
            schema_version=data["schema_version"],
            provider=source["provider"],
            archive_id=source["archive_id"],
            url=source["url"],
            split=batch["split"],
            batch_number=batch["batch_number"],
            chunk_index=batch["chunk_index"],
            seq=batch["seq"],
            file_name=batch["file_name"],
            num_games=stats["num_games"],
            num_positions=stats["num_positions"],
            duplicates_dropped=stats["duplicates_dropped"],
            file_sha256=stats["file_sha256"],
            date_min=range_["date_min"],
            date_max=range_["date_max"],
            config=config,
            config_fingerprint=provenance["config_fingerprint"],
            pipeline_version=provenance["pipeline_version"],
            created_at=provenance["created_at"],
        )


def build_batch_meta(
    table: pa.Table,
    file_bytes: bytes,
    split: str,
    batch_number: int,
    chunk_index: int,
    seq: int,
    provider: str,
    archive_id: str,
    url: str | None,
    duplicates_dropped: int,
    config_fingerprint: str,
    utc_dates: list[str],
    subsampling: GameSubsampling,
    time_control_filter: TimeControlFilter,
    train_ratio: float,
    batch_size: int,
) -> BatchMeta:
    """Build a BatchMeta from a finalized batch table and its parquet bytes.

    The full generation-relevant configuration (subsampling tiers, time
    control filter, split ratio and batch size) is recorded alongside the
    fingerprint so the values themselves are inspectable, not just their
    hash.
    """
    dates = sorted(d for d in utc_dates if d)
    return BatchMeta(
        schema_version=SCHEMA_VERSION,
        provider=provider,
        archive_id=archive_id,
        url=url,
        split=split,
        batch_number=batch_number,
        chunk_index=chunk_index,
        seq=seq,
        file_name=f"batch_{batch_number:04d}.parquet",
        num_games=table.num_rows,
        num_positions=sum(table.column("n_positions").to_pylist()),
        duplicates_dropped=duplicates_dropped,
        file_sha256=hashlib.sha256(file_bytes).hexdigest(),
        date_min=dates[0] if dates else None,
        date_max=dates[-1] if dates else None,
        config={
            "batch_size": batch_size,
            "train_ratio": train_ratio,
            "subsampling": subsampling.to_dict(),
            "time_control_filter": time_control_filter.to_dict(),
        },
        config_fingerprint=config_fingerprint,
        pipeline_version=PIPELINE_VERSION,
        created_at=_utcnow(),
    )
