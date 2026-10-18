"""Upload training checkpoints plus metadata to a HuggingFace bucket.

Checkpoints are mutable artifacts (overwritten ``last.pt``, step-versioned
bests), so they go to an HF bucket — S3-like object storage via
``batch_bucket_files`` — rather than a git-backed repo. A ``.meta.json``
sidecar is written next to every checkpoint so each one is self-describing.
"""

from __future__ import annotations

import json
from pathlib import Path

import pyarrow.parquet as pq
from huggingface_hub import batch_bucket_files


def collect_data_provenance(data_dir: Path) -> dict:
    """Describe the training data under ``data_dir``.

    Works uniformly for hub-downloaded snapshots and local directories:
    recursively finds ``*.parquet`` shards and reports per-shard path (relative
    to ``data_dir``), size in bytes, and game (row) count from parquet
    metadata — no data is read.
    """
    data_dir = Path(data_dir)
    shards = []
    for shard in sorted(data_dir.rglob("*.parquet")):
        rows = pq.ParquetFile(shard).metadata.num_rows
        shards.append(
            {
                "path": shard.relative_to(data_dir).as_posix(),
                "size": shard.stat().st_size,
                "games": rows,
            }
        )
    return {
        "shards": shards,
        "total_bytes": sum(shard["size"] for shard in shards),
        "total_games": sum(shard["games"] for shard in shards),
    }


def build_metadata(
    training_config: dict,
    model_config: dict,
    n_params: int,
    step: int,
    best_val_loss: float,
    data_repo: str | None,
    data_limit_gb: float | None,
    train_provenance: dict,
    val_provenance: dict,
    git_commit: str | None,
) -> dict:
    """Assemble the metadata payload accompanying a checkpoint."""
    return {
        "training": dict(training_config),
        "model": {**model_config, "n_params": n_params},
        "step": step,
        "best_val_loss": best_val_loss,
        "data": {
            "repo": data_repo,
            "limit_gb": data_limit_gb,
            "train": train_provenance,
            "val": val_provenance,
        },
        "git_commit": git_commit,
    }


def upload_checkpoint(
    checkpoint_path: Path,
    bucket_id: str,
    remote_name: str,
    metadata: dict,
    prefix: str = "checkpoints",
) -> None:
    """Upload a checkpoint file and its metadata sidecar to a bucket.

    Both files are committed together: ``{prefix}/{remote_name}.pt`` and
    ``{prefix}/{remote_name}.meta.json``.
    """
    encoded = json.dumps(metadata, indent=2, sort_keys=True).encode("utf-8")
    batch_bucket_files(
        bucket_id,
        add=[
            (Path(checkpoint_path), f"{prefix}/{remote_name}.pt"),
            (encoded, f"{prefix}/{remote_name}.meta.json"),
        ],
    )
    return
