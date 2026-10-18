"""Download a training split from a HuggingFace dataset repo.

See ``docs/hf_dataset_access.md``: the training side uses full, resumable
snapshot downloads (Xet-backed) rather than streaming. Datasets can outgrow a
pod's disk, so the train split can be bounded to a byte budget by selecting a
deterministic, evenly spread subset of Parquet shards.
"""

from __future__ import annotations

from pathlib import Path

from huggingface_hub import HfApi, snapshot_download


def download_split(
    repo_id: str,
    split: str,
    cache_dir: str | None = None,
    max_bytes: int | None = None,
) -> Path:
    """Download ``split`` of a dataset repo and return its local directory.

    With ``max_bytes`` set, only a deterministic, evenly spread subset of the
    split's Parquet shards (sized to fit the budget) is downloaded.
    """
    if max_bytes is None:
        allow_patterns: list[str] = [f"{split}/**"]
    else:
        shards = _list_split_shards(repo_id, split)
        if not shards:
            raise ValueError(f"no parquet shards found under {split}/ in {repo_id}")
        allow_patterns = select_shard_subset(shards, max_bytes)
    snapshot = snapshot_download(
        repo_id=repo_id,
        repo_type="dataset",
        allow_patterns=allow_patterns,
        cache_dir=cache_dir,
    )
    return Path(snapshot) / split


def select_shard_subset(shards: list[tuple[str, int]], max_bytes: int) -> list[str]:
    """Select shard paths totalling at most ``max_bytes``.

    Shards are sorted by path; if the total exceeds the budget, shards are
    taken at evenly spaced indices across the whole repo (a representative
    sample, not a temporal prefix), then trailing selections are dropped
    while the cumulative size still exceeds the budget.
    """
    shards = sorted(shards, key=lambda shard: shard[0])
    total_size = sum(size for _, size in shards)
    if total_size <= max_bytes:
        return [path for path, _ in shards]
    n = max(1, len(shards) * max_bytes // total_size)
    indices = [int(i * len(shards) / n) for i in range(n)]
    sizes = {path: size for path, size in shards}
    selected = [shards[index][0] for index in indices]
    cumulative = sum(sizes[path] for path in selected)
    while cumulative > max_bytes and len(selected) > 1:
        cumulative -= sizes[selected.pop()]
    return selected


def _list_split_shards(repo_id: str, split: str) -> list[tuple[str, int]]:
    """List the split's Parquet shards with their sizes, sorted by path."""
    api = HfApi()
    shards = []
    for entry in api.list_repo_tree(
        repo_id, repo_type="dataset", path_in_repo=split, recursive=True
    ):
        if entry.path.endswith(".parquet"):
            shards.append((entry.path, entry.size))
    return shards
