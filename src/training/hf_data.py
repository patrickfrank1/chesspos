"""Download a training split from a HuggingFace dataset repo.

See ``docs/hf_dataset_access.md``: the training side uses full, resumable
snapshot downloads (Xet-backed) rather than streaming. Datasets can outgrow a
pod's disk, so the train split can be bounded to a byte budget by selecting a
deterministic, evenly spread subset of Parquet shards.
"""

from __future__ import annotations

from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download, snapshot_download


def download_split(
    repo_id: str,
    split: str,
    cache_dir: str | None = None,
    max_bytes: int | None = None,
    log=None,
) -> Path:
    """Download ``split`` of a dataset repo and return its local directory.

    With ``max_bytes`` set, only a deterministic, evenly spread subset of the
    split's Parquet shards (sized to fit the budget) is downloaded.
    ``log`` is an optional callable receiving human-readable progress
    messages; when omitted, progress is silent.
    """
    log = log or (lambda message: None)
    if max_bytes is None:
        allow_patterns: list[str] = [f"{split}/**"]
        log(f"downloading {repo_id} split={split} (unbounded) ...")
    else:
        shards = _list_split_shards(repo_id, split)
        if not shards:
            raise ValueError(f"no parquet shards found under {split}/ in {repo_id}")
        total = sum(size for _, size in shards)
        log(
            f"downloading {repo_id} split={split}: {len(shards)} parquet "
            f"shards, {total / 1024**3:.2f} GiB total; budget "
            f"{max_bytes / 1024**3:.2f} GiB"
        )
        allow_patterns = select_shard_subset(shards, max_bytes)
        log(f"selected {len(allow_patterns)} of {len(shards)} shards")
    snapshot = snapshot_download(
        repo_id=repo_id,
        repo_type="dataset",
        allow_patterns=allow_patterns,
        cache_dir=cache_dir,
    )
    snapshot_root = Path(snapshot)
    if max_bytes is not None:
        _prune_unselected_shards(snapshot_root, split, allow_patterns)
    log(f"downloaded {repo_id} split={split} to {snapshot_root / split}")
    return snapshot_root / split


def _prune_unselected_shards(
    snapshot_root: Path, split: str, selected: list[str]
) -> None:
    """Drop shards left by earlier downloads that are not in ``selected``.

    ``snapshot_download`` keeps previously cached files, so a bounded run can
    otherwise train on more shards than its budget selected. Only parquet
    shards and their ``.meta.json`` sidecars inside the split are touched.
    """
    split_dir = snapshot_root / split
    if not split_dir.is_dir():
        return
    allowed = set(selected)
    for path in split_dir.rglob("*"):
        if not path.is_file():
            continue
        relative = path.relative_to(snapshot_root).as_posix()
        if relative.endswith(".parquet"):
            keep = relative in allowed
        elif relative.endswith(".meta.json"):
            keep = relative.replace(".meta.json", ".parquet") in allowed
        else:
            continue
        if not keep:
            path.unlink(missing_ok=True)
    return


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


def _list_split_shards(
    repo_id: str, split: str, revision: str | None = None
) -> list[tuple[str, int]]:
    """List the split's Parquet shards with their sizes."""
    api = HfApi()
    kwargs = {"repo_type": "dataset", "path_in_repo": split, "recursive": True}
    if revision is not None:
        kwargs["revision"] = revision
    shards = []
    for entry in api.list_repo_tree(repo_id, **kwargs):
        if entry.path.endswith(".parquet"):
            shards.append((entry.path, entry.size))
    return shards


def list_split_shards(
    repo_id: str, split: str, revision: str | None = None
) -> list[tuple[str, int]]:
    """Return the split's Parquet shards as sorted ``(path, size)`` pairs."""
    return sorted(_list_split_shards(repo_id, split, revision=revision))


def fetch_shard(
    repo_id: str,
    filename: str,
    cache_dir: str | None = None,
    revision: str | None = None,
) -> Path:
    """Download a single shard into ``cache_dir`` and return its local path.

    ``local_dir`` mode stores the file directly under ``cache_dir`` (rather
    than as a symlink into the global blob cache), so :func:`delete_shard`
    actually frees the disk space.
    """
    kwargs = {"repo_type": "dataset", "local_dir": cache_dir}
    if revision is not None:
        kwargs["revision"] = revision
    return Path(hf_hub_download(repo_id, filename, **kwargs))


def delete_shard(path: str | Path) -> None:
    """Remove a previously fetched shard, ignoring it if it is already gone."""
    try:
        Path(path).unlink()
    except FileNotFoundError:
        pass
    return
