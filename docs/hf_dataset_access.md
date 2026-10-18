# HF Dataset Access for Remote / Ephemeral Training

Decision note on how the training side should consume the HuggingFace Hub
dataset repo on remote workers (vast.ai ephemeral GPU pods). Complements
[`dataset_pipeline.md`](dataset_pipeline.md), which documents the write side.

## Context

- The Hub repo stores Parquet shards: `{split}/batch_*.parquet` (legacy ETL)
  or `{split}/{archive_id}/batch_*.parquet` (archive pipeline), one row per
  game with a packed `int16` token stream.
- The training read side (`src/training/window_dataset.py`) is a map-style
  `WindowDataset` that globs `*.parquet` from a local directory and holds all
  games in memory. `src/run/train_masked_transformer.py` points it at
  `--train-dir` / `--val-dir`.
- Ephemeral pods: local disk (NVMe) is fast but disposable; training state is
  preserved via checkpoints, not the data cache.

## Options

### Option A — Stream directly (`load_dataset(..., streaming=True)`)

Documented pattern ([HF datasets docs](https://github.com/huggingface/datasets/blob/main/docs/source/stream.mdx)):

```python
from datasets import load_dataset
from torch.utils.data import DataLoader

ds = load_dataset("user/chesspos", split="train", streaming=True)  # IterableDataset
ds = ds.shuffle(seed=17, buffer_size=10_000)   # shard-order + buffer shuffle (approximate)
ds = ds.with_format("torch")
loader = DataLoader(ds, batch_size=32, num_workers=4, collate_fn=collate_windows)

for epoch in range(n_epochs):
    ds.set_epoch(epoch)          # required, or every epoch sees identical order
    for batch in loader: ...
```

Mechanics:

- `num_shards` equals the Parquet shard count; DataLoader `num_workers`
  automatically split shards across workers.
- `.shuffle(buffer_size)` shuffles shard order and uses a shuffle buffer —
  approximate, not global. Each row is an independent game and window
  sampling is per-game, so this is statistically acceptable for MLM training.
- `set_epoch(epoch)` is mandatory per epoch.

Caveats for this project:

1. Requires a new `WindowIterableDataset`-style path. `WindowDataset`
   (src/training/window_dataset.py:30) is index-based (`__len__`,
   `__getitem__`, seed = `seed + index`); streaming has no indices, so
   per-row seeding would come from a counter instead.
2. Throughput: streaming does HTTP range reads per shard. Xet-backed repos
   are fast, but a fast GPU can outpace the stream. With one packed `int16`
   stream per row it is probably fine, but it is a risk.
3. No full global shuffle across epochs.

### Option B — Download once, train locally (chosen)

```python
from huggingface_hub import snapshot_download
path = snapshot_download(repo_id="user/chesspos", repo_type="dataset",
                         allow_patterns=["train/*.parquet"])
```
or on the pod CLI: `hf download <repo> --repo-type dataset`.

- **Xet backend** makes this fast (parallel chunk download) and the download
  is **resumable**: on preemption, a re-run continues from the local cache.
  Point the cache at the pod's NVMe disk.
- The data is compact: `int16` packed streams, one row per game — even
  millions of games land in single-digit GBs, which is cheaper than
  re-streaming HTTP reads across multiple epochs.
- `WindowDataset` works **unchanged** (point `data_dir` at the snapshot's
  `train/` directory), preserving the existing map-style shuffle, per-index
  seeding, and the tested code path.
- Allow patterns must cover the nested archive layout
  (`train/**` matches both `train/batch_*.parquet` and
  `train/{archive_id}/batch_*.parquet`).
- Val/test split is small: always download it fully, in either option.

## Subsetting for large datasets

The dataset is expected to grow to 100 GB – 1 TB, far beyond a pod's disk.
`snapshot_download` has no byte budget, so subsetting is done by selecting a
deterministic, evenly spread subset of Parquet shards and passing their exact
paths as `allow_patterns`:

1. `HfApi.list_repo_tree(repo_id, repo_type="dataset", path_in_repo=split,
   recursive=True)` lists the split's files with sizes.
2. `select_shard_subset(shards, max_bytes)` (pure function, unit-tested):
   - shards are sorted by path (deterministic; no Python `hash`);
   - if the total fits the budget, all shards are kept;
   - otherwise the target shard count is
     `n = max(1, floor(len(shards) * max_bytes / total_size))` and shards are
     taken at evenly spaced indices `round(i * len(shards) / n)`, which spans
     all archives/dates rather than only the earliest shards;
   - if the selected shards still exceed the budget (uneven shard sizes), the
     last ones are dropped until within budget.
3. `snapshot_download(..., allow_patterns=[<selected paths>])`.

Because shard selection is uniform over the whole repo, the subset is a
representative sample of games, not a temporal prefix. Note the archive
pipeline sorts rows by `game_id` *within* a chunk, so any single shard is
roughly random already; spreading across archives adds variety.

Size-bounded subset + Xet resumability composes well: a preempted pod re-runs
`download_split` with the same `max_bytes`, the same shard list is re-selected
(repo content unchanged), and cached shards are skipped.

## Decision

**Option B is the default.** One resumable download, zero changes to the
training data path, best shuffle fidelity and throughput. Revisit streaming
(Option A) only if the dataset outgrows pod disk. With a size-bounded,
evenly spread shard subset (above), pod disk no longer bounds dataset size.

Planned implementation:

1. New `src/training/hf_data.py`:
   - `download_split(repo_id, split, cache_dir=None, max_bytes=None)` →
     `Path` pointing at the snapshot's split directory (wraps
     `snapshot_download` with `repo_type="dataset"`;
     `allow_patterns=[f"{split}/**"]` when unbounded, the selected shard
     paths when bounded);
   - `select_shard_subset(shards, max_bytes)` — pure selection function.
2. `--data-repo`, `--data-cache`, `--data-limit-gb` CLI flags on
   `train_masked_transformer.py`: when `--data-repo` is set, train/val dirs
   resolve to `download_split(repo, "train", …, max_bytes=…)` /
   `download_split(repo, "test", …)`; otherwise the existing
   `--train-dir` / `--val-dir` flags apply as today.

## Pod hygiene

- `HF_TOKEN` is consumed at runtime by `huggingface_hub` from the
  environment; it must never be logged or persisted (see AGENTS.md secrets
  policy).
- On preemption: re-run the same command; the snapshot download resumes and
  `--resume` restores training state.
