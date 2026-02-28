## Context

The chesspos project generates chess position datasets for ML training. Currently, `generate_positions.py` saves data as `.npz` files to local disk with no versioning. The previous DVC-based versioning was removed. This design introduces HuggingFace Hub integration for dataset versioning and distribution.

**Current Pipeline:**
- `extract_board()` reads PGN files, samples positions by ELO and ply
- `board_to_token_sequence()` encodes positions as (69,) int8 arrays
- `save_train_test()` writes `.npz` files to `data/train/` and `data/test/`

**Stakeholders:** ML engineers training models on chess positions

## Goals / Non-Goals

**Goals:**
- Generate and push chess position datasets to HuggingFace Hub
- Support batch processing with configurable sizes
- Enable dataset versioning via HF Hub commits/tags
- Maintain memory efficiency (never load full data in memory)
- Support streaming for downstream consumers
- Allow appending new batches for future versions

**Non-Goals:**
- Modifying existing `extract_board()` sampling logic
- Adding position metadata (game source, ELO, ply)
- Real temporal window extraction (currently duplicates single position 10x)
- Supporting tensor or bitboard encodings (token_sequence only)

## Decisions

### 1. Use HuggingFace Datasets Library with IterableDataset

**Choice:** Use `datasets.IterableDataset.from_generator()` with Parquet format

**Alternatives:**
- Direct `huggingface_hub` API — More control but requires manual dataset schema management
- Custom S3/upload logic — No built-in versioning, more maintenance
- Keep `.npz` files — No versioning, harder to share
- Arrow format — Less compression efficiency for numerical data

**Rationale:** `IterableDataset` with Parquet provides:
- Memory-efficient streaming generation
- Built-in schema management via `features` parameter
- `push_to_hub()` with `path_in_repo` for batch-level control
- Seamless streaming for consumers via `load_dataset(..., streaming=True)`
- Better compression than Arrow for numerical data

### 2. Batch-at-a-Time Processing with Append Support

**Choice:** Generate and push one batch at a time, organized by version

**Implementation Pattern** (from proposal's minimum sample):
```python
batch_dataset = IterableDataset.from_generator(
    lambda: (sample for sample in zip(windows, scalars)),
    features={
        "window": Array2D(dtype=np.int8, shape=(10, 69)),
        "scalars": Array2D(dtype=np.int8, shape=(3,))
    }
)
batch_dataset.push_to_hub(
    repo_name,
    path_in_repo=f"train/batch_{i+1}.parquet",
    token=True,
    commit_message=f"Add batch {i+1}"
)
```

**Rationale:** Memory-safe approach. Each batch is independent. Future versions can append new batches without modifying existing data.

### 3. Dataset Schema

**Choice:** Follow proposal's schema with token_sequence encoding

| Field | Shape | Type | Description |
|-------|-------|------|-------------|
| window | (10, 69) | int8 | 10 copies of token_sequence |
| scalars | (3,) | int8 | Placeholder zeros |

**Rationale:** Maintains compatibility with existing training code that expects (10, 69) windows. Real temporal extraction is a future enhancement.

### 4. CLI Configuration via argparse

**Choice:** Standard argparse with sensible defaults

```bash
python -m src.run.generate_hf_dataset \
    --repo your-username/chesspos-positions \
    --batches 3 \
    --batch-size 100000 \
    --train-ratio 0.95 \
    --data-path ./data/raw
```

**Rationale:** Simple CLI matches existing `generate_positions.py` pattern.

## Implementation Approach

Following the proposal's minimum sample pattern:

1. **Initialize repo** — Clone or create HF dataset repo locally
2. **Generate batch** — Use `extract_board()` → `board_to_token_sequence()` → duplicate 10x
3. **Create IterableDataset** — Use `from_generator()` with proper features schema
4. **Split train/test** — 95/5 split per batch
5. **Push to hub** — Use `push_to_hub()` with `path_in_repo` for train/test separation
6. **Commit** — Each batch gets its own commit with descriptive message

**Output structure on HF Hub:**
```
train/
  ├── batch_001.parquet  (95k)
  ├── batch_002.parquet  (95k)
  └── batch_003.parquet  (95k)
test/
  ├── batch_001.parquet  (5k)
  ├── batch_002.parquet  (5k)
  └── batch_003.parquet  (5k)
```

## Risks / Trade-offs

| Risk | Mitigation |
|------|------------|
| HF Hub rate limits during push | Use batch sizes ≤100k; add `--dry-run` flag for testing |
| Authentication failure before push | Check `huggingface-cli login` status at startup with clear error message |
| PGN files exhausted mid-batch | Catch `StopIteration`, push partial batch, warn user |
| Dataset grows unversioned | Document tagging workflow; suggest version tags after each run |

## Migration Plan

1. **Create new script** `src/run/generate_hf_dataset.py`
2. **Test locally** with `--dry-run` flag (creates local dataset, skips push)
3. **Initial push** to new HF repo with single batch
4. **Verify streaming** — `load_dataset(..., streaming=True)` works correctly
5. **Iterate** with additional batches as needed

**No rollback needed** — this is additive, existing `.npz` pipeline remains untouched.

## Open Questions

1. Should we add a `--resume` flag to continue from last batch number?
   - **Recommendation:** Defer to future enhancement; for now, user manually specifies starting batch
2. Should train/test split ratio be configurable per-run or fixed?
   - **Recommendation:** Configurable via `--train-ratio` as specified in proposal
