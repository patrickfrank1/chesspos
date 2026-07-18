# AGENTS.md

Guidance for AI agents (and humans) working in this repository.

## Project

`chesspos` learns an embedding for chess positions. PGN games are parsed into
positions, encoded, published to a HuggingFace dataset repo, and consumed by a
Keras model for self-supervised training. The resulting embeddings feed
downstream tasks (semantic search, evaluation, pattern classification).

## Tech Stack

### Language & tooling
- **Python ≥ 3.11.8**, managed with **`uv`** (`uv.lock` is the source of truth).
- **Ruff** for lint and formatting; **pytest** for tests.
- **YAML** config files for the dataset generator
  (`src/run/generate_hf_dataset.py --config …`).

### ML framework
- **Keras 3** with **`keras-nlp`** transformer layers.
- Backend may be **TensorFlow** *or* **PyTorch** — Keras 3 supports both. Do not
  import `tensorflow` or `torch` directly in new code; go through `keras` /
  `keras.backend` so the backend stays swappable. Existing modules under
  `src/modeling/` and `src/run/train.py` still import `tensorflow` directly and
  are pending migration.

### Data processing
- **python-chess** for PGN parsing, board manipulation, and FEN handling.
- **Ray** (Ray Data) for distributed ETL in `src/dataset/etl.py`.
- **HuggingFace `datasets` + `huggingface_hub`** for dataset distribution as
  Parquet shards on the Hub.
- **NumPy** for in-memory tensors.

### Vector database
- **Not finalized.** The current implementation in `src/search/db.py`,
  `src/search/indexer.py`, and `src/run/build_vector_db.py` uses **Milvus** via
  `pymilvus`. Milvus is **deprecated and will be removed** — do not extend it.
- **LanceDB** is the leading candidate replacement (offers built-in vector
  indexes and ML-framework integration). Treat the vector store layer as in
  flux; new work should be written behind an abstraction so the backend can be
  swapped once a decision is made.

### Experiment tracking
- **MLflow** (`mlflow ui` to view runs).

## Data Pipeline

See [`docs/dataset_pipeline.md`](docs/dataset_pipeline.md) for the full
entity-relationship diagram and runtime flow. Summary:

- **Write side** (`src/dataset/etl.py`) — `ChessPositionDataset` composes
  `DatasetConfig`, `PreprocessingConfig`, `EncoderConfig`, and a
  `HuggingFaceClient`. Ray Data reads `*.pgn` → `PGNProcessor` extracts and
  samples positions (logistic ELO gate + ply² subsample) → one of three
  `PositionEncoder` implementations encodes each board → `train_test_split` →
  Parquet shards pushed to the Hub as `{train,test}/batch_{NNNN}_*.parquet`.
- **Read side** (`src/dataset/data_loader.py`) — `TrainingDataGenerator` loads
  the Hub dataset back via `datasets.load_dataset` and yields a
  `tf.data.Dataset` of `(window, target)` pairs with optional BERT-style
  masking (`mask_token_id=16`).
- **Encoders** (`src/dataset/position_encoder.py`) — `token_sequence` `(69,)`,
  `tensor` `(8,8,15)`, `bitboard` `(773,)`, all behind the `PositionEncoder`
  ABC and a registry (`get_encoder`).

> Note: there is a known schema mismatch between what `_encode_batch` writes
> (`encoded`, `ply`, …) and what `TrainingDataGenerator` reads (`window`,
> `scalars`). Documented in `docs/dataset_pipeline.md`.

## Development Workflow

1. **Branch from `main`** for every feature: `feature/<short-description>`.
2. **Optionally link to a GitHub issue** and/or an **OpenSpec** change
   (`.opencode/skills/openspec-*` and `openspec/` directory). Use the
   `openspec-new-change` / `openspec-apply-change` skills when working through
   an OpenSpec change.
3. **Implement** following existing conventions in `src/dataset/` — dataclasses
   for config, ABC + registry for pluggable components, static methods for Ray
   callables.
4. **Verify locally** before pushing:
   - `uv run ruff format && uv run ruff check`
   - `uv run pytest`
   - If you add a new library, confirm it resolves in `uv.lock`.
5. **Open a PR** into `main`. A CI pipeline runs the test suite on every merge
   to `main`; PRs are expected to be green before merge.
6. **Squash-merge** completed features into `main`.

## Using Context7 for Dependency Documentation

This repo has the **Context7** MCP server configured in `opencode.json`. Use it
to fetch current documentation for any dependency instead of guessing from
training data — especially for libraries with evolving APIs like
`python-chess`, `lancedb`, `keras-nlp`, `ray`, or `datasets`.

How to use it:

1. Call `context7_resolve-library-id` with the library name
   (e.g. `libraryName: "python-chess"`, `query: "how to read PGN games and
   iterate mainline moves"`). It returns Context7-compatible IDs such as
   `/niklasf/python-chess`.
2. Call `context7_query-docs` with that `libraryId` and a **specific** question
   (e.g. `"how to read PGN games and iterate mainline moves"`). Good queries
   include the API surface you need; bad queries are single keywords like
   `"board"`.
3. Cite the returned snippets when writing or modifying code, and prefer the
   documented API over assumptions.

When to use Context7:
- Adding or upgrading a dependency.
- Touching an unfamiliar library (e.g. the first time you write LanceDB code).
- Debugging a library-specific error.
- Migrating between backends (e.g. TensorFlow ↔ PyTorch via Keras 3).

When **not** to use it: refactoring business logic, writing tests against
internal modules, or general programming concepts — those don't need external
docs.

## Repository Layout

```
src/
  dataset/        # ETL + HF dataset client + position encoders (current focus)
  modeling/       # Keras model definitions
  training/       # Data generators for training
  search/         # Vector store (Milvus, deprecated) + indexer
  preprocessing/  # Legacy board representation / extraction helpers
  evaluation/     # Visualisation
  run/            # CLI entry points (generate_hf_dataset.py, train.py, …)
  utils/          # fileops, legacy data_loader
docs/             # Pipeline docs, research notes, framework evaluations
openspec/         # OpenSpec change specs
test/             # pytest suite
```
