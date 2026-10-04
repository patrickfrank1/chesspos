# Chess Position Embeddings

## Setup

npm install @fission-ai/openspec@latest
npx openspec init
npx openspec update
Run /opsx:onboard in opencode or another agent

## Cheat Sheet

- Generate training positions

    uv run python -m src.run.generate_hf_dataset --config configs/pipeline.yaml

- Train a neural network

    python -m src.run.train

- Evaluate a trained network by starting the notebook: `src/run/evaluate.ipynb`

## Generate HuggingFace Dataset

Generates chess position datasets from PGN files and pushes them to HuggingFace Hub.

### Setup

1. **Authenticate with HuggingFace Hub:**

   ```bash
   uv run huggingface-cli login
   ```

   You'll need a HuggingFace access token (get one at https://huggingface.co/settings/tokens).

2. **Prepare PGN files:**

   Place your PGN files in `data/raw/`. The directory already contains Lichess elite game files.

### Usage

You can configure the pipeline either via CLI flags or a YAML config file.

#### Via YAML config

Copy and edit the template:

```bash
cp configs/pipeline.yaml my_pipeline.yaml
```

Run with the config:

```bash
uv run python -m src.run.generate_hf_dataset --config my_pipeline.yaml
```

CLI flags override YAML values when both are provided:

```bash
uv run python -m src.run.generate_hf_dataset \
  --config my_pipeline.yaml \
  --min-elo 2500 \
  --dry-run
```

#### Via CLI only

**Basic usage (dry run to test locally):**

```bash
uv run python -m src.run.generate_hf_dataset --repo your-username/test-dataset --dry-run --batches 1
```

Note: even a dry run authenticates with HuggingFace Hub (the client checks the token on startup). Use `--debug` for single-threaded local mode — but see the known issue below.

**Push to HuggingFace Hub:**

```bash
uv run python -m src.run.generate_hf_dataset --repo your-username/chesspos-positions --batches 3
```

#### Configuration reference

Precedence: **CLI args > YAML file > defaults**.

| YAML key | CLI flag | Default | Description |
|----------|----------|---------|-------------|
| `repo_name` | `--repo` | *required* | HuggingFace Hub repository name |
| `data_path` | `--data-path` | `./data/raw` | Directory with `.pgn` files |
| `batch_size` | `--batch-size` | `100000` | Games per batch |
| `train_ratio` | `--train-ratio` | `0.95` | Train/test split ratio |
| `num_batches` | `--batches` | `3` | Number of batches to generate |
| `dry_run` | `--dry-run` | `false` | Generate locally without pushing |
| `resume` | `--resume` | `false` | Continue from last batch on Hub |
| `create_card` | `--create-card` | `false` | Push a dataset card |
| `preprocessing.worker_count` | `--workers` | `4` | Ray parallel workers |
| `preprocessing.memory_limit_mb` | `--memory` | `4096` | Memory per worker (MB) |
| `preprocessing.debug` | `--debug` | `false` | Single-threaded local mode (see known issue) |
| `sampling.tiers` | *(YAML only)* | see below | Tiered game subsampling by player strength |
| `sampling.min_time_control_seconds` | *(YAML only)* | `300` | Exclude games with base time below this (e.g. bullet) |

Default sampling tiers:

```yaml
sampling:
  tiers:
    - min_elo: 2400
      rate: 1.0
    - min_elo: 2000
      rate: 0.4
    - min_elo: 0
      rate: 0.001
```

A game is matched to the strictest tier both players qualify for (missing Elo counts as 0) and kept with that tier's `rate`.

**Example with custom settings:**

```bash
uv run python -m src.run.generate_hf_dataset \
  --repo your-username/chesspos-positions \
  --batches 5 \
  --batch-size 50000 \
  --workers 8 \
  --train-ratio 0.9 \
  --create-card
```

### Monitoring with the Ray Dashboard

The Ray dashboard is included via the `ray[default]` dependency and starts
automatically when a run begins:

- **Dashboard UI:** http://127.0.0.1:8265 (the URL is printed when Ray starts).
  The *Data* tab shows live per-operator progress of the ETL pipeline
  (read → extract → limit → encode → filter); the *Jobs* and *Cluster* tabs
  show workers, CPU/memory usage, and errors.
- **Terminal:** Ray Data prints per-stage progress bars to stdout while the
  pipeline runs.
- **Logs:** driver and worker logs live under `/tmp/ray/session_latest/logs/`
  (Ray Data logs in the `ray-data/` subdirectory). Logs persist after the run
  ends until the next session starts.

To watch a long run in the dashboard as a proper *job*, start a long-lived
local cluster first and submit through it:

```bash
uv run ray start --head
uv run ray job submit --address http://127.0.0.1:8265 -- \
  python -m src.run.generate_hf_dataset --config configs/pipeline.yaml
```

The job then appears in the dashboard's *Jobs* tab, including stdout/stderr,
even if the submitting terminal is closed. Stop the cluster afterwards with
`uv run ray stop`.

Known issue: `--debug` (Ray `local_mode=True`) currently fails at `ray.init`
on Ray 2.54 with a `working_dir` URI validation error. Run without `--debug`
(a normal local cluster with `--workers 1` works fine).

## Tools

- Start ML Flow UI, in correct python venv

    mlflow ui

- Export dependencies to requirements.txt

    poetry export > requirements.txt

## Notes

### Milvus

Start local milvus db instance with:

`docker compose -f milvus-2-3-10-standalone-docker-compose.yml up -d`

Stop the instance with:

`docker compose -f milvus-2-3-10-standalone-docker-compose.yml down`

- could only get milvus 2.3.1 to work, so use that for now
- but had to downgrade python to 3.9, because of compatibility issues
- and only works with recent tensorflow version, so it's incompatible with aws sage maker
  - maybe I need to build a different toolchain for different python versions

## TODOs

- [ ] Write to db from .npy files
  - [ ] write tokenized positions with some metadata and id
  - [ ] write embeddings generated from a model
- [ ] Write to db from .pgn file
  - maybe some refactoring is needed
- make embeddings better for search
  - document approaches
  - make a plan