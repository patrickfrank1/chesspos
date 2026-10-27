# Chess Position Embeddings

## Project Setup

The supported development environment is an NVIDIA OpenShell sandbox built from
the shared Dev Container image (`.devcontainer/Dockerfile`). The same image also
works as a plain Dev Container, but OpenShell is the default because it enforces
a default-deny network policy and injects credentials at the edge instead of
placing them in the workload.

### Prerequisites

- Docker on the host
- The [OpenShell](https://github.com/NVIDIA/OpenShell) CLI and a selected gateway
- VS Code with the Remote - SSH extension
- `uv` for work outside the sandbox

### 1. Install OpenShell

```bash
curl -LsSf https://raw.githubusercontent.com/NVIDIA/OpenShell/main/install.sh | sh
openshell gateway list
```

The installer normally starts and selects a local gateway. If no gateway is
selected, add one with:

```bash
openshell gateway add http://127.0.0.1:18080 --local --name local
openshell gateway select local
```

### 2. Export credentials

Export these in the host shell before creating the sandbox. Never commit them or
paste them into a terminal that writes to a log; `.env*` files are ignored and
excluded from sandbox uploads.

```bash
export DEEPSEEK_API_KEY=...
export ZAI_API_KEY=...
export GITHUB_TOKEN=...        # fine-grained token, Contents: read/write on patrickfrank1/chesspos
export HF_TOKEN=...            # HUGGING_FACE_HUB_TOKEN is also accepted
export VAST_AI_API_KEY=...     # VASTAI_API_KEY and VAST_API_KEY are also accepted
```

`VASTAI_API_KEY` is the name `scripts/provision_vast.py` reads; `VAST_API_KEY`
is the name the upstream `vastai` CLI reads. The OpenShell provider exposes
placeholders under all three names, so either works inside the sandbox.

### 3. Create the sandbox

```bash
scripts/openshell/create_sandbox.sh
```

The script imports the provider profiles, creates the providers from the
exported variables, builds `chesspos-agent:local`, creates the `chesspos`
sandbox, uploads the repository to `/workspace`, and runs
`uv sync --extra cpu`. It then opens VS Code in the sandbox.

Reconnect later with:

```bash
openshell sandbox connect chesspos --editor vscode
```

On a headless VM, create without launching VS Code:

```bash
NO_EDITOR=1 scripts/openshell/create_sandbox.sh
```

### 4. Develop inside the sandbox

The sandbox opens in `/workspace`. Run `opencode` from the integrated terminal:

```bash
opencode
opencode models
opencode -m deepseek/deepseek-chat
```

Git push works through the `github` provider and is restricted to the repository
selected by `GITHUB_REPO` (default: `patrickfrank1/chesspos`). Hugging Face and
Vast.ai calls use the `huggingface` and `vastai` providers, so the workload never
sees the real tokens:

```bash
uv run python scripts/provision_vast.py
uv run hf download patrickfrank1/chess-positions --repo-type dataset
```

### Syncing files

The sandbox workspace is a copy, not a bind mount:

```bash
scripts/openshell/sync.sh out   # sandbox -> host
scripts/openshell/sync.sh in    # host -> sandbox
```

### Rebuilding the image

After changing `.devcontainer/Dockerfile`, recreate the sandbox:

```bash
RECREATE=1 scripts/openshell/create_sandbox.sh
```

After changing only network rules, re-render and apply the policy to the running
sandbox:

```bash
sed "s|__GITHUB_REPO__|${GITHUB_REPO:-patrickfrank1/chesspos}|g" \
  openshell/policy.yaml > .openshell/policy.yaml
openshell policy set chesspos --policy .openshell/policy.yaml --wait
```

Filesystem, Landlock, and process settings require a recreate.

### Dev Container fallback

Open the repository in VS Code and choose **Reopen in Container**. The
Dev Container uses the same image, runs `uv sync --extra cpu` via
`.devcontainer/post-create.sh`, and intentionally has no root user, no
Docker-in-Docker, and no Docker socket mount.

### Sandbox network access

`openshell/policy.yaml` denies egress by default and allows only PyPI/PyTorch,
Hugging Face, npm/context7, `models.dev`, the VS Code server and marketplace,
and GitHub read plus repository-scoped push. Add hosts there when a task needs
them; the policy advisor can propose narrow rules via `openshell rule get
chesspos --status pending`.

The GitHub MCP from `opencode.json` is not wired into the sandbox and fails
closed; use the GitHub provider for push, or add explicit
`api.githubcopilot.com` rules if you want that MCP inside the sandbox.

## Cheat Sheet

- Generate training positions

    uv run python -m src.run.generate_hf_dataset --config configs/pipeline.yaml

- Train a neural network

    uv run python -m src.run.train_masked_transformer

- Evaluate a trained network with the notebooks in `src/run/`
  (`evaluate_token_models.ipynb`)

## Generate HuggingFace Dataset

Generates chess position datasets from PGN files and pushes them to HuggingFace Hub.

### Setup

1. **Authenticate with HuggingFace Hub:**

   Inside the OpenShell sandbox, the `huggingface` provider injects `HF_TOKEN`
   as an opaque placeholder at the approved endpoint, so no login is needed.
   Outside the sandbox, export `HF_TOKEN` or run:

   ```bash
   uv run hf auth login
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
| `output_dir` | `--output-dir` | *(none)* | Write parquet shards here in dry-run mode |
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

#### Full local run over `data/raw` as a Ray job

Example: process everything in `data/raw/` in a single batch and write the
Parquet shards locally (dry run, no Hub push):

```bash
uv run ray job submit --address http://127.0.0.1:8265 \
  --runtime-env-json '{"working_dir": ".", "excludes": ["data", "test", "docs", "notebooks"]}' \
  -- python -m src.run.generate_hf_dataset \
    --repo local/full-run \
    --data-path "$PWD/data/raw" \
    --output-dir "$PWD/data/processed" \
    --dry-run --batches 1 --batch-size 5000000 --workers 12
```

Two details matter here:

- **`--runtime-env-json` with `excludes`:** job mode packages the working
  directory and uploads it to the cluster, with a 500 MiB limit for a local
  `working_dir`. `data/` (several GB) must be excluded from that package.
- **Absolute paths for `--data-path` and `--output-dir`:** the job runs in an
  uploaded *copy* of the repo, not in the real directory, so relative paths
  point into the copy. Absolute paths read/write the real `data/` on the
  machine (for a single-node local cluster this is what you want).

A submitted job keeps running even if the submitting terminal is closed;
stream the logs with `uv run ray job logs --follow <job-id>` or watch them in
the dashboard.

Known issue: `--debug` (Ray `local_mode=True`) currently fails at `ray.init`
on Ray 2.54 with a `working_dir` URI validation error. Run without `--debug`
(a normal local cluster with `--workers 1` works fine).

### Training on a Ray cluster

The training job (`src/run/train_ray.py`) runs on the same Ray cluster as the
ETL via Ray Train (`TorchTrainer`). Data is snapshot-downloaded from the
HuggingFace dataset repo inside each training worker, so the ETL stays a
separate job on the same cluster.

On the training machine (e.g. a GPU box), start/join the cluster and submit:

```bash
uv run ray start --head   # or point RAY_ADDRESS at the existing head node
uv run ray job submit --address http://127.0.0.1:8265 \
  --runtime-env-json '{"working_dir": ".", "excludes": ["data", ".venv", "ray_results"]}' \
  -- python -m src.run.train_ray --config configs/training.yaml
```

Notes:

- Ray Train resolves checkpoints into `--storage-path` (default
  `ray_results/`); use a path on a shared filesystem for multi-node clusters.
- `--num-training-workers > 1` enables DDP; with one GPU per worker keep
  `--use-gpu` (auto-detected on the training machine).
- **Run the submit command through `uv run --extra gpu` (or `uv run --active`):**
  when the driver runs under `uv run`, Ray replicates the same `uv run`
  invocation on the workers; without the extra they would build an environment
  without torch.
- MLflow tracking and checkpoint upload to the HF bucket reuse the same flags
  as the standalone trainer (`--mlflow-uri`, `--upload-bucket`).

## Tools

- Start ML Flow UI, in correct python venv

    mlflow ui