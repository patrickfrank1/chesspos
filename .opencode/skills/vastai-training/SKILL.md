---
name: vastai-training
description: >
  Set up, launch, resume, and monitor chesspos training on vast.ai: the puzzle
  theme classifier and foundation masked-transformer runs, watchdogs, crash
  recovery, log/GPU checks, checkpoint syncing, and the HuggingFace retrieval
  benchmark (build/upload/pull, eval at epoch end). Use when starting or
  restarting a training job, debugging a stalled or flat run, or evaluating
  checkpoints.
---

# vast.ai training runs

Prerequisite: the box is provisioned, SSH works, and the environment is
installed. Those steps (secrets, provisioning, SSH keys, system torch, pip
deps, HF token, flash-attn, code deployment, teardown) live in the
**`vastai-gpu`** skill. Do not duplicate them here.

Full plan context: `docs/plan_gpu_training_vastai.md`.

## 1. Training setup

### Data

**Puzzle classifier** needs local parquet shards with `puzzle_id`, `packed`,
and theme columns under `data/processed/puzzles`. Build them from the Lichess
dump on the box (system python, Ray is already installed):

```bash
head -n 500001 data/raw/lichess_db_puzzle.csv > /tmp/puzzles_500k.csv
PYTHONPATH=/workspace/chesspos python3 -m src.run.generate_puzzle_dataset \
  --input /tmp/puzzles_500k.csv --output-dir data/processed/puzzles \
  --workers 16 --memory 4096
```

500k puzzles produced ~412 MiB in ~85 s on the pod. The full CSV is ~1.14 GB;
`head -n` is the cheap way to bound a run. If the CSV is not on the box,
download `https://database.lichess.org/lichess_db_puzzle.csv.zst` and
decompress it.

**Foundation MLM** runs stream from the HF dataset repo
`patrickfrank1/chess-positions` (`--data-repo`, `--data-cache`).

**Foundation encoder checkpoint** (the frozen backbone used by the puzzle
classifier): verify architecture compatibility before launching — vocab 52,
expected `d_model`/`n_layers`, and slot embeddings. A quick check:

```bash
python3 - <<'PY'
import torch
ck = torch.load("data/models/d512l8_20261021/best_step48000.pt",
                map_location="cpu", weights_only=False)
print(ck["model_config"])
PY
```

Classifier checkpoints store `encoder_config`/`encoder_state` and
`head_config`/`head_state` instead.

### Retrieval benchmark from HuggingFace

The benchmark is built once from the puzzle corpus and uploaded to the **same
dataset repo** that holds the puzzle parquet shards, under
`benchmarks/retrieval/`. Defaults: 10k positions, 1k queries, cross-puzzle
exact/near duplicates (grades 0–2) plus theme clusters. Strategic queries use
balanced pools (10 same-theme positives + 490 non-theme negatives each); the
default cluster list contains the specific endgame types, `promotion`, and
`zugzwang` — the broad `endgame` phase label (~50% of puzzles) is excluded.

Build and upload (local machine with `uv`; on the box use
`PYTHONPATH=/workspace/chesspos python3 -m ...` and export `HF_TOKEN` from
`/root/.hf_token` first):

```bash
uv run python -m src.run.build_retrieval_benchmark \
  --data-dir data/processed/puzzles \
  --output-dir data/benchmarks/retrieval \
  --val-ratio 0.1 --max-positions 10000 --max-queries 1000 \
  --repo-id <org>/<puzzle-dataset-repo> --revision v1 \
  --commit-message "retrieval benchmark v1"
```

Train pulling it from the Hub (only `benchmarks/retrieval/**` is downloaded,
and HF caches the snapshot):

```bash
... --retrieval-benchmark <org>/<puzzle-dataset-repo> \
    --retrieval-benchmark-revision v1
```

Rules:

- Pass the **same `--val-ratio`** as training (default 0.1); the benchmark
  only contains validation-split puzzles, so train/benchmark stay disjoint.
- Rebuild and bump `--revision` when the corpus, `--limit`, or `--val-ratio`
  changes. Commits/branches/tags provide versioning.
- A local benchmark directory is also accepted (handy before uploading):
  `--retrieval-benchmark data/benchmarks/retrieval`.
- A missing manifest fails at startup — there is no silent skip.

Standalone evaluation of any encoder checkpoint:

```bash
uv run python -m src.run.evaluate_retrieval \
  --benchmark <org>/<puzzle-dataset-repo> --revision v1 \
  --checkpoint data/models/<run>/best.pt
```

At every epoch with `--retrieval-benchmark` set, training logs and stores in
the checkpoint: `retrieval_near_recall@1/5/10` (with bootstrap CI),
`retrieval_near_map`, `retrieval_near_ndcg@*`, `retrieval_strategic_*`,
`retrieval_rank_effective`, `retrieval_alignment`, `retrieval_uniformity`.
Strategic metrics use balanced pools per query (default 10 same-theme
positives + 490 non-theme negatives); per-theme MAP/chance/lift are logged as
`retrieval_strategic_theme_<theme>_{map,chance_map,map_lift,n_queries}`.
The classifier encoder is currently frozen, so these numbers stay constant
within a run; they become meaningful once encoder fine-tuning starts (and are
still useful as a whitening/normalization baseline).

### Puzzle classifier command

```bash
cd /workspace/chesspos
PYTHONPATH=/workspace/chesspos python3 -m src.run.train_puzzle_classifier \
  --data-dir data/processed/puzzles --theme sacrifice \
  --checkpoint data/models/d512l8_20261021/best_step48000.pt \
  --output-dir data/models/puzzle_classifier_attn_focal \
  --epochs 3 --batch-size 128 --num-workers 4 --log-every 50 \
  --loss focal --limit 250000 \
  --head-hidden-dim 256 --pooling attention \
  --retrieval-benchmark <org>/<puzzle-dataset-repo> \
  --retrieval-benchmark-revision v1
```

Key flags:

- `--limit` bounds the corpus (halves steps/epoch); the benchmark split is
  independent (hash-based), so changing `--limit` does not leak train puzzles.
- `--loss focal` auto-calibrates `gamma`/`alpha` from the train label counts;
  `--focal-gamma`/`--focal-alpha` override them. Startup prints both.
- `--pooling attention|mean`, `--head-hidden-dim N`, `--dropout`.
- Validation loss is always BCE so runs remain comparable across losses.
- `last.pt`/`best.pt` (best by validation PR-AUC) are written per epoch. This
  entry point does **not** support `--resume`; relaunching starts a fresh
  head, so only restart when you mean to.

Sanity check the loss: with `--loss focal`, startup shows
`focal_gamma=... focal_alpha=...` and the running training loss should be
roughly 0.03–0.1. A loss around 0.63–0.69 means BCE is being used — verify
that `train_epoch(...)` is called with `focal=focal`.

### Foundation MLM command

Use the watchdog below; the trainer supports `--resume` from `last.pt`, which
makes crash-looping self-healing.

## 2. Launching

Keep long-lived commands detached and redirect all stdio:

```bash
ssh ... 'setsid nohup /workspace/watchdog_<run>.sh > /dev/null 2>&1 < /dev/null & echo started'
```

The ssh channel may hang until the child exits; expect it and verify with a
separate short ssh call (`pgrep -af ...`).

### Watchdog pattern (recommended)

Auto-resumes from `last.pt` after crashes and exits at the step target. Keep
it at `/workspace/watchdog_<run>.sh` (template in repo history /
`/tmp/opencode/watchdog_d256l3.sh` pattern):

```bash
#!/bin/bash
# Watchdog: (re)start training, resuming from last.pt after crashes.
set -uo pipefail
export HF_TOKEN="$(cat /root/.hf_token)"
export PYTHONPATH=/workspace/chesspos
export TORCHINDUCTOR_COMPILE_THREADS=1   # only helps if compile works at all
cd /workspace/chesspos
STEPS=12000
DIR=data/models/<run>
mkdir -p "$DIR"
while true; do
  step=0
  if [ -f "$DIR/last.pt" ]; then
    step=$(python3 - "$DIR/last.pt" <<'EOF'
import sys, torch
print(torch.load(sys.argv[1], map_location="cpu", weights_only=False)["step"])
EOF
    ) || step=0
  fi
  if [ "$step" -ge "$STEPS" ]; then
    echo "$(date -u +%FT%TZ) training complete at step $step" >> "$DIR/watchdog.log"
    break
  fi
  echo "$(date -u +%FT%TZ) launching training (resume from step $step)" >> "$DIR/watchdog.log"
  resume_args=()
  if [ "$step" -gt 0 ]; then resume_args=(--resume "$DIR/last.pt"); fi
  python3 src/run/train_masked_transformer.py \
    --data-repo patrickfrank1/chess-positions --data-cache data/hf-cache \
    --steps "$STEPS" --batch-size 256 --lr 3e-4 --warmup 500 \
    --streaming --num-workers 16 --log-every 50 --eval-every 2000 \
    --save-every 1000 --bf16 \
    --upload-bucket patrickfrank1/chesspos-checkpoints \
    --upload-prefix run_<run>_streaming \
    --checkpoint-dir "$DIR" --log-file "$DIR/train.log" \
    "${resume_args[@]}" >> "$DIR/nohup.out" 2>&1
  echo "$(date -u +%FT%TZ) training exited (status $?), restarting in 10s" >> "$DIR/watchdog.log"
  sleep 10
done
```

### Killing and relaunching

`pkill -f <pattern>` matches the *shell running your ssh command* whenever the
pattern text appears anywhere in it (including a later `setsid ...` in the
same command — this killed relaunches twice). Rules:

- Use a bracket pattern that cannot match its own literal text:
  `pkill -f 'watchdog_d256l[3].sh'`, `pkill -f 'train_puzzle_classif[i]er'`.
- Never put pkill and the relaunch in the **same** ssh command; use two
  separate calls.

Restart recipe:

1. Kill the trainer (separate ssh call; confirm `pgrep` is empty).
2. Update code on the box via git (see `vastai-gpu`): fetch + reset to the
   pushed commit. If you scp'd uncommitted files, remember a later reset
   discards them.
3. Relaunch detached, then verify the first log lines in a new ssh call.

## 3. Monitoring

**Polling discipline: never `sleep` longer than 15 s per command** — the
operator aborts long sleeps. Prefer short single checks; startup (snapshot
download, unigram pass, compile warmup) takes 1–3 min, so a few spaced short
polls are enough.

```bash
ssh ... 'tail -5 /workspace/chesspos/data/models/<run>/train.log'
ssh ... 'tail -5 /workspace/chesspos/data/models/<run>/nohup.out'
ssh ... 'tail -3 /workspace/chesspos/data/models/<run>/watchdog.log'   # crash-loop check
ssh ... 'nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader'
ssh ... 'free -m | sed -n 2p; cat /proc/loadavg'
```

Puzzle classifier logs:

```bash
ssh ... 'tail -f /workspace/chesspos/data/models/<run>/nohup.out'
```

Interpreting startup (foundation): `params=... bf16=... compile=...` banner →
snapshot download → unigram pass (expect "from sidecars") → val split load →
first steps. VRAM jumping with GPU at 0% is normal warmup, not a hang.

Reference profiles (4090, bf16): 25M model, batch 64, seq 4096: ~490k tok/s,
24/47 GB VRAM, GPU ~97%. 2.4M model (d256/3L/4h/ff1024), batch 256: ~2.9M
tok/s, 41/47 GB VRAM, GPU 100%.

If GPU shows 0% util and idle clocks, the run has died — read
`watchdog.log` (relaunch loop) and `nohup.out` (traceback). Past causes:
`ValueError: low >= high` from single-segment games, missing Brotli upgrade,
missing gcc, triton `invalid device context` with `--compile` (see
`vastai-gpu`).

For the puzzle classifier, an epoch ends with
`epoch=N train_loss=... val_loss=... auroc=... pr_auc=... f1=... threshold=...`
followed by `retrieval_*` lines when a benchmark is configured. A flat
~0.63 focal loss means the loss is not wired (see setup).

## 4. Checkpoints and artifacts

- Foundation: `last.pt` (full training state, for `--resume`) and `best.pt`
  (eval metrics) are written atomically every `--save-every` steps.
- Puzzle classifier: `last.pt` and `best.pt` (best validation PR-AUC) per
  epoch, including encoder/head config and state, planner state, and metrics
  (retrieval metrics included when configured).
- Sync checkpoints off the box with a background loop:

```bash
hf buckets create chesspos-checkpoints || true
nohup sh -c 'while true; do
  hf buckets sync ./models/run_full hf://buckets/patrickfrank1/chesspos-checkpoints/run_full/ \
    --exclude nohup.out || true; sleep 900; done' &
```

Pull locally:
`hf buckets sync hf://buckets/patrickfrank1/chesspos-checkpoints/run_full/ ./models/run_full/`

### Finishing a run (one-shot shutdown)

For operator-offline runs, use a local detached script (`setsid nohup`, session
leader so it survives the agent session) that sleeps, then over ssh: kill
watchdog → kill trainer → grace for in-flight uploads → `hf buckets sync`
checkpoints + logs → verify exit codes → `vastai destroy instance <ID> -y
--api-key "$VASTAI_API_KEY"` (sourced from local `.env`). On any sync failure:
leave the pod running, skip destroy, log everything. Instance id:
`vastai show instances --raw` via the `--api-key` per-call pattern. Keep the
step count checked: a watchdog auto-resume run can finish `--steps` ahead of
schedule, and the trainer uploads its final checkpoint itself ("uploaded last
to ..."). Teardown template: `/tmp/opencode/shutdown_d256l3.sh` (2026-10-19).

## 5. Debug playbook

| Symptom | Likely cause / fix |
|---|---|
| `TypeError: process() takes no keyword arguments` during hf download | old Brotli; `pip install -U Brotli` (see `vastai-gpu`) |
| `Failed to find C compiler` | `apt-get install -y gcc` (see `vastai-gpu`) |
| Triton `invalid device context` | drop `--compile`; torch 2.2.1 driver mismatch (see `vastai-gpu`) |
| `ValueError: low >= high` | single-segment games in the batch; fix data/windowing |
| GPU 0%, idle clocks | process died; read `watchdog.log` and `nohup.out` |
| Focal loss ≈ 0.63 | `focal=focal` missing in the `train_epoch` call; run the focal wiring test |
| Retrieval metrics identical every epoch | expected while the encoder is frozen |
| Benchmark download 401 | plain ssh shell lacks `HF_TOKEN`; export from `/root/.hf_token` |

Teardown: `vastai destroy instance <ID> -y` only after checkpoints are safe
(see `vastai-gpu`).
