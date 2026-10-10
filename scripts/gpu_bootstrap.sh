#!/usr/bin/env bash
# Bootstrap a GPU instance (vast.ai, base image: pytorch/pytorch) for training.
#
# Required env: HF_TOKEN
# Optional env: GITHUB_TOKEN (private repo https clone), REPO_REF
#               (default features/gpu-training), DATASET
#               (default patrickfrank1/chess-positions), TRAIN_STEPS
#               (default 12000), LAUNCH_TRAINING=1 to start a full run,
#               SYNC_INTERVAL (default 900 s; 0 disables checkpoint upload).
# Checkpoints sync to the HF bucket patrickfrank1/chesspos-checkpoints.
#
# Runbook: vastai create instance <ID> --image pytorch/pytorch --disk 40 --ssh --direct
#          vastai ssh-url <ID>; on the box:
#          HF_TOKEN=... GITHUB_TOKEN=... LAUNCH_TRAINING=1 bash gpu_bootstrap.sh
set -euo pipefail

REPO_URL="${REPO_URL:-https://github.com/patrickfrank1/chesspos.git}"
REPO_REF="${REPO_REF:-features/gpu-training}"
DATASET="${DATASET:-patrickfrank1/chess-positions}"
TRAIN_STEPS="${TRAIN_STEPS:-12000}"
SYNC_INTERVAL="${SYNC_INTERVAL:-900}"
BUCKET="hf://buckets/patrickfrank1/chesspos-checkpoints/run_full/"

if [ -z "${HF_TOKEN:-}" ]; then
    echo "HF_TOKEN must be set" >&2
    exit 1
fi

cd /workspace 2>/dev/null || cd "$HOME"

if [ ! -d chesspos ]; then
    if [ -n "${GITHUB_TOKEN:-}" ]; then
        git clone "https://x-access-token:${GITHUB_TOKEN}@github.com/patrickfrank1/chesspos.git" chesspos
    else
        git clone "$REPO_URL" chesspos
    fi
fi
cd chesspos
git fetch origin "$REPO_REF"
git checkout "$REPO_REF"
git pull --ff-only origin "$REPO_REF"

curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"

uv python install 3.12
uv sync
uv pip install "torch==2.14.1" --force-reinstall

.venv/bin/hf auth login --token "$HF_TOKEN"
.venv/bin/hf download "$DATASET" --repo-type dataset --local-dir data/processed

TRAIN_DIR=data/processed/data/train
VAL_DIR=data/processed/data/test

echo "smoke test: 20 steps"
PYTHONPATH=. .venv/bin/python src/run/train_masked_transformer.py \
    --steps 20 --batch-size 32 --max-seq-len 2048 \
    --train-dir "$TRAIN_DIR" --val-dir "$VAL_DIR" \
    --num-workers 4 --eval-batches 5 --eval-every 20 \
    --log-every 10 --bf16 --compile

if [ "${LAUNCH_TRAINING:-0}" = "1" ]; then
    mkdir -p models/run_full
    nohup env PYTHONPATH=. .venv/bin/python src/run/train_masked_transformer.py \
        --steps "$TRAIN_STEPS" --batch-size 32 --max-seq-len 2048 \
        --train-dir "$TRAIN_DIR" --val-dir "$VAL_DIR" \
        --num-workers 8 --log-every 50 --eval-every 1000 \
        --save-every 2000 --bf16 --compile \
        --checkpoint-dir models/run_full --log-file models/run_full/train.log \
        > models/run_full/nohup.out 2>&1 &
    echo "training launched (pid $!), log: models/run_full/train.log"

    if [ "$SYNC_INTERVAL" != "0" ]; then
        .venv/bin/hf buckets create chesspos-checkpoints 2>/dev/null || true
        nohup sh -c "while true; do
            .venv/bin/hf buckets sync ./models/run_full '$BUCKET' \
                --exclude nohup.out || true
            sleep $SYNC_INTERVAL
        done" > models/run_full/sync.log 2>&1 &
        echo "bucket sync launched (every ${SYNC_INTERVAL}s), log: models/run_full/sync.log"
    fi
fi
