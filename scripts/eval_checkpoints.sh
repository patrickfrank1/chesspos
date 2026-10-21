#!/bin/bash
# Evaluate every best_step checkpoint of a training run on this machine and
# sync the reports (metrics, PCA plots, neighbor HTML) to the HF bucket next
# to the checkpoints. Designed to run on the GPU box before teardown; skips
# checkpoints whose metrics are already complete, so it is safe to re-run.
#
# Requires: HF_TOKEN (or /root/.hf_token), hf CLI, the repo's python deps.
#
# Usage:
#   bash scripts/eval_checkpoints.sh \
#     --bucket hf-name/bucket \
#     --prefix run_d512l8_20261021 \
#     --data-dir data/test \
#     --out-dir data/eval/d512l8_20261021
set -uo pipefail

usage() {
    grep '^#' "$0" | sed 's/^# \{0,1\}//' | sed -n '2,20p'
    exit 1
}

BUCKET="" ; PREFIX="" ; DATA_DIR="" ; OUT_DIR=""
DEVICE="cuda" ; BATCH_SIZE="64" ; CKPT_DIR="/workspace/eval_ckpts"

while [ $# -gt 0 ]; do
    case "$1" in
        --bucket) BUCKET="$2" ; shift 2 ;;
        --prefix) PREFIX="$2" ; shift 2 ;;
        --data-dir) DATA_DIR="$2" ; shift 2 ;;
        --out-dir) OUT_DIR="$2" ; shift 2 ;;
        --device) DEVICE="$2" ; shift 2 ;;
        --batch-size) BATCH_SIZE="$2" ; shift 2 ;;
        --ckpt-dir) CKPT_DIR="$2" ; shift 2 ;;
        *) usage ;;
    esac
done

for var in BUCKET PREFIX DATA_DIR OUT_DIR; do
    if [ -z "${!var}" ]; then
        echo "missing required --$(echo "$var" | tr '_' '-' | tr 'A-Z' 'a-z')" >&2
        usage
    fi
done

if [ -z "${HF_TOKEN:-}" ] && [ -f /root/.hf_token ]; then
    export HF_TOKEN="$(cat /root/.hf_token)"
fi
if [ -z "${HF_TOKEN:-}" ]; then
    echo "HF_TOKEN not set and /root/.hf_token missing" >&2
    exit 1
fi

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"

echo "downloading checkpoints from $BUCKET/$PREFIX"
rm -rf "$CKPT_DIR"
mkdir -p "$CKPT_DIR"
hf buckets sync "hf://buckets/$BUCKET/$PREFIX/" "$CKPT_DIR/" --include 'best_step*.pt'
if ! ls "$CKPT_DIR"/best_step*.pt > /dev/null 2>&1; then
    echo "no best_step*.pt checkpoints found in bucket" >&2
    exit 1
fi

failed=0
total=0
for ckpt in "$CKPT_DIR"/best_step*.pt; do
    name="$(basename "$ckpt" .pt)"
    out="$OUT_DIR/$name"
    mkdir -p "$out"
    total=$((total + 1))
    if [ -s "$out/metrics.txt" ] && grep -q 'pc1_piece_corr' "$out/metrics.txt"; then
        echo "skip $name (already evaluated)"
        continue
    fi
    echo "eval $name"
    python3 -u -m src.run.eval_model \
        --checkpoint "$ckpt" \
        --data-dir "$DATA_DIR" \
        --device "$DEVICE" \
        --batch-size "$BATCH_SIZE" \
        > "$out/metrics.txt" 2>&1
    rc=$?
    echo "$name rc=$rc"
    if [ "$rc" -ne 0 ]; then
        failed=$((failed + 1))
    fi
done

echo "syncing eval reports to $BUCKET/$PREFIX/evals/"
hf buckets sync "$OUT_DIR/" "hf://buckets/$BUCKET/$PREFIX/evals/"
sync_rc=$?

echo "eval summary: total=$total failed=$failed sync_rc=$sync_rc"
if [ "$failed" -ne 0 ] || [ "$sync_rc" -ne 0 ]; then
    echo "EVAL FAILED - inspect metrics.txt files above" >&2
    exit 1
fi
echo "EVAL OK"
