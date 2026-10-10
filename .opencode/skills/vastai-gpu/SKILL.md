---
name: vastai-gpu
description: >
  Provision and operate vast.ai GPU instances for chesspos training:
  searching offers, creating instances, SSH access, running training jobs,
  monitoring, pushing checkpoints, and teardown. Use when renting a GPU,
  launching/debugging a remote training run, or cleaning up instances.
---

# vast.ai GPU operations

Lessons from the first real runs (2026-10-10). Full plan context:
`docs/plan_gpu_training_vastai.md`.

## Secrets rules (hard requirements)

- `VASTAI_API_KEY`, `HF_TOKEN`, `GITHUB_TOKEN` live in `.env` (repo root).
  Source with `set -a && source .env && set +a` — never print, echo, or
  `cat` variable values. Referencing names is fine; values are not.
- The vast.ai `create instance` response contains an `instance_api_key` —
  never dump raw API responses; parse and extract only what is needed.
- If a required secret is missing, stop and escalate to the human operator.

## Provisioning

Use `scripts/provision_vast.py` (wraps the `vastai` CLI, passes
`--api-key` per call so nothing is persisted):

```bash
uv run python scripts/provision_vast.py               # search -> pick -> confirm -> create -> wait -> ssh-url
uv run python scripts/provision_vast.py --comfort     # 2x minimum requirements
uv run python scripts/provision_vast.py --instance-id <ID>   # reconnect/wait for existing
uv run python scripts/provision_vast.py --destroy <ID>       # stops billing
```

Hard-won facts:

- **Offers vanish within a minute.** Show the list, get the user's pick,
  then create immediately. For pre-authorized runs pipe confirmation:
  `printf '\ny\n' | uv run python scripts/provision_vast.py` (blank = best
  value). A `success: false` create response can still create a phantom
  contract — always verify with `vastai show instances --raw` and destroy
  duplicates.
- **Server-side offer search bugs:** `ram` is not a valid key (use
  `cpu_ram`), and `cpu_ram>=`/`gpu_ram>=` comparisons return 0 results —
  filter RAM/VRAM client-side (the script does this). `driver_version` is
  a string; no numeric compare server-side. `dlperf_usd` is dead (all
  zeros) — sort by `dph_total`.
- **Sizing (profiled):** minimum 8 GB VRAM, 16 GB RAM, 4 cores, 20 GB
  disk; comfort = 2x each (`--comfort`). A 3.4M-param model at batch 32 x
  2048 tokens uses 5.2 GB VRAM, ~12 GB RAM, ~3 CPU cores.
- Instance ids are `new_contract` values from the create response.

## Connecting (SSH)

`vastai execute` only works on **stopped** instances. For a running box:

```bash
vastai attach ssh <ID> "$(cat vastai_gpu_key.pub)"   # per-instance key attach
vastai ssh-url <ID>                                   # -> ssh://root@HOST:PORT
ssh -i vastai_gpu_key -p <PORT> root@<HOST> '<cmd>'
```

- Keys live in the repo root (`vastai_gpu_key`, gitignored) — never in
  `~/.ssh` (operator preference) and never committed.
- After `attach ssh`, auth may fail for ~20 s while it propagates; retry.
- Container logs without SSH: `vastai logs <ID> --tail 100`.
- When backgrounding processes over ssh (nohup), redirect stdin/stdout and
  expect the channel to close; the remote job keeps running.

## Running a training job

The `pytorch/pytorch` image ships **torch with CUDA preinstalled**
(system conda python). Do NOT `uv sync` or install torch — pyproject pins
the CPU index and `tool.uv.index` overrides even `uv pip install
--index-url`. Use the system python:

```bash
cd /workspace && git clone --depth 1 -b features/gpu-training \
  https://github.com/patrickfrank1/chesspos.git && cd chesspos
pip install -q chess pyarrow huggingface_hub datasets ray
pip install -q -U brotli zstandard httpx2   # stale codecs break hf downloads
```

HF token: transfer via stdin, never argv:

```bash
printf '%s' "$HF_TOKEN" | ssh ... 'cat > /root/.hf_token && chmod 600 /root/.hf_token'
export HF_TOKEN=$(cat /root/.hf_token)   # on the box
hf download patrickfrank1/chess-positions --repo-type dataset --local-dir data/processed
# data lands in data/processed/data/{train,test}
```

Launch (tmux/nohup), GPU flags, no `--compile` on torch < 2.6 (variable
sequence lengths cause constant recompiles):

```bash
nohup env PYTHONPATH=. python3 src/run/train_masked_transformer.py \
  --steps 12000 --batch-size 32 --max-seq-len 2048 \
  --train-dir data/processed/data/train --val-dir data/processed/data/test \
  --num-workers 8 --log-every 50 --eval-every 1000 --eval-batches 20 \
  --save-every 2000 --bf16 \
  --checkpoint-dir models/run_full --log-file models/run_full/train.log \
  > models/run_full/nohup.out 2>&1 &
```

Checkpoints: `last.pt` (full training state, for `--resume`) and `best.pt`
(eval metrics) are written atomically every `--save-every` steps. Sync them
off the box with a background loop:

```bash
hf buckets create chesspos-checkpoints || true
nohup sh -c 'while true; do
  hf buckets sync ./models/run_full hf://buckets/patrickfrank1/chesspos-checkpoints/run_full/ \
    --exclude nohup.out || true; sleep 900; done' &
```

Pull locally: `hf buckets sync hf://buckets/patrickfrank1/chesspos-checkpoints/run_full/ ./models/run_full/`

## Monitoring / debugging

```bash
ssh ... 'tail -5 /workspace/chesspos/models/run_full/train.log'
ssh ... 'nvidia-smi --query-gpu=utilization.gpu,memory.used,power.draw --format=csv,noheader'
ssh ... 'free -m | sed -n 2p; cat /proc/loadavg'
```

Reference profile (4090, bf16, batch 32): ~650k tok/s, GPU 97%, 299 W,
69-71 C, 5.2 GB VRAM, ~12 GB RAM, ~3 of 48 cores busy. If GPU shows 0%
util and idle clocks, the run has died — read the log tail (past cause:
`ValueError: low >= high` from single-segment games, fixed in
`window_dataset.py`).

## Teardown

```bash
vastai destroy instance <ID> -y    # stops ALL billing
```

Destroy only after checkpoints are confirmed in the HF bucket. Checkpoint
meaning (loss/acc interpretation): `docs/training_metrics.md`.
