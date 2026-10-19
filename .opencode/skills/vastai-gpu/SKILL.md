---
name: vastai-gpu
description: >
  Provision and operate vast.ai GPU instances for chesspos training:
  searching offers, creating instances, SSH access, running training jobs,
  monitoring, pushing checkpoints, and teardown. Use when renting a GPU,
  launching/debugging a remote training run, or cleaning up instances.
---

# vast.ai GPU operations

Lessons from real runs (2026-10-10 and 2026-10-19). Full plan context:
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
(system conda python, e.g. torch 2.2.1 / cu121 / python 3.10). Do NOT
`uv sync` or install torch — pyproject pins the CPU index and
`tool.uv.index` overrides even `uv pip install --index-url`. Use the
system python:

```bash
cd /workspace && git clone https://github.com/patrickfrank1/chesspos.git && cd chesspos
pip install -q chess pyarrow huggingface_hub datasets ray einops
pip install -q -U brotli zstandard httpx2   # stale codecs break hf downloads
```

**Deploying code:** prefer git over scp. The box's cached remote ref goes
stale easily, so verify the tip against the local repo, then hard-reset:

```bash
git ls-remote origin refs/heads/<branch>        # on box; compare to local rev
git fetch origin && git reset --hard origin/<branch> && git status --short
```

scp'ing individual files leaves the tree dirty and confuses later resets.

**Flash attention:** the box has **no nvcc** — flash-attn must be a
prebuilt wheel matching the box's torch/CUDA/python/abi exactly. Find the
wheel from the flash-attn GitHub releases page for the box's versions
(e.g. torch 2.2.1 + cu121 →
`flash_attn-2.6.3+cu123torch2.2cxx11abiFALSE-cp310-cp310-linux_x86_64.whl`;
install with `pip install <wheel-url>`). Code guards with
`FLASH_AVAILABLE` in `src/modeling/varlen.py`, so a missing wheel
degrades to SDPA instead of failing.

HF token: transfer via stdin, never argv:

```bash
printf '%s' "$HF_TOKEN" | ssh ... 'cat > /root/.hf_token && chmod 600 /root/.hf_token'
export HF_TOKEN=$(cat /root/.hf_token)   # on the box
```

**Watchdog pattern** (recommended over bare nohup): a shell loop that
auto-resumes from `last.pt` after crashes and exits at the step target —
crash-looping is self-healing and restarts need no manual resume. Keep it
at `/workspace/watchdog_<run>.sh` (template in repo history /
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

Launch it detached (the ssh session may hang until the child exits —
expect it; the job keeps running):

```bash
scp -q -i vastai_gpu_key -P <PORT> watchdog_<run>.sh root@<HOST>:/workspace/
ssh ... 'setsid nohup /workspace/watchdog_<run>.sh > /dev/null 2>&1 < /dev/null & echo started'
```

**Killing jobs — pkill self-match gotcha:** `pkill -f <pattern>` matches
the *shell running your ssh command* whenever the pattern text appears
anywhere in it (including a later `setsid ... watchdog_x.sh` in the same
command — that killed relaunches twice). Rules:

- Use a bracket pattern that cannot match its own literal text:
  `pkill -f 'watchdog_d256l[3].sh'`.
- Never put pkill and the relaunch in the **same** ssh command; use two
  separate calls.

**torch.compile does NOT work on torch 2.2.1 / triton 2.2.0 boxes.** All
of these must be fixed first, and it still fails:

1. `libcuda.so cannot found` →
   `ln -sf /lib/x86_64-linux-gnu/libcuda.so.1 /lib/x86_64-linux-gnu/libcuda.so`
2. `Failed to find C compiler` → `apt-get update && apt-get install -y gcc`
3. `invalid device context` in forked compile workers →
   `export TORCHINDUCTOR_COMPILE_THREADS=1`
4. **Still** `RuntimeError: Triton Error [CUDA]: invalid device context`
   during inductor autotune even in-process — torch 2.2.1/triton 2.2.0 is
   incompatible with the container's driver stack. Fix = upgrade torch on
   the box (rebuild flash-attn wheel accordingly). Until then: no
   `--compile`. Dynamo logs `Unsupported: HigherOrderOperator ... tuple`
   warnings around flash-attn's autograd.Function — benign graph breaks.

**Brotli codec bug:** `TypeError: process() takes no keyword arguments`
during hf_hub download (CDN serves brotli-compressed small files, e.g.
changed `.meta.json` sidecars) = Brotli too old for httpx2. Fix:
`pip install -U Brotli` (≥1.2 accepts `output_buffer_limit`).

Batch sizing: a 25M model at batch 64 used 24/47 GB; a 2.4M model at
batch 256 used 41/47 GB. Scale `--batch-size` to VRAM. Unigram pass
runs instantly from `.meta.json` sidecar histograms when the dataset
shards carry them (falls back to a parallel parquet pass otherwise).

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

**Polling discipline: never `sleep` longer than 30 s per command** — the
operator aborts long sleeps. Prefer short single checks; startup (snapshot
download, unigram pass, compile warmup) takes 1–3 min, so a few spaced
short polls are enough.

```bash
ssh ... 'tail -5 /workspace/chesspos/data/models/<run>/train.log'
ssh ... 'tail -3 /workspace/chesspos/data/models/<run>/watchdog.log'   # crash-loop check
ssh ... 'tail -30 /workspace/chesspos/data/models/<run>/nohup.out'     # tracebacks (grep -v Fetching)
ssh ... 'nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader'
ssh ... 'free -m | sed -n 2p; cat /proc/loadavg'
```

Interpreting startup: `params=... bf16=... compile=...` banner → snapshot
download → unigram pass (expect "from sidecars") → val split load → first
steps. VRAM jumping with GPU at 0% is normal warmup/compile, not a hang.

Reference profiles (4090, bf16): 25M model, batch 64, seq 4096: ~490k
tok/s, 24/47 GB VRAM, GPU ~97%. 2.4M model (d256/3L/4h/ff1024), batch
256: ~2.9M tok/s, 41/47 GB VRAM, GPU 100%. If GPU shows 0% util and idle
clocks, the run has died — read watchdog.log (relaunch loop) and
nohup.out (traceback); past causes: `ValueError: low >= high` from
single-segment games, missing Brotli upgrade, missing gcc, triton
`invalid device context` with `--compile`.

## Teardown

```bash
vastai destroy instance <ID> -y    # stops ALL billing
```

Destroy only after checkpoints are confirmed in the HF bucket. Checkpoint
meaning (loss/acc interpretation): `docs/training_metrics.md`.
