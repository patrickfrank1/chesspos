---
name: vastai-gpu
description: >
  Provision and operate vast.ai GPU boxes for chesspos: search offers, create
  instances, attach SSH keys, install the training environment (system torch,
  pip deps, HF token, flash-attn), deploy code, size the box, and tear down.
  Use when renting a GPU or preparing/repairing a training box. For launching,
  resuming, and monitoring training runs, use the vastai-training skill.
---

# vast.ai GPU and environment setup

Lessons from real runs (2026-10-10, 2026-10-19, 2026-10-22). Full plan
context: `docs/plan_gpu_training_vastai.md`.

Training commands, watchdogs, monitoring, and the retrieval benchmark live in
the **`vastai-training`** skill. This skill only covers the box and its
environment.

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

## Base environment

The `pytorch/pytorch` image ships **torch with CUDA preinstalled**
(system conda python, e.g. torch 2.2.1 / cu121 / python 3.10). Do NOT
`uv sync` or install torch — pyproject pins the CPU index and
`tool.uv.index` overrides even `uv pip install --index-url`. Use the
system python on the box:

```bash
cd /workspace && git clone https://github.com/patrickfrank1/chesspos.git && cd chesspos
pip install -q chess pyarrow huggingface_hub datasets ray einops
pip install -q -U brotli zstandard       # stale codecs break hf downloads
```

On the box, invoke project modules as
`PYTHONPATH=/workspace/chesspos python3 -m src.run.<module>` — not
`uv run`.

### Deploying code

Prefer git over scp. The box's cached remote ref goes stale easily, so
verify the tip against the local repo, then hard-reset:

```bash
git ls-remote origin refs/heads/<branch>        # on box; compare to local rev
git fetch origin && git reset --hard origin/<branch> && git status --short
```

scp'ing individual files leaves the tree dirty and confuses later resets.
Use scp only for throwaway experiments or files not yet pushed.

### Flash attention

The box has **no nvcc** — flash-attn must be a prebuilt wheel matching the
box's torch/CUDA/python/abi exactly. Find the wheel from the flash-attn
GitHub releases page for the box's versions (e.g. torch 2.2.1 + cu121 →
`flash_attn-2.6.3+cu123torch2.2cxx11abiFALSE-cp310-cp310-linux_x86_64.whl`;
install with `pip install <wheel-url>`). Code guards with
`FLASH_AVAILABLE` in `src/modeling/varlen.py`, so a missing wheel
degrades to SDPA instead of failing.

### HuggingFace token

Transfer via stdin, never argv:

```bash
printf '%s' "$HF_TOKEN" | ssh ... 'cat > /root/.hf_token && chmod 600 /root/.hf_token'
export HF_TOKEN=$(cat /root/.hf_token)   # on the box
```

Plain ssh shells have **no** `HF_TOKEN` — only a launcher that exports it
does. Any `hf` CLI call over a bare ssh session (bucket sync, downloads)
fails with `401 Unauthorized ... xet-write-token`; source it from the token
file without printing:

```bash
ssh ... 'export HF_TOKEN=$(cat /root/.hf_token); hf buckets sync ...'
```

### Environment gotchas

- **torch.compile does NOT work on torch 2.2.1 / triton 2.2.0 boxes.**
  These must be fixed first, and it still fails:
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
- **Brotli codec bug:** `TypeError: process() takes no keyword arguments`
  during hf_hub download (CDN serves brotli-compressed small files, e.g.
  changed `.meta.json` sidecars) = Brotli too old for httpx2. Fix:
  `pip install -U Brotli` (>=1.2 accepts `output_buffer_limit`).
- **No nvcc** — see flash attention above.
- **Missing gcc** breaks compilation paths; install it when any build step
  needs a compiler.

## Batch sizing

A 25M model at batch 64 used 24/47 GB; a 2.4M model at batch 256 used
41/47 GB. Scale `--batch-size` to VRAM. The puzzle classifier (frozen
encoder, small head) is much lighter: batch 128 used ~2.7/12 GB on an
RTX 3060.

## Teardown

```bash
vastai destroy instance <ID> -y    # stops ALL billing
```

Destroy only after checkpoints are confirmed in the HF bucket or copied
off the box. For the full "finish a run correctly" sequence (kill trainer,
sync checkpoints, verify, then destroy), see the **`vastai-training`**
skill.
