# Plan: GPU Training on vast.ai

Status: planning (2026-10-10). Goal: train the masked stream transformer
(`src/run/train_masked_transformer.py`, 3.43M params) on the full dataset
(~750M train tokens / 368k games) — CPU throughput (~1.7–3.2k tok/s) makes
one epoch a ~5-day job, which is not viable.

---

## 1. Ray for training? No.

- **Single GPU, single node.** Ray Train buys distributed-data-parallel
  orchestration across many devices/nodes. One 4090 saturates a 3.4M-param
  model; DDP would add nothing but moving parts (worker startup, env setup
  over SSH, GPU util monitoring).
- **Data fits in RAM.** The full parquet set is 538 MB on disk, ~2–3 GB as
  int16 arrays after load. Ray Data's streaming/caching machinery is
  unnecessary; `torch.utils.data.DataLoader` with worker processes is
  enough. (If the dataset ever outgrows RAM, `ray.data.iter_torch_batches`
  is the escape hatch — keep in mind, don't build it now.)
- **Ray stays ETL-only** (PGN → parquet on this workstation). Training
  reads parquet + HF Hub directly.
- Revisit Ray only if/when: multi-GPU nodes, or dataset >> RAM.

## 2. Provider: vast.ai (recommended) vs runpod

Both are fine; vast.ai is cheaper for raw GPU-hours and has better CLI/SDK
automation for the full lifecycle. Both have APIs — full automation is
possible.

| | vast.ai | runpod |
|---|---|---|
| Pricing (RTX 4090 class) | ~$0.25–0.45/h | ~$0.35–0.70/h |
| API | GraphQL + Python SDK + CLI (`pip install vastai`) | REST + Python SDK (`runpod`) |
| Run commands on box | `vast.execute()` in SDK, or plain SSH | SDK only manages pods; commands via SSH separately |
| Fit for "rent box, run long training, destroy" | excellent | fine |

Recommended: **vast.ai**, automated via CLI/SDK, one instance at a time.

### 2.1 vast.ai automation surface

```bash
pip install vastai                    # CLI + SDK (vastai, vastai_sdk)
vastai set api-key $VASTAI_API_KEY    # key from console.vast.ai/manage-keys/
vastai show user                      # verify auth + balance

# find a GPU (sorted by price; gpu_ram/cpu_ram filtering happens client-side
# because vast.ai's server-side >= comparison on these fields is broken)
vastai search offers 'gpu_name=RTX_4090 num_gpus=1 verified=true rentable=true \
  disk_space>=20 cpu_cores>=4 reliability>0.98' -o 'dph_total'

vastai create instance <OFFER_ID> --image pytorch/pytorch --disk 40 --ssh --direct
vastai show instance <ID>             # poll until actual_status == running
vastai ssh-url <ID>                   # -> ssh -p PORT root@HOST
vastai destroy instance <ID> -y       # stops billing
```

Python equivalent (`from vastai import VastAI`): `search_offers`,
`create_instance`, `execute(id, cmd)`, `ssh_url`, `destroy_instance` —
`scripts/provision_vast.py` (written 2026-10-10) covers search →
**confirmation prompt** → create → wait for running → print ssh-url,
plus `--instance-id` (reconnect) and `--destroy` (with confirmation). It
reads `VASTAI_API_KEY` from the environment and must never log, print, or
persist it (see Secrets Handling in AGENTS.md); the key is passed to the
vastai CLI per invocation via `--api-key`, so `vastai set api-key` is not
needed. Bootstrap/launch stays in `scripts/gpu_bootstrap.sh` (run over
SSH).

### 2.2 runpod alternative

```python
import runpod
runpod.api_key = ...
pod = runpod.create_pod("chesspos", "runpod/pytorch", "NVIDIA GeForce RTX 4090")
runpod.terminate_pod(pod.id)
```
Command execution is via SSH on the returned pod address — same bootstrap
script applies.

## 3. Instance sizing

Requirements profiled on a real run (2026-10-10, batch 32 × 2048 tokens, bf16,
8 workers): GPU util 97%, VRAM 5.2 GB, system RAM ~12 GB, ~3 CPU cores busy,
~650k tokens/s (~20 min/epoch). `scripts/provision_vast.py` defaults to the
minimum column; `--comfort` doubles all minimums.

| Resource | Minimum | Comfort (2× min) |
|---|---|---|
| GPU VRAM | 8 GB | 16 GB |
| RAM | 16 GB (dataset load peaks ~9 GB) | 32 GB |
| Disk | 20 GB (image torch; data 0.6 GB, checkpoints ~0.2 GB) | 40 GB |
| CPU | 4 cores | 8 cores |

Throughput measured on a 4090 with bf16: ~650k tok/s → **~20 min per epoch**
(750M tokens) vs 5 days on CPU. Driver: image-bundled CUDA works on any
driver ≥ 535; the provisioner filters client-side for that.

## 4. Pre-work (do locally, before renting anything)

1. **Upload the dataset to HF** (skips re-running Ray ETL remotely; raw
   PGNs are 4.2 GB vs 538 MB parquet) — done 2026-10-10, 238 train + 194
   test shards live in `patrickfrank1/chess-positions`:
   ```bash
   hf upload patrickfrank1/chess-positions data/processed/train data/train --repo-type dataset
   hf upload patrickfrank1/chess-positions data/processed/test data/test --repo-type dataset
   ```
   Files land under `data/{train,test}/…` in the repo; after
   `hf download --local-dir data/processed` the training dirs are
   `data/processed/data/train` and `data/processed/data/test`.
2. **Training script hardening** (`src/run/train_masked_transformer.py`):
   - `--resume PATH`: restore model/optimizer/scheduler/step from a
     checkpoint (needed for spot-instance interruptions).
   - periodic checkpoints (`--save-every N`), not just best-val: save
     `last.pt` (full training state) + `best.pt` (eval metrics only).
   - GPU performance flags: `torch.set_float32_matmul_precision("high")`,
     autocast bf16 + `torch.compile` (behind `--compile` flag, off by
     default so CPU runs are unaffected).
   - log to `--log-file` (stdout already flushes).
3. **Bootstrap script** (`scripts/gpu_bootstrap.sh`) — runs ON the vast.ai
   box (base image `pytorch/pytorch` already has CUDA + python):
   ```bash
   # 1. clone repo (needs a GitHub token for a private repo, or rsync src/ over)
   # 2. curl -LsSf https://astral.sh/uv/install.sh | sh
   # 3. uv sync                      # installs CPU torch from lock — then:
   # 4. uv pip install "torch==2.14.1" --force-reinstall
   #    (the PyPI linux wheel of torch 2.14.1 IS the CUDA 13 build — it pulls
   #     nvidia-*-cu13 deps; no separate cu126 index needed. The lock pins
   #     torch 2.14.1+cpu, so bypass `uv run` (it would re-sync CPU torch)
   #     and invoke .venv/bin/python directly. CUDA 13 needs driver >= 580.)
   # 5. hf auth login --token $HF_TOKEN   (or: HF_TOKEN env var)
   # 6. hf download patrickfrank1/chess-position-streams --repo-type dataset --local-dir data/processed
   # 7. PYTHONPATH=. uv run python src/run/train_masked_transformer.py --steps 20 --max-seq-len 2048 --batch-size 32   # smoke test
   # 8. nohup uv run python src/run/train_masked_transformer.py ... > train.log 2>&1 &
   ```
4. **Secrets on the box**: pass as environment variables at create time
   (`HF_TOKEN`, optional `GITHUB_TOKEN`) — never bake into images/scripts
   committed to the repo.

## 5. Pushing checkpoints to HF (from the GPU box)

Two options; **buckets are the better fit** for mutable checkpoints (no
git history, deduplicated, rsync-style sync). Decision (2026-10-10):
**Option A** — `hf buckets sync` runs on the GPU box every 15 min via the
bootstrap script; a model repo for the final artifact comes later.

```bash
# Option A (chosen): HF bucket patrickfrank1/chesspos-checkpoints
hf buckets create chesspos-checkpoints
hf buckets sync ./models/run_full hf://buckets/patrickfrank1/chesspos-checkpoints/run_full/
# re-run periodically; only changed files upload
# (the bootstrap script loops this every SYNC_INTERVAL seconds, default 900)

# Option B: model repo (versioned commits)
hf upload patrickfrank1/chesspos-masked-stream-transformer ./models/run_full/best.pt checkpoints/best.pt
```

Note: `hf upload --every=10` (minutes) can tail the checkpoint dir during
training as a poor-man's autosync.

## 6. Testing checkpoints locally (after the run)

```bash
hf buckets sync hf://buckets/patrickfrank1/chesspos-checkpoints/run_full/ ./models/run_full/
```

Then:

1. **Load & smoke-check**: reconstruct
   `MaskedStreamTransformer(TransformerConfig(**ckpt["model_config"]))`,
   load state dict, confirm val loss/acc recorded in the checkpoint
   reproduce when re-evaluating on local val data.
2. **pytest** (23 tests) against the checkpoint config — architecture
   invariants (padding invariance, masking modes) still hold.
3. **Embedding sanity checks** (the actual success criterion — to be
   written as `src/run/evaluate_embeddings.py`):
   - kNN over val-set position embeddings: nearest neighbours of a position
     should be positions reachable by one legal move and transpositions of
     it.
   - same-position-different-move-order pairs should embed nearly
     identically.
   - UMAP/PCA scatter colored by game phase / material balance
     (matplotlib + umap-learn are already dev dependencies).

## 7. Runbook (condensed)

1. Local: upload dataset to HF, add resume/compile to trainer, write
   bootstrap script, run pytest + ruff.
2. `vastai search offers` → `create instance` (4090, 40 GB disk).
3. Wait for running → `vastai ssh-url` → run bootstrap script (sets up
   repo, CUDA torch, downloads data) → smoke test 20 steps.
4. Launch training in `tmux`/`nohup` → `hf buckets sync` checkpoints
   periodically.
5. Monitor via SSH (`tail train.log`); on completion push final checkpoint.
6. Local: download checkpoint, verify, run embedding evaluations.
7. `vastai destroy instance <ID> -y`.
