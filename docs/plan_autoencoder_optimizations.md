# Plan: Autoencoder Training Optimizations

Status: proposed (2026-11-24). Goal: raise the compute/depth ceiling and the
throughput of the attention autoencoder on **one 48 GB GPU**, without losing
`torch.compile` or FlashAttention.

Companion plan: `docs/plan_depth_probing.md` (decides *how much* depth is
worth buying; this plan decides *how to afford it*).

---

## 1. Baseline (measured, 6000 Ada 48 GB)

| item | value |
|---|---|
| model | 20.4M params, `d_model 512`, 6 enc + 3 dec, `d_ff 1024`, latent 512 |
| batch / seq | 128 × 2048 (padded) |
| VRAM | **34.7 GB / 49.1 GB** |
| throughput | **~445k padded tok/s**, ~1.8 steps/s |
| stack | torch 2.8.0+cu128, flash-attn 2.8.3, sm_89, `bf16`, `compile=True` |
| memory model | ~**29.7 MB per block per sample**, ~0.6 GB weights+optimizer+context |

Consequence: at 48 GB the current shape supports ~20 blocks at batch 64 or
~16 blocks at batch 128. Everything below exists to break that trade-off.

---

## 2. P0 — Activation checkpointing (biggest lever, do first)

**Target.** Keep batch 128 and go to 16-24 blocks. Estimated VRAM with
checkpointing (stores block inputs, recomputes block internals):

| config | stored inputs | live block | total |
|---|---:|---:|---:|
| 9 blocks @ b128 (today, no ckpt) | — | — | 34.7 GB (measured) |
| 16 blocks @ b128 | 4.3 GB | 3.8 GB | **~9 GB** |
| 24 blocks @ b128 | 6.4 GB | 3.8 GB | **~11 GB** |
| 32 blocks @ b256 | 17.2 GB | 7.6 GB | **~26 GB** |

**API contract** (PyTorch 2.8/2.14 docs):

- `use_reentrant=False` — **required**. Non-reentrant records the autograd
  graph, supports nested structures, kwargs and **detached tensors**; the
  reentrant variant runs under `no_grad` and *errors* on detached regions.
  Torch 2.9+ raises if the argument is omitted, so pass it explicitly now.
- `early_stop=True` (default, non-reentrant only) — recomputation stops once
  the needed tensors exist. Cheapest variant.
- `preserve_rng_state=True` — **ignored under `torch.compile`; RNG is always
  preserved.** So checkpointing is compatible with our `dynamic=True` compile.
- `context_fn` (non-reentrant only) — returns two context managers for forward
  and recompute. **Use it to force bf16 autocast on recompute**, because
  recompute runs inside backward, outside the training autocast block.

**Why `context_fn` matters here (correctness, not style).**
`src/modeling/varlen.py:varlen_attention` silently falls back to SDPA when the
input dtype is not fp16/bf16. If recompute runs in fp32, the model still trains
but quietly loses the FlashAttention path — a ~2× attention slowdown that looks
like "checkpointing is expensive" rather than "we broke the fast path".

**Sketch.**

```python
from torch.utils.checkpoint import checkpoint

def _amp_ctx():
    return torch.autocast("cuda", dtype=torch.bfloat16)

def _run(self, block, x, cu, max_seqlen):
    if not self.grad_checkpointing:
        return block(x, cu, max_seqlen)
    return checkpoint(
        block, x, cu, max_seqlen,
        use_reentrant=False,
        preserve_rng_state=True,
        context_fn=lambda: (_amp_ctx(), _amp_ctx()),
    )
```

Wrap **encoder blocks** first (the memory hog at depth), then decoder blocks.
Head/CE are not worth checkpointing.

**Flags.** `--grad-checkpointing`, `AttentionAutoencoderConfig.grad_checkpointing`.
Store it in checkpoint metadata so embedders rebuild the same module graph.

**Tests (write first, BDD Given/When/Then):**

1. **Equivalence** — checkpointed vs plain forward/backward, same seed and
   inputs: loss equal within `1e-5` relative; every parameter gradient within
   `1e-4` relative (fp32 CPU test).
2. **Recomputation actually happens** — counter incremented once per block in
   forward and again in backward.
3. **Flash path preserved on recompute** — assert the recompute sees bf16
   (monkeypatch `varlen_attention` to record dtypes; assert never fp32).
4. **Compile stays healthy** — `TORCH_LOGS=recompiles` shows a bounded count
   that does not grow per step. Expect benign `Unsupported:
   HigherOrderOperator … tuple` graph breaks around flash-attn's
   `autograd.Function`; they are warnings, not failures.

**Cost.** ~+25-30 % step time (one extra block forward per block).

**Acceptance.** 16 enc + 4 dec at batch 128 trains under 48 GB at ≥ 300k tok/s.

---

## 3. P1 — Gradient accumulation

**Purpose.** Keep the *effective* batch at 128 (or 256) while running a
micro-batch that fits the deeper model, and keep the gradient noise of the
4→6-layer runs comparable.

**Design.** `--grad-accum-steps N`: `loss / N → backward`, `optimizer.step()`
and `scheduler.step()` only on the boundary, `clip_grad_norm_` only on the
boundary. No `GradScaler` (bf16 autocast does not need it).

**Documented pattern (AMP notes):** divide the loss by the accumulation count,
step once per effective batch, and clip after all micro-batches.

**Gotcha specific to this trainer.** `worker_skip_windows()` fast-forwards the
data stream on resume using `batch_size × steps`; with accumulation the
consumed stream advances by `micro_batch × accum × steps`. The resume math must
use the **effective** batch or resumed runs will silently re-see data.

**Tests.**
1. `accum=4 × micro=32` equals a single `batch=128` step on identical data:
   parameters match within bf16 tolerance, loss identical to 4 dp.
2. Scheduler advances once per optimizer step, not per micro-batch.
3. `--resume` after accumulation reports the correct stream position.

---

## 4. P2 — Other PyTorch tricks (ranked by expected value / risk)

| # | change | expected win | risk | notes |
|---|---|---|---|---|
| 1 | **Chunked cross-entropy** — compute logits+CE in slices over supervised tokens instead of materializing `[tokens, 806]` | memory, not speed: the head is the largest single activation in eval paths | low | decoder already restricted to supervised tokens in the loss path; eval computes all valid |
| 2 | `optimizer.zero_grad(set_to_none=True)` | small, free | none | currently plain `zero_grad()`; `src/run/train_masked_transformer.py:1101` |
| 3 | `torch.backends.cuda.matmul.allow_tf32 = True` | only affects the fp32 PCA/SVD probes | none | training is bf16 already |
| 4 | Tune `--num-workers`, `--prefetch-factor` (persistent/pinned already on) | data-side headroom for deeper models | none | verify `cpu_load` stays < cores |
| 5 | Fusion of head+CE via a fused kernel (Liger / `torch.compile` on the loss fn) | speed + memory | medium | new dependency; only after 1 and 2 |
| 6 | `mode="reduce-overhead"` (CUDA graphs) | up to +15 % | **high** | dynamic varlen lengths cause graph recapture; defer until lengths are bucketed |
| 7 | `torch.cuda.set_per_process_memory_fraction(0.95)` | prevents one process from OOM-killing the watchdog | low | operational guard only |

**Do not do:** switch to fp16 (stability + wheel matching), disable
`dynamic=True` (recompiles per sequence length), or use reentrant checkpointing
(§2).

---

## 5. P3 — Looped encoder (optional, train-time and test-time)

**Idea.** Share one block stack and apply it `n_loops` times, so effective depth
grows without new parameters. Popularised recently as recurrent-depth /
looped transformers; the nearest reference implementation in the wild is
PonderNet-style adaptive step counts rather than weight-shared depth, so expect
to write it (~30 lines) rather than import it.

**Design.**

- `n_loops` in `AttentionAutoencoderConfig`; the encoder forward applies
  `blocks` `n_loops` times.
- **Per-loop input injection**: add the embedded input (or a learned projection
  of it) at the start of each loop — the standard stabiliser.
- **Gated, zero-initialised residual** on the loop output so the first
  iterations reduce to the plain stack.
- Optional per-loop LayerNorm scale (a scalar per iteration) to keep the
  residual stream from drifting.
- Checkpoint **per loop iteration**; with `use_reentrant=False` memory is then
  ~one iteration, which is the whole point.

**Why it is optional now.** Once checkpointing exists (§2), memory is no longer
an argument for loops — depth is affordable directly. Loops remain interesting
for two reasons: (a) parameter sharing is a strong regulariser and our budget is
*tokens*, not parameters; (b) **test-time compute scaling**.

**Test-time mode.** `--inference-loops K` evaluated at `K = 1, 2, 3, 4` on a
fixed pool. If accuracy or `R²_material` rises with `K` at fixed parameters,
that is a capability a fixed-depth model does not have. Record `n_loops` in
checkpoint metadata so `feature/position-vector-index` embedders reproduce it.

**Ablation (matched compute, ~5k steps each).**

| arm | blocks | loops | effective depth |
|---|---:|---:|---:|
| L1 | 12 | 1 | 12 |
| L2 | 6 | 2 | 12 |
| L3 | 4 | 3 | 12 |
| L4 | 6 | 2 (checkpointed) | 12, memory of 6 |

**Failure mode.** Fixed-point collapse: later iterations converge and compute
is wasted. Detect with residual drift / early-exit flatness (see plan A, P1/P2)
and stop the experiment early if `cos(h_loop, h_loop−1) → 1`.

**Instability guard.** If loss spikes with loops ≥ 3, reduce loop LR via
per-loop scaling or drop to 2 loops; report rather than tune away.

---

## 6. Verification matrix (must be green after every step)

| check | how | pass |
|---|---|---|
| unit suite | `uv run pytest -m "not integration"` | all pass |
| lint | `uv run ruff format --check . && uv run ruff check .` | clean |
| compile active | training banner `compile=True`; inductor cache grows | yes |
| no recompile storm | `TORCH_LOGS=recompiles` | count bounded, flat per step |
| FlashAttention active | `FLASH_AVAILABLE` import + dtype assert on recompute | never fp32 |
| numerics | checkpointed vs plain equivalence test | ≤ 1e-5 loss |
| memory | `torch.cuda.max_memory_allocated()` around 30 steps | ≤ 44 GB |
| throughput | `tokens/s` after warmup, second logged interval | ≥ 300k |
| resume | kill + restart, step and data position correct | yes |

---

## 7. Rollout (one PR each, all from `main`)

1. **PR1 — activation checkpointing** (+ config, flags, 4 tests, docs).
   Immediately unlocks `16:4 @ batch 128`.
2. **PR2 — gradient accumulation** (+ effective-batch resume fix, 3 tests).
3. **PR3 — chunked CE, `zero_grad(set_to_none=True)`, dataloader tuning**
   (memory/latency only; no numerics change).
4. **PR4 — looped encoder** (optional; only if plan A's P1-P3 leave the
   depth-vs-loops question open).

Each PR runs the verification matrix on a fresh box before merge; PR1 and PR2
also get a real 200-step smoke run at `16:4 @ batch 128` to confirm the
throughput and memory estimates on hardware rather than on paper.

---

## 8. Success metrics

- `16:4` (4:1, 20 blocks, 43.5M params) trains at batch 128 under 48 GB.
- ≥ 300k padded tok/s at that shape (i.e. checkpointing overhead < 35 %).
- Zero recompilation growth across 1k steps; FlashAttention used on both passes.
- Equivalence tests prove checkpointing and accumulation are numerically inert.
- If P3 lands: a measured accuracy-vs-inference-loop curve for the looped model.
