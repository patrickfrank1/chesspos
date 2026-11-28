# Plan: Autoencoder Training Optimizations

Status: proposed (2026-11-24; DCA added 2026-11-28). Goal: raise the
compute/depth ceiling and the throughput of the attention autoencoder on **one
48 GB GPU**, without losing `torch.compile` or FlashAttention. P0-P3 lower the
cost of depth; P4 (DeepCrossAttention) is the one lever that aims to raise the
quality obtained *per unit* of depth.

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

**DCA interaction.** If P4 lands, checkpoint the block **loop** rather than
each block; a per-block checkpoint would save the depth stack as a block input
and multiply the stored activations by its slot count. See §6.2.

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

## 6. P4 — DeepCrossAttention (DCA): learned residual routing (optional, architecture)

**Status.** Proposed; measure before adopting. DCA is orthogonal to P0-P3: P0
and P1 make depth affordable, DCA aims to make the depth you buy more
efficient. Do not start the DCA ablation until activation checkpointing (P0) is
stable, because the 16-block arm depends on it.

**Source.** Heddes, Javanmard, Axiotis, Fu, Bateni, Mirrokni,
*DeepCrossAttention: Supercharging Transformer Residual Connections*
(arXiv:2502.06785). A compact PyTorch reference is
`lucidrains/deep-cross-attention`. The design below adapts the method to the
packed varlen autoencoder; it is deliberately not a copy of the decoder-only LM
code.

### 6.1 Why this model, why this plan

The residual stream is the running sum of every previous block output. A plain
sum treats all depths as equally important, so a few informative layers get
diluted by the rest. The paper's motivating experiment is stark: a 10-layer
ResNet whose layers are rank 3 (full rank only because of the residual sum)
cannot learn the identity after 10⁵ examples, while the same stack with
learnable residual weights gets there with ~100× fewer. DCA generalises this to
input-dependent weights and, in the transformer, lets the queries, keys and
values of a block be composed from *different* depths.

Reported results (LM1B/C4, 6-42 layers, TPU):

- lower perplexity for a given parameter budget and training time;
- the same quality up to **3× faster**;
- **~+0.08M parameters** on a 49.65M model (0.16%);
- fewer loss spikes than the baseline transformer;
- pre-trained models can be retrofitted, because the method reduces to a
  standard residual network at initialisation.

Relevance here: our autoencoder squeezes an entire packed window into one
512-d vector before reconstruction. Any token-level information lost to depth
dilution cannot be recovered by the decoder or by a downstream probe. The
encoder is also the part we are trying to deepen. DCA is the only proposal in
this plan that raises quality **per unit of depth**; the rest only lower the
price of depth.

Keep expectations calibrated. The evidence is decoder-only language modelling
on TPUs. The paper's width-scaling table shows the gain shrinking as width
grows (−3.09 PPL at width 64, −0.23 at width 1024). Our `d_model=512` sits in
the middle, and a bidirectional encoder with a global bottleneck may already
dilute less than a causal LM. Treat DCA as an A/B arm with a stop rule, not as
a default.

The accompanying theory gives the same rule: the gain over a plain ResNet is
largest when the layers' collective rank is small relative to the width, and
it shrinks as that ratio grows. Read it as "expect more from routing at depth
than from widening", and do not generalise a negative result from one width.

### 6.2 Method in this codebase

Notation, per token (the varlen stream runs every token independently):

- `x0` — embedded input (encoder), or `from_latent(latent) + positions`
  (decoder).
- `inc_t` — the increment emitted by block `t` (attention output plus FF
  output, **without** the block input).
- `stream_t = x0 + Σ_{i<t} inc_i` — the current residual stream. This is what
  `VarlenTransformerLayer` returns today and what `ln_final` sees.

For the standard block, `inc_t = attn_out + FF(LN(stream_t + attn_out))` and
`stream_{t+1} = stream_t + inc_t`. DCA changes how attention inputs are
composed, not the attention kernel, the FF, or the final norm.

**Stack of previous outputs.** Two modes, both satisfying
`Σ G_t = stream_t`:

```
full:          G_t = [x0, inc_0, ..., inc_{t-1}]
first+last-k:  G_t = [x0, collapse_t, inc_{t-1}, ..., inc_{t-k}]
               collapse_t = Σ_{i<t-k} inc_i
```

The paper collapses the intermediate layers by addition. The `lucidrains`
implementation drops them instead, which breaks `Σ G_t = stream_t`; we follow
the paper so the model is exactly the current one at initialisation.

**GRN-v3, per token.** With `G ∈ R^{S×d}`, `W_s ∈ R^{3×d}`,
`b ∈ R^{3×S×d}`:

```
scores            = ReLU(G W_sᵀ)                 # [S, 3]
weights[o, s]     = b[o, s] + scores[s, o]       # broadcast over d
(q_in, k_in, v_in) = Σ_s G[s] · weights[o, s]    # o = 0, 1, 2
```

`b` is initialised to ones; `W_s` is initialised to `N(0, 0.01)` (see the
gotcha below). `collapse` is zero and missing recent slots are zero-padded
early in the stack. One module with three outputs is mathematically the three
independent GRN instances the paper describes (separate bias rows, one pass
over the stack); the `grn-v1`/`grn-v3` ablations use one output.

**DCA block.**

```
q = W_q · LN_q(q_in);  k = W_k · LN_k(k_in);  v = W_v · LN_v(v_in)
a = varlen_attention(q, k, v, cu, max_seqlen)          # unchanged kernel
inc = attn_out(a) + FF(LN_ff(q_in + attn_out(a)))       # q_in is the FF residual
stream += inc;  G.append(inc)
```

- Reuse the existing fused `qkv` weight (and bias) in three row slices
  (`W_q`, `W_k`, `W_v`). A pre-trained baseline block then maps onto a DCA
  block without reinitialising attention.
- Use three `LayerNorm`s, one per composed input, each initialised from the
  pre-trained `ln_attn`. They are identical at step 0 and free to diverge.
- The block emits `inc`, **not** `stream`. The loop owns `stream`, and
  `ln_final(stream)` is unchanged. This preserves the current model at init
  and keeps the bottleneck input semantics.
- Do **not** add `stream` to the block output. Attention and FF already see it
  through `q_in`; adding it back double-counts depth. The reference
  implementation returns `ff_out + attn_out` for exactly this reason.

**Gotcha: zero `W_s` under ReLU is a dead branch.** The paper says the
input-dependent weights are initialised at zero, but `ReLU'(0)=0`, so
`∂L/∂W_s = 0` and `W_s` never leaves zero. Use the reference implementation's
small normal init (`std=1e-2`) and accept an O(1e-2) deviation from the
baseline at step 0, or swap `ReLU` for `GELU`/`SiLU` if exact zero-init is
required. Add a test that `W_s.grad` is non-zero after one step; do not "fix"
this by zero-initialising both `W_s` and `b`.

**Memory: never materialise the stack.** A `[S, T, d]` copy at `T=262k`,
`d=512`, `S=4`, bf16 is ~1.07 GB; one per block is ~17 GB at 16 blocks and
undoes the P0 win. Pass the slots as a **list of tensors** and accumulate
`out[o] += slot * (b[o, i] + score[:, o])` slot by slot inside the GRN. The only
extra per-block activation is the `[3, S, T]` score tensor (~6 MB) plus the
three `[T, d]` composed inputs (~0.8 GB, the same order as the existing fused
QKV activation).

**Checkpointing interaction.** With per-block `checkpoint`, the stack becomes a
saved block input. Two safe options:

1. **Loop-level checkpoint (preferred for DCA).** One `checkpoint` around the
   whole `for block` loop. Only `x0` is stored; the stack is rebuilt on
   recompute. Memory is ~one live block, independent of `S` and depth.
2. **Per-block checkpoint with stream references.** Pass
   `(stream_t, stream_{t-1}, …, stream_{t-k})` and rebuild
   `inc_i = stream_{i+1} - stream_i`, `collapse_t = stream_{t-k} - x0` inside
   the checkpointed callable. The references share storage, so unique stored
   activations remain one `[T, d]` stream per block — the same as standard P0.

Never call `torch.stack(slots)` before `checkpoint`: that turns shared
references into per-block copies.

**Static shapes for `torch.compile`.** With first+last-k, keep the slot count
`S = k + 2` for every block: zero-pad missing recent slots and use
`collapse = 0` early. A zero slot contributes nothing for any finite `b`, so
no mask is needed and its bias gradient is zero. In full-stack mode each block
has its own fixed `S = t + 1`; the depth dimension is Python-static, so this
does not create recompiles.

**Config and checkpoint metadata.** Add to `AttentionAutoencoderConfig` and
persist it with the model config (the embedder must rebuild the same graph):

```python
dca: bool = False
dca_locations: str = "encoder"   # "encoder" | "decoder" | "both"
dca_past_layers_k: int = 2       # 0 = full stack
dca_variant: str = "dca"         # "dca" | "grn-v1" | "grn-v3"
```

`dca` uses three GRN-v3 outputs as q/k/v. `grn-v1` and `grn-v3` are the
paper's single-GRN ablations: one composed output feeds the existing fused
`qkv`, with no q/k/v split. Validate the enums and
`dca_past_layers_k >= 0`. Old checkpoints load with `dca=False`; a DCA
checkpoint must fail loudly if loaded into a non-DCA config rather than
silently dropping the new tensors.

**Parameter budget** (`d_model=512`, `k=2`, `S=4`):

| item | per block | 20 blocks (16:4) | full stack, 20 blocks |
|---|---:|---:|---:|
| `W_s` (`d→3`) | 1,536 | 30,720 | 30,720 |
| GRN bias (`3·S·d`) | 6,144 | 122,880 | ~0.23M |
| extra LNs (`2·2d`) | 2,048 | 40,960 | 40,960 |
| **total** | **~9.7k** | **~0.19M (+0.4%)** | **~0.30M (+0.7%)** |

On the 43.5M target both modes stay under 1%; the paper's 6-layer model adds
0.08M. The cost is not parameters but step time (~30% in the paper) and the
stack bookkeeping.

### 6.3 Tests (write first, BDD Given/When/Then)

1. **Block equivalence at init.** Given a `VarlenTransformerLayer` and a
   `DCALayer` with `b=1`, `W_s=0`, LNs copied from `ln_attn` and q/k/v slices
   copied from `qkv`; when both run on the same `stream`; then `inc` and
   `stream + inc` match within `1e-6` in fp32 on CPU.
2. **Stack invariant.** For `k=0,1,2` and every depth `t`, `Σ G_t == stream_t`
   in fp32, including early-depth zero padding and `collapse`. This is the
   test that catches all interval bookkeeping bugs.
3. **Increment semantics.** The DCA loop's final `stream` and
   `ln_final(stream)` match the standard stack's final output; catches the
   "add the input back" bug.
4. **Full-model retrofit.** Copy a baseline checkpoint's weights, add DCA, and
   compare logits/latent on a fixed batch at step 0 (fp32, eval). Measure and
   pin the tolerance from the chosen `W_s` init rather than assuming a fixed
   output delta. Reverse direction: a DCA checkpoint loaded with `dca=False`
   raises, not silently partial-loads.
5. **Gradients reach the routing.** After one backward, every present slot's
   bias row has non-zero grad; padded slots have exactly zero; `W_s.grad` is
   non-zero (dead-ReLU guard).
6. **No materialisation.** Unit-test that the GRN forward accepts a list and
   never calls `torch.stack` on the slots; in the CUDA smoke run, assert peak
   allocated memory does not scale as `S × T × d` per block.
7. **Checkpoint + compile.** DCA with `grad_checkpointing` and `compile=True`
   matches the plain path within bf16 tolerance; recompile count is flat over
   50 steps; FlashAttention sees bf16 on recompute.
8. **Varlen isolation.** Changing one row's tokens leaves other rows' latents
   and gradients unchanged (reuses the existing varlen contract).
9. **Config round-trip.** Save/load metadata; `dca`, `dca_locations`,
   `dca_past_layers_k` and `dca_variant` reproduce the same graph and parameter
   count.
10. **Masking contract.** All slots derive from the masked input; changing
    `targets` must not change the latent (only the loss).

### 6.4 Ablation (short, matched compute)

Hardware: 6000 Ada 48 GB, P0+P1 landed, effective batch 128, bf16, compile,
flash. Use the current packed-window corpus and one fixed validation pool so
`val_loss` is comparable across arms.

| arm | blocks | routing | stack | purpose |
|---|---|---:|---:|---|
| A0 | 9 | standard | — | current baseline |
| A1 | 9 | GRN-v1 only | 2 | cheapest DenseFormer control; paper's largest single gain |
| A2 | 9 | DCA | 2 | DCA at the affordable depth |
| A3 | 16 | standard | — | depth bought by P0 |
| A4 | 16 | DCA | 2 | main candidate |
| A5 | 16 | DCA | 0 (full) | cost of completeness |
| A6 | 16 | DCA, decoder only | 2 | where the gain lives |

- 5k steps per arm, same seed and data order. Report `val_loss` at fixed
  **tokens** and at fixed **wall-clock**; DCA costs ~30% step time, so the two
  rankings can differ.
- Secondary: `test_r2_material` with 95% CI on the fixed 4,096-position pool,
  phase probe, `mean_top10_cos`, empty-square recall, max grad norm, and
  loss-spike count (steps with loss > 2× the trailing median).
- `k` sweep on A4 at 2k steps: `k = 1, 2, 4, full`. Pick the smallest `k`
  within noise; the paper reports `k=2` at 48% lower inference latency than
  `k=24`.
- Retrofit check: add DCA to the current 20M checkpoint and continue 2k steps;
  step 0 must not regress.
- Stop rule: if A4 does not beat A3 at matched wall-clock and does not beat A2
  at matched tokens, record the negative result and stop. If A1 captures most
  of A4's gain, ship A1 (one GRN, no q/k/v split) instead.

### 6.5 Risks and interactions

| risk | mitigation |
|---|---|
| dead input-dependent branch (`ReLU(0)`) | non-zero `W_s` init; gradient test |
| stack bookkeeping silently changes the model | sum invariant + retrofit tests; keep the loop in one place |
| per-block stack copies blow memory | list-based GRN; loop-level checkpoint; measure peak VRAM |
| compile recompiles from varying stack depth | static `S=k+2`; zero pad; per-block module for full mode |
| checkpoint + tuple/list inputs graph-break | acceptable warnings; monitor `TORCH_LOGS=recompiles`; loop-level first |
| bidirectional autoencoder gains smaller than LM | stop rule; report per arm; do not assume transfer |
| bottleneck sees a different residual distribution | keep `ln_final`; check step-0 loss; optional final-GRN arm |
| DCA + P3 loops | reset the stack at each loop iteration; do not mix iteration outputs |
| "identical at init" claim | true only at `W_s=0`; with small init it is approximate — report the deviation |

Open question worth one extra run after A4: a **final GRN** over the encoder
stack before the bottleneck (the paper's `final_grn`). It is one extra `d→1`
GRN and lets the pooled query read any depth directly. At init it is the
identity, so it can be A/B'd on top of A4 rather than treated as a new
architecture.

### 6.6 Acceptance for P4

- Baseline and DCA arms differ only by config flags; no per-arm code forks.
- A4 reaches A3's final `val_loss` in **≤ 0.67× the tokens and ≤ 0.9× the
  wall-clock**, or improves final `val_loss` by ≥ 2% relative at matched
  wall-clock.
- `test_r2_material` improves outside the 95% CI, not just reconstruction
  loss.
- Peak VRAM at `16:4 @ b128` stays under 44 GB with checkpointing; step time
  is ≤ 1.35× the same-depth baseline; recompile count stays flat.
- No loss spikes over 5k steps; max grad norm stays inside the baseline band.
- If the acceptance fails, defaults stay `dca=False` and the experiment report
  records why.

---

## 7. Verification matrix (must be green after every step)

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
| DCA init equivalence | §6.3 tests 1 and 3 | ≤ 1e-6 (fp32) |
| DCA stack invariant | §6.3 test 2 | exact in fp32 |
| DCA param overhead | count at `d=512, k=2` | < 1% |
| DCA memory | peak VRAM with `S=4`, 16 blocks | no `S×` scaling; ≤ 44 GB |
| DCA throughput | `tokens/s` at `16:4 @ b128` | ≥ 220k (≤ 1.35× step time) |

---

## 8. Rollout (one PR each, all from `main`)

1. **PR1 — activation checkpointing** (+ config, flags, 4 tests, docs).
   Immediately unlocks `16:4 @ batch 128`.
2. **PR2 — gradient accumulation** (+ effective-batch resume fix, 3 tests).
3. **PR3 — chunked CE, `zero_grad(set_to_none=True)`, dataloader tuning**
   (memory/latency only; no numerics change).
4. **PR4 — DeepCrossAttention** (GRN-v3, DCA block, config flags, 10 tests,
   loop-level checkpoint integration, ablation report). Ships disabled by
   default; promote only if §6.6 passes.
5. **PR5 — looped encoder** (optional; only if plan A's P1-P3 leave the
   depth-vs-loops question open). If both P3 and P4 land, the looped encoder
   resets the DCA stack at each iteration.

Each PR runs the verification matrix on a fresh box before merge; PR1 and PR2
also get a real 200-step smoke run at `16:4 @ batch 128` to confirm the
throughput and memory estimates on hardware rather than on paper.

---

## 9. Success metrics

- `16:4` (4:1, 20 blocks, 43.5M params) trains at batch 128 under 48 GB.
- ≥ 300k padded tok/s at that shape (i.e. checkpointing overhead < 35 %).
- Zero recompilation growth across 1k steps; FlashAttention used on both passes.
- Equivalence tests prove checkpointing and accumulation are numerically inert.
- If P3 lands: a measured accuracy-vs-inference-loop curve for the looped model.
- If P4 lands: A4 reaches A3's `val_loss` in ≤ 0.67× tokens and ≤ 0.9×
  wall-clock, `test_r2_material` improves outside the 95% CI, and the 16:4
  shape still fits under 44 GB at ≥ 220k tok/s. Otherwise the negative result
  is recorded and `dca` stays off.
