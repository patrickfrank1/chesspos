# Plan: Autoencoder Training Optimizations

Status: proposed (2026-11-24; DCA added 2026-11-28; priorities revised
2026-11-28 after the depth probes). Goal: raise the compute ceiling and the
throughput of the attention autoencoder on **one 48 GB GPU**, without losing
`torch.compile` or FlashAttention. P0-P3 lower the cost of compute; P4
(DeepCrossAttention) is the one lever that aims to raise the quality obtained
*per unit* of depth.

Companion plan: `docs/plan_depth_probing.md` (decides *how much* depth is
worth buying; this plan decides *how to afford it*). Its first pass ran on
2026-11-28 and the answer is in §1.2-1.5: **depth pays up to about 6-8 encoder
blocks per token and then flattens**, so freed memory should buy tokens before
it buys layers. Full tables and artifacts:
`docs/experiments/depth_probing_2026-11-28.md` (lands with the probe scripts
from `feature/depth-probing-plan`); §1.2-1.5 below is self-contained.

---

## 1. Baseline and measured depth value (6000 Ada 48 GB)

### 1.1 Current shape

| item | value |
|---|---|
| model | 20.4M params, `d_model 512`, 6 enc + 3 dec, `d_ff 1024`, latent 512 |
| batch / seq | 128 × 2048 (padded) |
| VRAM | **34.7 GB / 49.1 GB** |
| throughput | **~445k padded tok/s**, ~1.8 steps/s |
| stack | torch 2.8.0+cu128, flash-attn 2.8.3, sm_89, `bf16`, `compile=True` |
| memory model | ~**29.7 MB per block per sample**, ~0.6 GB weights+optimizer+context |

Consequence: at 48 GB the current shape supports ~20 blocks at batch 64 or
~16 blocks at batch 128. But §1.3 says only 6-8 encoder blocks are worth
buying per token, so the trade-off worth breaking is **batch × steps**, not
depth. Everything below is retargeted at that.

### 1.2 Depth is not saturated inside a converged model

From `best_step96000` of `run_ae_d512l6h8_100k` (6 enc + 3 dec), 128 paired
windows from three fresh `test/` shards (8 192 games, board masking, seed 17):

| probe | result |
|---|---|
| residual drift | no near-identity block; `rel_delta` 0.51-0.65 per block, the last block largest (0.648) |
| early exit | 0.689 → 0.728 → 0.803 → 0.843 → 0.884 → **0.940**; `k*` = 6, the last block alone +5.6 pt |
| latent effective rank (99 %) | 74 → 87 → 107 → 124 → 142 → **170** with depth |
| layer pruning | every block ≥ 2.0 pt; the last block costs the most (**5.64 pt**) |

Read: do not shrink to 4 encoder blocks to save compute. At convergence the
final block is the single most valuable one to keep.

### 1.3 At a fixed token budget, the marginal arm past 6-8 blocks is calibration

Four matched arms, 5 000 steps at batch 64, only `--n-layers` differs
(`p_rect 0.34 / p_skip_gram 0.66`, balanced-v1, lr 2e-4, seed 17, compiled
bf16):

| arm | enc:dec | params | val_loss @5k | val_acc @5k | early-exit acc @full depth | rank 99 % | VRAM @ b64 | step time @ b64 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| D4 | 4:3 | 16.19M | 0.6597 | 0.8421 | 0.8499 | 91 | 15.9 GB | 0.30 s |
| D6 | 6:3 | 20.40M | 0.6202 | 0.8459 | 0.8608 | 99 | 19.2 GB | 0.35 s |
| D8 | 8:3 | 24.60M | 0.6237 | 0.8483 | 0.8596 | 101 | 22.5 GB | 0.39 s |
| D12 | 12:3 | 33.01M | 0.5776 | 0.8494 | 0.8630 | 99 | 29.1 GB | 0.51 s |

- 4 → 6 blocks: −0.040 val loss, +1.1 pt probe accuracy, +8 rank. Real.
- 6 → 8 blocks: 0.004 **worse** val loss and +0.0 pt top-1 on the probe. Flat
  at the 0.3 pt noise floor.
- 8 → 12 blocks: −0.046 val loss but only +0.3 pt on the paired probe (+0.1 pt
  on the training val set) and rank −2. The extra depth sharpens confidence; it
  does not add decodable content.
- Material `R²` on the final latent still rises with arm depth
  (0.18 / 0.22 / 0.28 / 0.39), so deeper arms do buy attribute information
  even where reconstruction is flat.
- VRAM grows ~1.65 GB per encoder block per 64 samples, consistent with the
  29.7 MB/block/sample model above; step time grows sublinearly (0.30 → 0.51 s
  from 4 to 12 encoder blocks).
- Every arm held the **decoder at 3 blocks**, so this sweep says nothing about
  decoder depth; it is the largest untested axis (§1.6).

### 1.4 The last block earns its keep late

Pruning the last block costs **5.64 pt at 96 000 steps** but **0.51 pt at
5 000 steps** in the same 6-layer recipe. Short-horizon arms understate the
final block, so the 6-8 block knee is a fixed-token result at ~0.37 B tokens
and must be re-confirmed at the target horizon before it is treated as final
(the verification matrix in §7 carries the check).

### 1.5 What this changes here

| item | before | after |
|---|---|---|
| P0 target shape | `16:4 @ b128` (20 blocks) | **`8:4 @ b256`** (candidate; `8:3` until the decoder-depth arm lands) |
| where freed memory goes | more depth | **more tokens** (batch first, then steps) |
| P3 loops | optional capacity lever | test-time-compute / regulariser question, deferred |
| P4 DCA | main arm at `16:4` | main arm at **8 enc + 4 dec**; 16-block arms secondary; encoder and decoder routed independently |
| depth acceptance | "does depth still help?" | known to be flat past ~8 enc blocks per token; judge new ideas on wall-clock and tokens |

### 1.6 Open questions the probes did not answer

Ordered by expected value per GPU-hour; each is a small matched-token arm on
the same 5 000-step protocol.

1. **Decoder depth.** Every arm held `n_decoder_layers=3`, and the decoder is
   the only consumer of the latent: it rebuilds ~1 150 tokens from one 512-d
   vector, so its capacity may bind where encoder depth does not. Run
   `8:3` / `8:4` / `6:4` (and `12:4` if time allows). The bottleneck makes
   the decoder's job harder than the encoder's, so a gain here is plausible
   even though encoder depth saturated.
2. **Iso-FLOP, not iso-token.** The sweep fixed steps × batch, so it answers
   "does depth help per token". Measured step times (0.30 / 0.35 / 0.39 /
   0.51 s) turn that into a wall-clock comparison: `D4 @ ~5 800`, `D6 @ 5 000`,
   `D8 @ 5 000`, `D12 @ 5 000` steps cost the same. If the D4-at-5.8k arm
   matches D6-at-5k, depth is not paying per FLOP; if it loses, it is.
3. **Masking mixture.** The probes used board masking only. Training is
   34 % rectangle / 66 % skip-gram, which is a harder inference problem and
   could move the knee. Re-run P2 with the training mixture before locking
   the shape (needs a rect/skip-gram entry in `MODE_CONFIGS`).
4. **Scalar tokens.** At 96 k, early exit shows `turn`/`castle` at chance until
   the final block (0.51 → 1.00) and `scalar_acc` 0.81 → 1.00. Part of the last
   block's apparent value is the state tokens, not board content; the piece-token
   jump (+5.0 pt) is the honest content number. Worth one probe with the scalar
   tokens excluded from the metric.
5. **Latent width.** Effective rank tops out at ~170 of 512, and accuracy is
   insensitive to truncation above `k=192`. A 256-d latent would be cheaper to
   decode and nearly free on reconstruction — but the material probe does
   improve at full 512 in deeper arms, so this is a recipe question, not an
   optimization one. Defer.
6. **Seed spread.** Every arm is a single seed. The 6→8 and 8→12 gaps
   (0.1-0.3 pt) are the size of unmeasured seed noise. Two extra runs
   (`D6`, `D12`, ~25 and ~43 min) settle whether the knee is 6 or 8 before a
   long run is committed to it.
7. **Batch-256 optimization.** Doubling the batch doubles tokens per step,
   which changes the useful LR, warmup and schedule horizon. Budget a short
   LR re-scan (3 values × 1k steps) at `8:4 @ b256` before comparing it to
   the `b128` baseline; otherwise the shape change will be confounded with a
   mis-tuned schedule.

---

## 2. P0 — Activation checkpointing (biggest lever, do first)

**Target.** With the measured knee at 6-8 encoder blocks (§1.3), the
first-class shape is **`8:4` at batch 256** (`8:3` as the fallback until the
decoder-depth arm in §1.6 confirms the fourth decoder block), not `16:4` at
batch 128. Measured without checkpointing at batch 64: 22.5 GB for `8:3`, so
`8:4` is ~24.2 GB — ~45 GB at batch 128 and OOM at batch 256. Checkpointing is
what buys the token budget. `16:4` remains a one-off depth check, not a
production target.

**VRAM with checkpointing** (stores block inputs, recomputes block internals):

| config | stored inputs | live block | total |
|---|---:|---:|---:|
| `6:3` @ b128 (today, no ckpt) | — | — | 34.7 GB (measured) |
| **`8:4` @ b256 (new target)** | 6.4 GB | 7.6 GB | **~15 GB** |
| `8:3` @ b256 (fallback) | 4.3 GB | 7.6 GB | **~13 GB** |
| `16:4` @ b128 | 4.3 GB | 3.8 GB | **~9 GB** |
| `24:4` @ b128 | 6.4 GB | 3.8 GB | **~11 GB** |
| `32:4` @ b256 | 17.2 GB | 7.6 GB | **~26 GB** |

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

**Acceptance.** Primary: `8:4` (12 blocks, 26.7M params) at **batch 256**
trains under 48 GB at ≥ 300k padded tok/s; `8:3` (11 blocks, 24.6M) is the
fallback if the decoder-depth arm does not pay. That is the real win over more
blocks: doubling the batch at the depth knee is worth ~1.5× tokens/s after the
~30% checkpointing overhead, whereas another 4 blocks were worth ≤ 0.3 pt of
top-1 at fixed tokens (§1.3). Secondary: 16 enc + 4 dec at batch 128 still fits
(the depth check from §1.4).

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
an argument for loops — depth is affordable directly. The depth probes sharpen
this: at a fixed token budget the marginal encoder block past 8 is worth ≤ 0.3
pt of top-1 (§1.3), so loops are **not** a cheap route to capacity. They remain
interesting for two reasons: (a) parameter sharing is a strong regulariser and
our budget is *tokens*, not parameters; (b) **test-time compute scaling**.
Ship PR1/PR2 first; only start this if the checkpointed `8:3 @ b256` run leaves
the loops-vs-depth question open.

**Test-time mode.** `--inference-loops K` evaluated at `K = 1, 2, 3, 4` on a
fixed pool. If accuracy or `R²_material` rises with `K` at fixed parameters,
that is a capability a fixed-depth model does not have. Record `n_loops` in
checkpoint metadata so `feature/position-vector-index` embedders reproduce it.

**Ablation (matched compute, ~5k steps each).** Target the measured knee
(§1.3) rather than 12 effective blocks: the question is whether 2 loops × 4
blocks matches 1 loop × 8 blocks, not whether either reaches 12.

| arm | blocks | loops | effective depth |
|---|---:|---:|---:|
| L1 | 8 | 1 | 8 (knee baseline) |
| L2 | 4 | 2 | 8 |
| L3 | 2 | 4 | 8 |
| L4 | 4 | 2 (checkpointed) | 8, memory of 4 |

If L1-L4 tie, loops add nothing at the knee; re-run L2 at effective depth 12
only if a later long-horizon run moves the knee (§1.4).

**Failure mode.** Fixed-point collapse: later iterations converge and compute
is wasted. The detectors now exist: run `scripts/probe_residual_drift.py` and
`scripts/probe_early_exit.py` on looped checkpoints and stop early if
`cos(h_loop, h_loop−1) → 1` or the exit curve is flat after the second loop.
Both scripts share `src/evaluation/probes.py` with the depth probes, so their
numbers are directly comparable to §1.2.

**Instability guard.** If loss spikes with loops ≥ 3, reduce loop LR via
per-loop scaling or drop to 2 loops; report rather than tune away.

---

## 6. P4 — DeepCrossAttention (DCA): learned residual routing (optional, architecture)

**Status.** Proposed; measure before adopting. DCA is orthogonal to P0-P3: P0
and P1 make depth affordable, DCA aims to make the depth you buy more
efficient. The 16-block arms still depend on P0, so land checkpointing before
them; the 9-block arms (A0-A2) can run without it.

**Revised scope (2026-11-28).** The depth probes (§1.3) put the fixed-token
encoder knee at 6-8 blocks, so the encoder-side question is "does routing make
8 blocks behave like more". The decoder has not been probed at any depth and is
the only consumer of the latent, so the decoder-side question is both "is a
fourth decoder block worth it" and "does routing pay there". The two sides are
routed independently (§6.2): no slot, stream or parameter crosses the
bottleneck. Lead with A0-A2 (encoder, no P0 needed), then A7-A8 (decoder at the
`8:4` candidate); A3-A6 are the secondary check. The paper's own caveat still
applies: the reported gain shrinks with width, and `d_model=512` sits
mid-range.

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
dilution cannot be recovered by the decoder or by a downstream probe. Depth is
worth buying up to the measured knee of 6-8 encoder blocks (§1.3), and DCA is
the only proposal in this plan that raises quality **per unit of depth**; the
rest only lower the price of compute.

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

**Scope: one stack per module, never across the bottleneck.** The encoder and
the decoder each own a private slot list and private GRNs. `dca_locations`
selects which of them route; `"both"` means **two disjoint stacks**, not one
shared list, and the two are parameterised independently. Nothing crosses the
bottleneck: the encoder slots die when `AttentionBottleneck` pools the window
into one `latent_dim` vector, and the decoder starts a fresh stack from
`from_latent(latent) + segment/slot embeddings`. This is a correctness
requirement, not a simplification — carrying encoder slots into the decoder
would hand it the per-token encoder states that the 512-d latent exists to
summarize, silently turning the autoencoder into a skip-connected model and
invalidating every downstream probe that treats the latent as the product.
It is also the only memory-safe option: the stacks have different lifetimes
and the decoder's token axis is rebuilt, not inherited.

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
dca_locations: str = "encoder"   # "encoder" | "decoder" | "both" = two independent stacks
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
11. **Bottleneck isolation.** Given identical encoder weights, `encode_latent`
    is bit-identical in fp32 between `dca_locations="encoder"` and `"both"`;
    only the reconstruction changes. Structurally, the decoder's stack is
    empty at `x0` and holds no tensor that aliases an encoder activation
    (`data_ptr` disjointness after one forward). This is the test that stops a
    future refactor from threading one stack through the bottleneck.

### 6.4 Ablation (short, matched compute)

Hardware: 6000 Ada 48 GB, P0+P1 landed, effective batch 128, bf16, compile,
flash. Use the current packed-window corpus and one fixed validation pool so
`val_loss` is comparable across arms.

| arm | blocks | routing | stack | purpose |
|---|---:|---|---:|---|
| A0 | 9 | standard | — | current baseline |
| A1 | 9 | GRN-v1 only | 2 | cheapest DenseFormer control; paper's largest single gain |
| A2 | 9 | DCA | 2 | DCA at the affordable depth — **primary comparison** |
| A3 | 16 | standard | — | depth bought by P0 |
| A4 | 16 | DCA | 2 | depth check (secondary) |
| A5 | 16 | DCA | 0 (full) | cost of completeness |
| A6 | 16 | DCA, decoder only | 2 | where the gain lives |
| A7 | 12 (8:4) | standard | — | decoder-depth control at the candidate shape |
| A8 | 12 (8:4) | DCA, decoder only | 2 | decoder routing at the candidate shape |

Run **A0-A2 first**: the encoder-side comparison, fits without P0. Then
**A7/A8** at the `8:4` candidate shape — decoder depth is the largest untested
axis (§1.6), and decoder routing can only be judged after the decoder depth is
chosen. A3-A6 are secondary and depend on checkpointing. Because A3 is already
known to be within noise of A0 on top-1 at fixed tokens (§1.3), the meaningful
bars are A2 vs A0 and A8 vs A7 at matched wall-clock — not raw depth against a
fixed token count. `dca_locations="both"` adds both stacks and costs their sum;
it is a separate arm, not a free flag on A4/A8.

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
- **Encoder side (A0-A2):** A2 reaches A0's final `val_loss` in **≤ 0.85× the
  tokens and ≤ 0.95× the wall-clock**.
- **Decoder side (A7-A8, candidate shape):** A8 beats A7 at matched
  wall-clock; if A7 already shows the fourth decoder block is worth < 0.5 pt
  over `8:3`, the candidate shape reverts to `8:3` and A8 is judged there.
  Either way the decoder-depth decision is recorded with numbers.
- **Depth check (A3-A4):** A4 reaches A3's final `val_loss` in **≤ 0.67× the
  tokens and ≤ 0.9× the wall-clock**, or improves final `val_loss` by ≥ 2%
  relative at matched wall-clock.
- `test_r2_material` improves outside the 95% CI, not just reconstruction
  loss.
- Peak VRAM at `8:4 @ b256` stays under 44 GB with checkpointing; step time
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
| DCA isolation | `encode_latent` with `dca_locations="both"` vs `"encoder"`, same encoder weights | bit-identical (fp32); no aliased encoder tensor in the decoder stack |
| DCA memory | peak VRAM with `S=4`, 16 blocks | no `S×` scaling; ≤ 44 GB |
| DCA throughput | `tokens/s` at `16:4 @ b128` | ≥ 220k (≤ 1.35× step time) |
| depth knee at the target horizon | `scripts/probe_early_exit.py` + `scripts/probe_layer_pruning.py` on the final checkpoint | `k*` = full depth and every block ≥ 0.5 pt, i.e. the §1.3 knee has not moved |

---

## 8. Rollout (one PR each, all from `main`)

1. **PR1 — activation checkpointing** (+ config, flags, 4 tests, docs).
   Unlocks the new primary shape **`8:4` @ batch 256** (`8:3` fallback); the
   `16:4 @ b128` run stays as the depth check.
2. **PR2 — gradient accumulation** (+ effective-batch resume fix, 3 tests).
   Also the fallback route to a large effective batch if `b256` is tight.
3. **PR3 — chunked CE, `zero_grad(set_to_none=True)`, dataloader tuning**
   (memory/latency only; no numerics change).
4. **PR4 — DeepCrossAttention** (GRN-v3, DCA block, config flags, 11 tests,
   loop-level checkpoint integration, ablation report). Encoder and decoder
   stacks are independent by construction (§6.2, test 11). Lead with A0-A2
   (encoder side), then A7-A8 (decoder side at `8:4`); run A3-A6 only if the
   first two groups win. Ships disabled by default; promote only if §6.6
   passes.
5. **PR5 — looped encoder** (optional; only if the checkpointed `8:4 @ b256`
   run and P4 leave the depth-vs-loops question open). If both P3 and P4 land,
   the looped encoder resets the DCA stack at each iteration.

Each PR runs the verification matrix on a fresh box before merge; PR1 and PR2
also get a real 200-step smoke run at `8:4 @ b256` (plus one `16:4 @ b128`
smoke for the depth check) to confirm the throughput and memory estimates on
hardware rather than on paper.

---

## 9. Success metrics

- **Primary shape (revised 2026-11-28):** `8:4` (12 blocks, 26.7M params) at
  **batch 256** under 48 GB, at ≥ 300k padded tok/s (checkpointing overhead
  < 35 %); `8:3` (11 blocks, 24.6M) is the fallback if the fourth decoder block
  does not pay.
- Secondary: `16:4` (20 blocks, 43.5M params) at batch 128 still fits, for the
  one-off depth re-check (§1.4).
- Zero recompilation growth across 1k steps; FlashAttention used on both passes.
- Equivalence tests prove checkpointing and accumulation are numerically inert.
- The depth knee holds at the target horizon: full-depth `k*` and every block
  ≥ 0.5 pt of pruning cost; the decoder-depth arm (§1.6) picks 3 vs 4 decoder
  blocks with numbers.
- If P3 lands: a measured accuracy-vs-inference-loop curve for the looped model.
- If P4 lands: A2 beats A0 at matched wall-clock (encoder side), A8 beats A7 at
  `8:4` (decoder side), `test_r2_material` improves outside the 95% CI, and the
  `16:4` shape still fits under 44 GB at ≥ 220k tok/s. Encoder and decoder
  stacks stay independent (isolation test). Otherwise the negative result is
  recorded and `dca` stays off.
