# Plan: Probing the Effect of Encoder Depth

Status: proposed (2026-11-24). Owner: training. Applies to the attention
autoencoder (`src/modeling/autoencoder/`), currently `d_model 512`, 6 encoder /
3 decoder layers, `d_ff 1024`, 512-d latent, 20.4M params.

---

## 1. The question

We increased encoder depth 4 → 6 and reconstruction improved at equal
bottleneck width (+0.6 pt at `k=512..192`, +1.5 pt at `k=64`; see the PCA
truncation probe in `feature/autoencoder-eval-loss-and-architecture`
discussion). That change also moved latent 1024 → 512, `d_ff` 1536 → 1024 and
masking 50/50 → 34/66, so **depth itself is not yet isolated**.

The question this plan answers, in the form that can actually be measured:

> Does another encoder layer still add information to the pooled window
> embedding, at a fixed token budget?

"Depth is capacity-limited" and "depth is saturated" make opposite predictions
on six cheap signals:

| signal | depth-limited | depth-saturated |
|---|---|---|
| last-layer pruning cost | ≥ noise floor | ~0 |
| early-exit accuracy | rises to the final block | plateaus mid-stack |
| consecutive-block drift | stays high | → 0 (near-identity blocks) |
| latent effective rank vs depth | grows | flat |
| depth sweep slope at the top | positive past noise | flat |
| iso-FLOP: depth vs tokens | depth wins | tokens win |

Every probe below is one of those rows. The last row is the only one that
decides how to spend money, and it is the most expensive.

---

## 2. Common instrumentation

Everything is measured on a **fixed, paired** pool so differences are not pool
noise.

- **Checkpoints.** `best_step34000.pt` from
  `hf://buckets/patrickfrank1/chesspos-checkpoints/run_ae_d512l6h8_100k` (6 layers) and,
  for reference, `run_ae_20m_100k/best_step34000.pt` (4 layers). The depth
  sweep (P5) produces the rest.
- **Pool.** Three freshly downloaded `test/` shards (**not** the cached
  snapshot — its symlinks dangle because the trainer deletes blobs after
  streaming). `hf download patrickfrank1/chess-positions --repo-type dataset
  --include "test/lichess_db_standard_rated_2020-09/batch_0000.parquet" …`,
  then `load_game_pool(dir, 8192, prefetch_shards=2)`.
- **Windows.** `WindowDataset` + `MODE_CONFIGS["board"]`, `seed=17`,
  `max_seq_len=2048`, `max_segments=30`. Same window indices for every arm and
  every probe → paired comparisons.
- **Noise floor.** 128 windows × ~1,150 masked content tokens ≈ 147k token
  predictions; binomial SE at p≈0.91 is 0.075 pt, but windows are correlated,
  so treat **0.3 pt as the per-run floor and 0.5 pt as the smallest
  difference worth acting on**.
- **Artifacts.** Each probe writes JSON to `data/eval/<probe>/<ckpt>.json` and
  is copied to
  `hf://buckets/patrickfrank1/chesspos-checkpoints/run_ae_d512l6h8_100k/probes/`.
- **Device.** P1-P4 are CPU (`torch.set_num_threads(8)`); they must not touch
  the training GPU.
- **Scripts.** `scripts/probe_residual_drift.py`,
  `scripts/probe_early_exit.py`, `scripts/eval_latent_truncation.py`,
  `scripts/probe_layer_pruning.py` (optional), all reusing
  `src/evaluation/eval_suite.py` (`sample_segments`, `board_metadata`,
  `material_probe_ci`, `embed_segments`).

---

## 3. Probe ladder, fastest → slowest

| # | probe | device | wall clock | decides |
|---|---|---|---|---|
| P1 | residual drift | CPU | < 1 min | are any blocks near-identity? |
| P2 | early exit | CPU | ~15 min | where does decodable information stop growing? |
| P3 | latent richness vs depth | CPU | ~15 min | does depth enrich the embedding or just the residual stream? |
| P4 | layer pruning (optional) | CPU | ~20 min | per-layer contribution |
| P5 | depth sweep | GPU | ~2.2 h | does the trained curve have a knee? |
| P6 | test-time compute scaling | GPU | ~1 day | do loops buy accuracy at inference? (blocked on plan B) |

### P1 — Residual drift (fastest; run first)

**Method.** For each batch, capture `encoder.blocks[i]` outputs (hook, flat
varlen tensor reconstructed to `[B, T, d]` via `scatter_valid`) and report per
block:

- `rel_delta_l = ‖h_l − h_{l−1}‖ / ‖h_{l−1}‖`
- `cos_l = cos(h_l, h_{l−1})`

**Reads as depth-limited** if `rel_delta` stays roughly flat across the stack.
**Reads as depth-saturated** if the last blocks trend toward `rel_delta → 0`,
`cos → 1` — near-identity blocks burning compute.

**Output.** One table: block, mean ± sd of `rel_delta`, mean `cos`.

```bash
PYTHONPATH=/workspace/chesspos python3 scripts/probe_residual_drift.py \
  --checkpoint /workspace/ckpts/new_34000.pt \
  --data-dir /workspace/probe-test/test --windows 32 --out /workspace/probes/drift.json
```

### P2 — Early exit

**Method.** Re-implement the encoder loop with an exit point: after block *k*,
apply `encoder.ln_final`, pool with `model.bottleneck(x, valid)`, then
`model.decode(latent, …)` + `model.head` and score masked content tokens. Same
128 windows and the same targets for every *k*, so the curve is paired.

**Output.** `k` vs masked-token accuracy (and the piece-class split), 1..6.

**Decision.** Let `k*` be the first depth within 0.3 pt of the full model.
`k* = 6` → depth still paying. `k* ≤ 4` → the last two blocks add nothing
decodable through the trained bottleneck.

```bash
python3 scripts/probe_early_exit.py --checkpoint … --data-dir … \
  --windows 128 --batch-size 8 --out /workspace/probes/early_exit.json
```

### P3 — Latent richness vs depth

**Method.** The PCA-truncation probe (`scripts/eval_latent_truncation.py`,
prototype already written during this investigation) run at **each early-exit
depth**: encode, pool, PCA, then for `k ∈ {full, 256, 192, 128, 96, 64}`
measure reconstruction accuracy and the ridge material probe.

**Why it is the decisive probe for "capacity for reasoning".** It asks whether
the *embedding* — the only artifact downstream tasks ever see — carries more
information at depth 6 than at depth 4, independently of what the residual
stream holds. A deeper encoder that leaves the 192-dim PCA summary unchanged
has not improved the product.

**Output.** Per depth: effective rank (dims for 99 % / 99.9 % variance),
`R²_material@k`, `accuracy@k`.

**Reads as depth-limited** if effective rank or `R²@192` grows monotonically
with depth. **Depth-saturated** if both are flat from `k*` onward. Baseline for
reference: the 6-layer model has **effective rank ≈ 192** and
`R²_material@k=192 ≈ 0.56`.

### P4 — Layer pruning (optional, cheap)

**Method.** Zero-ablate block *i* (skip it; residual carries through) and
measure the accuracy drop and `R²_material` drop, for each `i`. No retraining.

**Reads as depth-limited** if every block costs ≥ 0.5 pt. **Saturated** if the
last blocks cost nothing — a stronger statement than P1/P2 because it measures
*causal* contribution.

**Caveat.** Zero-ablation is a distribution shift, not a trained ablation; use
it as a screen, not proof.

### P5 — Depth sweep (the real test)

**Method.** Train matched arms and measure the curve, not a single point.

| arm | flags | blocks | params | step time est. |
|---|---|---:|---:|---:|
| D4 | `--n-layers 4` | 7 | 16.4M | 0.26 s |
| D6 | `--n-layers 6` (baseline) | 9 | 20.4M | 0.33 s |
| D8 | `--n-layers 8` | 11 | 24.4M | 0.40 s |
| D12 | `--n-layers 12` | 15 | 33.0M | 0.55 s |

Held fixed: `d_model 512`, `n_heads 8`, `d_ff 1024`, `latent_dim 512`,
`n_decoder_layers 3`, `--token-weight-preset balanced-v1`, `p_rect 0.34 /
p_skip_gram 0.66`, `lr 2e-4`, `warmup 500`, `seed 17`, `batch_size 64`,
`max_seq_len 2048`, `max_segments 30`, **5,000 steps** (~2-3× past the point
where val accuracy differences are visible in the 20M run).

Batch 64 for all arms: D12 at batch 128 needs ~45.6 GB and would not fit, and
the comparison must be at matched batch, not matched VRAM.

```bash
for N in 4 6 8 12; do
  python3 -m src.run.train_masked_transformer \
    --model-arch attention_autoencoder --n-layers $N --n-decoder-layers 3 \
    --d-model 512 --n-heads 8 --d-ff 1024 --latent-dim 512 \
    --token-weight-preset balanced-v1 \
    --data-repo patrickfrank1/chess-positions --data-cache data/hf-cache \
    --stream-shards --streaming --stream-shard-cache data/hf-shards \
    --num-workers 12 --steps 5000 --batch-size 64 --lr 2e-4 --warmup 500 \
    --max-seq-len 2048 --max-segments 30 --seed 17 \
    --log-every 100 --eval-every 1000 --eval-batches 20 --save-every 1000 \
    --bf16 --compile --no-resume-data-skip \
    --checkpoint-dir data/models/depth_sweep/N$N \
    --log-file data/models/depth_sweep/N$N/train.log
done
```

**Metrics.** Content-only val loss (requires the eval-loss fix from
`feature/autoencoder-eval-loss-and-architecture`; without it `val_loss` is
inflated by untrained `CLS`/`SEP` but the *slope across arms* is still usable),
masked accuracy, `R²_material`, effective latent rank, and P2/P3 re-run on each
final checkpoint.

**Decision rule.** Fit accuracy vs blocks over the four arms.
- Monotone with slope above noise at 15 blocks → depth is capacity-limited, and
  the next run should be 3:1 or 4:1 at 16-24 blocks (see
  `docs/plan_autoencoder_optimizations.md`).
- Knee at ≤ 11 blocks → stop scaling depth; spend the budget on tokens
  (batch 128, more steps) instead.
- Non-monotone → treat as noise; re-run the ambiguous arms with a second seed
  before concluding.

**Cost.** ≈ 2.2 GPU-hours ≈ $1.50 on a 6000 Ada; checkpoints are ~35-90 MB per
arm.

### P6 — Test-time compute scaling (slowest, blocked)

**Dependency.** Needs the looped encoder from
`docs/plan_autoencoder_optimizations.md` (§5). Nothing to run until that
lands.

**Method.** Two questions, both cheap once a looped checkpoint exists:
1. **Train-time loops.** Matched-compute arms at ~5k steps:
   `12 blocks × 1 loop`, `6 × 2`, `4 × 3`, all with `p_rect 0.34/0.66`,
   identical data order. Compare accuracy, `R²_material`, effective rank.
2. **Test-time loops.** Evaluate the `6 × 2` and `4 × 3` checkpoints at 1×, 2×,
   3×, 4× the trained loop count on the same 128 windows. Accuracy vs loop
   count at inference is the whole point of the family.

**Reads as a win** if accuracy or `R²_material` improves monotonically with
inference loops at *fixed parameters*. **Reads as fixed-point collapse** if the
curve is flat from 2× — the same signal as P1's `cos → 1`.

---

## 4. Execution order and gates

```
P1 drift        (1 min)   ─┐
P2 early exit   (15 min)   ├─→ gate: any sign of saturation?
P3 latent rank  (15 min)   │      no  → run P5 as planned
P4 pruning      (20 min)  ─┘      yes → run P5 anyway, but weight tokens over depth
P5 sweep        (2.2 h)    → gate: knee at ≤11 blocks? → depth vs tokens decision
P6 test-time    (1 day)    → only after plan B's looped encoder exists
```

Do P1-P4 in one sitting (~1 h, free relative to training). If P1-P4 all say
"depth still paying", P5's expected result is confirmed and P6 becomes the
interesting question. If they say "saturated", P5 quantifies the knee and the
next training run should buy tokens, not layers.

---

## 5. Confounds to control

1. **Tokens per hour.** Depth and data exposure trade off at fixed wall clock.
   The sweep holds *steps and batch* fixed, so deeper arms see the same tokens
   but cost more wall clock; P5 therefore answers "does depth help per token",
   and the iso-FLOP question is answered separately by comparing 16 blocks ×
   13 B tokens against 9 blocks × 26 B tokens at equal GPU-hours.
2. **Dataset drift.** The repo grows (536 → 555+ train shards during this
   work). Pin all probes to the three downloaded `test/` shards and record the
   shard list in every JSON artifact.
3. **Baseline eval noise.** The 20M/1024-d run and the current run used
   different pools; only *within-this-plan* comparisons are valid.
4. **Compile warmup.** Throughput and per-step timings must be read after the
   first logged interval, not the first step.

---

## 6. Deliverables

- Four probe scripts under `scripts/`, each with a `--out` JSON contract.
- `docs/experiments/depth_probing_<date>.md` with the tables from P1-P6 and an
  explicit verdict per row of the §1 table.
- Decision recorded: next run is `(blocks, batch, steps)` with the reasoning
  that produced it.
- Artifacts under
  `hf://buckets/patrickfrank1/chesspos-checkpoints/run_ae_d512l6h8_100k/probes/`.
