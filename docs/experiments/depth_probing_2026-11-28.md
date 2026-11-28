# Probing the effect of encoder depth (2026-11-28)

Status: in progress. Plan: [`docs/plan_depth_probing.md`](../plan_depth_probing.md).

Primary checkpoint: `run_ae_d512l6h8_100k/best_step96000.pt` (d_model 512,
6 encoder / 3 decoder layers, d_ff 1024, latent 512, 20.4M params) — the plan
was written while this run was at step 34 000; the run finished at 96 000.
Reference arm: `run_ae_20m_100k/best_step34000.pt` (4 encoder layers, d_ff
1536, latent 1024, 20.4M params).

## Setup

- **Pool.** Three freshly downloaded `test/` shards
  (`lichess_db_standard_rated_2020-09/10/11/batch_0000.parquet`), streamed
  with `load_game_pool(dir, 8192, prefetch_shards=2)`.
- **Windows.** `WindowDataset` + `MODE_CONFIGS["board"]`, `seed=17`,
  `max_seq_len=2048`, `max_segments=30`, first 128 indices of the seeded
  permutation. Identical for every probe and arm, so all comparisons are
  paired.
- **Device.** P1-P4 on CPU, 8 threads; P5 on one RTX 6000 Ada (vast.ai
  instance 63013073).
- **Noise.** 128 windows ≈ 34 304 masked token predictions per arm. Windows
  are correlated, so treat 0.3 pt as the per-run floor and 0.5 pt as the
  smallest difference worth acting on.

## P1 — Residual drift

`scripts/probe_residual_drift.py`, 32 windows per checkpoint.

| block | 6L `rel_delta` | 6L `cos` | 4L `rel_delta` | 4L `cos` |
|---:|---:|---:|---:|---:|
| 1 | 0.556 ± 0.017 | 0.865 | 0.776 ± 0.015 | 0.771 |
| 2 | 0.530 ± 0.021 | 0.884 | 0.703 ± 0.037 | 0.839 |
| 3 | 0.558 ± 0.018 | 0.890 | 0.584 ± 0.029 | 0.884 |
| 4 | 0.581 ± 0.022 | 0.884 | 0.877 ± 0.075 | 0.805 |
| 5 | 0.512 ± 0.022 | 0.905 | — | — |
| 6 | 0.648 ± 0.055 | 0.855 | — | — |

No block is anywhere near the identity: `rel_delta` is flat around 0.5-0.65
for the 6-layer model and 0.58-0.88 for the 4-layer model, and the **last
block has the largest delta in both**. **Verdict: depth-limited — no
near-identity blocks.**

## P2 — Early exit

`scripts/probe_early_exit.py`, 128 windows, board masking, masked-token
accuracy after block `k` (ln_final + bottleneck + decoder + head).

| k | 6L acc | 6L piece | 4L acc | 4L piece |
|---:|---:|---:|---:|---:|
| 1 | 0.6890 | 0.6856 | 0.7318 | 0.7209 |
| 2 | 0.7278 | 0.7260 | 0.7821 | 0.7735 |
| 3 | 0.8030 | 0.8018 | 0.8670 | 0.8615 |
| 4 | 0.8428 | 0.8446 | **0.9153** | 0.9116 |
| 5 | 0.8839 | 0.8875 | — | — |
| 6 | **0.9403** | 0.9375 | — | — |

The curve rises monotonically to the final block in both models; the last
block buys **+5.6 pt** (6L: 0.8839 → 0.9403 ⇒ k\* = 6) and **+4.8 pt**
(4L: 0.8670 → 0.9153 ⇒ k\* = 4). **Verdict: depth-limited — decodable
information grows to the final block.**

Caveat: `ln_final` is trained on the last block's output, so early exits are
slightly off-distribution; the uniform rise and the P4 causal screen below
are the controls for that.

## P3 — Latent richness vs depth

`scripts/eval_latent_truncation.py`, 512 basis windows, 128 accuracy
windows, 1024-board material pool, all six depths. One encoder pass per pool
keeps the latent at every exit depth; PCA is fit per depth.

| depth | eff. rank 99 % | eff. rank 99.9 % | acc @512 | acc @192 | acc @64 | `R²_mat` @512 | `R²_mat` @192 | `R²_mat` @64 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 74 | 143 | 0.6890 | 0.6890 | 0.6866 | 0.427 | 0.426 | 0.189 |
| 2 | 87 | 156 | 0.7278 | 0.7278 | 0.7233 | 0.434 | 0.432 | 0.191 |
| 3 | 107 | 166 | 0.8030 | 0.8030 | 0.7859 | 0.467 | 0.461 | 0.138 |
| 4 | 124 | 170 | 0.8428 | 0.8427 | 0.8094 | 0.387 | 0.364 | 0.034 |
| 5 | 142 | 174 | 0.8839 | 0.8834 | 0.8448 | 0.436 | 0.398 | 0.027 |
| 6 | 170 | 186 | 0.9403 | 0.9402 | 0.8871 | 0.553 | 0.507 | 0.025 |

The latent gets **strictly richer** with depth: the 99 %-variance rank grows
74 → 170 and the 99.9 % rank 143 → 186, both monotone. Reconstruction is
insensitive to truncation above ~192 dims (the full-rank and `k=192` columns
agree to <0.1 pt), so the extra rank lives in low-variance directions.

The material probe is the one noisy row: at full rank the deepest model is
best (0.553) but the per-depth CIs (±0.08, 1024 boards) overlap, and in a
64-dim PCA subspace material `R²` falls with depth (0.19 → 0.03). Read that
as "deeper latents spend their leading components on finer board detail, not
global material", not as a material regression — the probe is a screen.
**Verdict: depth-limited — embedding rank grows with every added block.**

## P4 — Layer pruning

`scripts/probe_layer_pruning.py`, 128 windows, 1024-board material pool.
Zero-ablation = skip the block, residual passes through. Baseline:
masked accuracy 0.9403.

| block | masked acc | drop | `R²_material` |
|---:|---:|---:|---|
| 0 | 0.9090 | 3.12 pt | 0.394 |
| 1 | 0.9200 | 2.03 pt | 0.533 |
| 2 | 0.9187 | 2.16 pt | 0.461 |
| 3 | 0.9103 | 2.99 pt | 0.524 |
| 4 | 0.9018 | 3.85 pt | 0.328 |
| 5 | 0.8839 | 5.64 pt | 0.436 |

Every block costs at least 2.0 pt and the **last block costs the most**
(5.64 pt). **Verdict: depth-limited — the causal contribution of the final
block is the largest, not zero.**

## P5 — Depth sweep

`scripts/…` → 4 / 6 / 8 / 12 encoder layers, 5000 steps, batch 64,
`p_rect 0.34 / p_skip_gram 0.66`, balanced-v1, lr 2e-4, warmup 500, seed 17,
compiled bf16 on an RTX 6000 Ada.

<!-- P5 table -->

## Verdict per §1 row

<!-- verdict table -->
