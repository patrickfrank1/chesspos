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

Four matched arms, 5000 steps each, batch 64, `p_rect 0.34 / p_skip_gram
0.66`, `balanced-v1` token weights, lr 2e-4, warmup 500, seed 17, compiled
bf16 on one RTX 6000 Ada (vast.ai instance 63013073). Only `--n-layers`
varies (4 / 6 / 8 / 12 encoder blocks; 3 decoder blocks in every arm).

```bash
python3 -m src.run.train_masked_transformer \
  --model-arch attention_autoencoder \
  --n-layers $N --n-decoder-layers 3 \
  --d-model 512 --n-heads 8 --d-ff 1024 --latent-dim 512 \
  --token-weight-preset balanced-v1 --p-rect 0.34 --p-skip-gram 0.66 \
  --data-repo patrickfrank1/chess-positions --data-cache data/hf-cache \
  --stream-shards --streaming --stream-shard-cache data/hf-shards \
  --num-workers 12 --steps 5000 --batch-size 64 --lr 2e-4 --warmup 500 \
  --max-seq-len 2048 --max-segments 30 --seed 17 \
  --log-every 100 --eval-every 1000 --eval-batches 20 --save-every 1000 \
  --bf16 --compile --no-resume-data-skip \
  --checkpoint-dir data/models/depth_sweep/N$N \
  --log-file data/models/depth_sweep/N$N/train.log \
  --upload-bucket patrickfrank1/chesspos-checkpoints \
  --upload-prefix depth_sweep/N$N
```

| arm | encoder blocks | params | val loss @1k | @2k | @3k | @4k | @5k | val acc @5k |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| D4 | 4 | 16.19M | 0.9369 | 0.7473 | 0.6964 | 0.6691 | **0.6597** | 0.8421 |
| D6 | 6 | 20.40M | 0.9318 | 0.7411 | 0.6675 | 0.6279 | **0.6202** | 0.8459 |
| D8 | 8 | 24.60M | 0.9165 | 0.7398 | 0.6604 | 0.6321 | **0.6237** | 0.8483 |
| D12 | 12 | 33.01M | 0.8992 | 0.7110 | 0.6310 | 0.5889 | **0.5776** | 0.8494 |

<!-- P5 table -->

### Response probes on each arm (5 000-step checkpoints)

Early-exit masked accuracy (128 paired windows, board masking). `k*` is the
first depth within 0.3 pt of the full model.

| arm | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 | last-block gain | k\\* |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| D4 | 0.7130 | 0.7702 | 0.8094 | **0.8457** | | | | | | | | | +3.6 pt | 4 |
| D6 | 0.6674 | 0.7448 | 0.7902 | 0.8236 | 0.8557 | **0.8608** | | | | | | | +0.5 pt | 6 |
| D8 | 0.6923 | 0.7385 | 0.7633 | 0.7895 | 0.8137 | 0.8335 | 0.8527 | **0.8590** | | | | | +0.6 pt | 8 |
| D12 | 0.6817 | 0.7289 | 0.7611 | 0.7787 | 0.7894 | 0.8066 | 0.8205 | 0.8347 | 0.8383 | 0.8491 | 0.8556 | **0.8635** | +0.8 pt | 12 |

Residual drift (`rel_delta` / `cos` per block) on the same checkpoints:

| arm | deltas |
|---|---|
| D4 | 0.322, 0.322, 0.267, 0.337 |
| D6 | 0.295, 0.294, 0.260, 0.280, 0.241, 0.310 |
| D8 | 0.286, 0.269, 0.270, 0.252, 0.229, 0.246, 0.236, 0.239 |
| D12 | 0.286, 0.277, 0.264, 0.250, 0.204, 0.191, 0.182, 0.180, 0.160, 0.169, 0.157, 0.201 |

Every block still carries decodable information in every arm, but the
margin shrinks with depth (+3.6 pt for D4's last block, +0.5-0.8 pt for the
deeper arms), and D12's trailing blocks attenuate to `rel_delta 0.16` before
the last block ticks back up to 0.20 — mild saturation, not collapse.

Latent truncation on each final arm checkpoint (512 basis windows, 1024
boards, GPU):

| arm | acc @512 | acc @192 | rank 99 % | `R²_mat` @512 |
|---|---:|---:|---:|---:|
| D4 | 0.8499 | 0.8500 | 91 | 0.180 |
| D6 | 0.8608 | 0.8607 | 99 | 0.215 |
| D8 | 0.8596 | 0.8594 | 101 | 0.275 |
| D12 | 0.8630 | 0.8626 | 99 | 0.392 |

Reconstruction accuracy and effective rank **stop moving** past D6 while the
material probe keeps improving (0.18 → 0.39): extra depth at a fixed token
budget buys attribute information, not more reconstructable content.

Layer pruning at 5 000 steps (D6 arm, skip one block):

| block | 0 | 1 | 2 | 3 | 4 | 5 |
|---|---:|---:|---:|---:|---:|---:|
| acc drop | 4.21 pt | 2.25 pt | 2.34 pt | 2.17 pt | 3.03 pt | **0.51 pt** |

The same block that costs 5.64 pt to remove in the converged 96 k model is
nearly free at 5 k: the final block **becomes load-bearing late in training**.
That is the main caveat when reading a short-horizon sweep.

### What P5 says

- 4 → 6 blocks: real gain at matched tokens (val loss −0.040, probe accuracy
  +1.1 pt, rank +8).
- 6 → 8 blocks: flat on loss (+0.004) and on top-1 probe accuracy (+0.0 pt,
  within noise), rank +2.
- 8 → 12 blocks: NLL keeps improving (−0.046) but top-1 only +0.3 pt and
  effective rank −2, i.e. the extra depth mostly sharpens confidence.
- Every arm is still improving at step 5 000 (≈ 0.37 B tokens), so the knee
  should be re-checked at the full budget before it is treated as final.





## Verdict per §1 row

| signal | observation | reads as |
|---|---|---|
| last-layer pruning cost | 5.64 pt in the converged 6L model; 0.51 pt at 5 k steps | **depth-limited** at convergence, free early |
| early-exit accuracy | rises to the final block in all four arms (`k*` = L) | **depth-limited** within a model |
| consecutive-block drift | no block near identity; trailing blocks of D12 attenuate to 0.16 | depth-limited, mild saturation at ≥ 8 blocks |
| latent effective rank vs depth | 74 → 170 inside the 96 k model; 91 / 99 / 101 / 99 across arms | grows within a model, **flat across arms past D6** |
| depth sweep slope at top | 4→6 −0.040 loss; 6→8 +0.004; 8→12 −0.046 loss, +0.3 pt top-1 | **knee at 6-8 blocks at fixed tokens** |
| iso-FLOP: depth vs tokens | not run (needs equal-GPU-hour arms) | open |

**Decision.** Depth is not saturated inside a trained model — every block,
including the last, is causally load-bearing once training has converged.
But at a fixed token budget the *marginal* arm beyond 6-8 blocks buys
calibration, not decodable content or latent rank. The next run should stay
at 6 encoder blocks (optionally test 8) and spend the saved compute on tokens
(batch 128 or a longer step budget), then re-run P2/P4 to confirm the knee at
the full horizon.

## Artifacts

- Probe JSONs: `data/eval/depth_probe/` locally, synced to
  `hf://buckets/patrickfrank1/chesspos-checkpoints/run_ae_d512l6h8_100k/probes/`.
- Depth-sweep checkpoints, logs and eval history:
  `hf://buckets/patrickfrank1/chesspos-checkpoints/depth_sweep/N{4,6,8,12}/`.
- Scripts: `scripts/probe_residual_drift.py`, `scripts/probe_early_exit.py`,
  `scripts/eval_latent_truncation.py`, `scripts/probe_layer_pruning.py`,
  shared plumbing in `src/evaluation/probes.py`, BDD tests in
  `test/unit/test_depth_probes.py`.
- GPU cost: one RTX 6000 Ada at $0.54/h, instance 63013073, ~2.6 h ≈ $1.40.
