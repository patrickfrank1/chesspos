# Perceiver 6M: LR schedule collapse and validation drift (2026-11-10)

## Status

Stopped at logged step **41,500** (last checkpoint step **41,000**). The GPU pod
was decommissioned after syncing all run artifacts to the Hugging Face bucket.

This report covers the segment-aware Perceiver IO experiments after the initial
content-collapse fix. The earlier failure analysis is in
[analysis_perceiver_content_collapse_2026-11-09.md](analysis_perceiver_content_collapse_2026-11-09.md).

## TL;DR

- The first fixed Perceiver run (`run_perceiver6m`) trained normally to ~33k
  steps, then **train loss exploded** and the retrieval embedding collapsed
  again. This was an optimization/LR-schedule failure, not a forward-pass bug.
- Extending the run from 72k to 200k steps stretched the cosine schedule and
  kept the LR near **2.8e-4** at 30–33k, versus ~**1.7–1.9e-4** under the original
  72k schedule. Divergence began at ~33.4k.
- The resume run (`run_perceiver6m_lr2e4`) used the best 20k checkpoint, peak LR
  **2e-4**, eval every **2,000** steps, and an NCE collapse guard. Training loss
  stayed flat at ~**0.60–0.63** and NCE stayed at ~**0.56–0.62**. The NCE guard
  did **not** trip.
- However, validation loss drifted upward after step 34k, and material R²
  remained poor. This is a **content-preservation / validation-drift** problem,
  not another loss explosion.
- The Perceiver architecture is not proven to be fundamentally broken, but the
  current content objective is not learning material. More MLM steps alone are
  unlikely to fix that.

## Runs

| Run directory | Purpose | Outcome |
|---|---|---|
| `run_perceiver6m_maxseg128_failed_20261109T230652Z` | Initial `max_segments=128` attempt | Failed before useful training (OOM/config) |
| `run_perceiver6m_collapsed_20261109T232217Z` | First full Perceiver run | Content embeddings collapsed at step 1000 |
| `run_perceiver6m_latentcollapse_20261109T232839Z` | Diagnostic after small segment-embedding init | Still collapsed before decoded-slot readout |
| `run_perceiver6m` | Fixed readout, 3e-4 peak, 200k schedule | Train-loss explosion at ~33.4k; stopped at 54k |
| `run_perceiver6m_lr2e4` | Resume from best 20k, 2e-4 peak, eval 2k, NCE guard | Stable train loss; val drifted after 34k; stopped at 41.5k |

## Model and data configuration

- Parameters: **5.96M**
- Architecture: `perceiver_io`
- `d_model=256`, `n_heads=16`, `d_ff=832`, `n_layers=4`
- `n_content_latents=16`, `n_state_latents=1`
- `n_local_latent_layers=0`, `n_cross_segment_layers=1`, `n_decoder_layers=1`
- `max_seq_len=2048`, `max_segments=30`
- `decoder_reconstruction_weight=0.0`
- Batch size: 256
- Masking: 50% rectangle, 50% skip-gram
- Objectives: MLM + invariance + VICReg + InfoNCE (`nce_weight=0.1`)
- Data: `patrickfrank1/chess-positions`, lazy train-shard streaming, 8,192-game
  validation pool

## What happened in `run_perceiver6m`

Training was healthy through ~22k steps:

| Step | Total loss | MLM | NCE | Val loss |
|---:|---:|---:|---:|---:|
| 2,000 | 1.012 | 0.588 | 0.801 | 1.303 |
| 8,000 | 0.689 | 0.412 | 0.637 | 1.001 |
| 16,000 | 0.619 | 0.369 | 0.578 | 0.885 |
| 20,000 | 0.613 | 0.365 | 0.591 | 0.859 |
| 24,000 | 0.691 | 0.448 | 0.584 | 0.859 |
| 28,000 | 0.757 | 0.494 | 0.625 | 0.964 |
| 32,000 | 0.887 | 0.561 | 0.874 | 1.043 |

Then the model destabilized:

```text
step 33,360  MLM 0.665
step 33,400  MLM 1.436
step 33,420  MLM 2.742
step 36,000  total 4.633, NCE 7.486, cov_loss 3.524
```

Evaluation confirms a real collapse, not just noisy loss:

| Step | Top-10 cosine | Random-pair cosine | Material R² | Side-to-move |
|---:|---:|---:|---:|---:|
| 20,000 | 0.465 | 0.090 | -0.534 | 0.585 |
| 32,000 | 0.516 | 0.111 | -0.531 | 0.546 |
| 36,000 | **0.984** | **0.483** | -0.083 | 0.517 |
| 52,000 | **1.000** | **0.974** | -0.019 | 0.517 |

The immediate cause was a bad update around 33.4k. The underlying cause was the
LR schedule change: `run_perceiver6m` was resumed from step 2,000 with
`--steps 200000` instead of the original 72,000. At step 33k the 200k cosine
schedule still used `~2.8e-4`, while the 72k schedule would have used
`~1.7e-4`. The model stayed near peak LR substantially longer and eventually
destabilized.

The NCE collapse guard did not exist during this run. Eval every 4,000 steps
also delayed detection: the run was already collapsed by the first post-onset
eval at 36k.

## What happened in `run_perceiver6m_lr2e4`

Resumed from the step-20,000 best checkpoint (`val_loss=0.8588`).

Configuration changes:

- Peak LR: **2e-4**, cosine decay
- Eval: every **2,000** steps
- NCE collapse guard: trip if NCE > 1.5 for 2 logged intervals after NCE < 1.0
- `--no-resume-data-skip`: replay the data stream from the beginning instead of
  fast-forwarding through 20k steps

Training remained stable:

```text
step=20060  loss=0.6117  mlm=0.3631  nce=0.5901  lr=1.95e-4
step=41500  loss=0.6271  mlm=0.3779  nce=0.6062  lr≈1.80e-4
```

NCE was stable at ~0.56–0.62. The NCE guard never tripped, and the retrieved
geometry stayed healthy:

| Step | Top-10 cosine | Random-pair cosine | Material R² | Side-to-move | PC1 piece corr |
|---:|---:|---:|---:|---:|---:|
| 22,000 | 0.470 | 0.096 | -0.526 | 0.546 | 0.654 |
| 28,000 | 0.467 | 0.086 | -0.526 | 0.551 | 0.653 |
| 32,000 | 0.478 | 0.087 | -0.625 | 0.566 | 0.655 |
| 34,000 | 0.477 | 0.085 | -0.539 | 0.581 | 0.637 |
| 36,000 | 0.476 | 0.086 | -0.584 | 0.590 | 0.649 |
| 38,000 | 0.484 | 0.091 | -0.672 | 0.566 | 0.651 |
| 40,000 | 0.484 | 0.092 | -0.668 | 0.566 | 0.656 |

Validation loss improved slightly to **0.8510 at step 34,000**, then drifted
upward:

| Step | Val loss |
|---:|---:|
| 28,000 | 0.8656 |
| 30,000 | 0.8576 |
| 32,000 | 0.8528 |
| 34,000 | **0.8510** |
| 36,000 | 0.8645 |
| 38,000 | 0.8917 |
| 40,000 | 0.9015 |

### Interpretation

This run did **not** reproduce the train-loss explosion. It exposed two separate
issues:

1. **Validation drift / repeated data.** Because resume-data-skip was disabled,
   steps 20k–40k replayed the same first 20k steps the model had already seen.
   The model only reached fresh data at ~40k. This likely contributed to the
   val-loss rise and means the 36k–40k validation trend is not a clean test of
   the architecture.
2. **Content is still not material-aware.** Material R² stayed negative
   (`-0.50` to `-0.67`) and top-5 material delta was ~1.96 versus a random-pair
   delta of ~2.23. The content embedding is not forming a usable material axis.

The Perceiver’s latent bottleneck may be making content preservation harder,
but the current logs do not prove the architecture is the sole cause. The
current objective also has no term that directly forces content latents to
preserve material: `decoder_reconstruction_weight=0.0`, and the retrieval
readout is trained only through the downstream MLM/InfoNCE/VICReg mix.

## Conclusions

- **Do not extend a run by changing `--steps` alone.** The cosine LR schedule
  changes with the target; treat the extension as a new optimization run or
  explicitly re-plan the LR schedule.
- **A flat train loss does not mean the representation is healthy.** The first
  run’s NCE and train loss looked fine until the collapse phase transition.
- **Validation drift can be caused by data replay.** Streaming resume at large
  step counts is expensive; `--no-resume-data-skip` avoids the skip but repeats
  data. A fast skip or saved data-stream position is needed for clean long
  resumes.
- **The Perceiver content objective is not learning material.** This is the
  important open issue. More steps alone are not a credible fix.
- The NCE guard is useful for the specific collapse mode, but it cannot detect
  slow semantic drift or material loss.

## Recommended next experiments

1. **Content-preservation loss.** Add a visible-content identity loss (predict a
   token from its own final hidden state at visible positions) or set a
   non-zero decoder reconstruction weight on the content path.
2. **Material auxiliary head.** Add a small linear material/count head on the
   pooled content embedding and train it jointly. This directly tests whether
   the content latent can carry material.
3. **Short architecture bake-off.** Run 10k–20k-step comparisons with eval every
   1k–2k:
   - current Perceiver + content-preservation loss
   - transformer + content-preservation loss
   - Perceiver with a weaker bottleneck / larger content bank
4. **Fix large-step resume.** Implement fast data skipping (advance raw games
   without masking) or save the data-stream position in the checkpoint.
5. **Keep collapse guards.** Keep the NCE guard and add a validation-loss /
   material-R² early-stop guard.

## Artifacts

Hugging Face bucket:
`hf://buckets/patrickfrank1/chesspos-checkpoints/`

- `run_perceiver6m_lr2e4/`
  - `best.pt` (step 34,000, `val_loss=0.8510`)
  - `last.pt` (step 41,000, `val_loss=0.9015`)
  - `best_step*.pt` uploads for 30k/32k/34k
  - `train.log`, `eval_history.jsonl`, `eval_metrics.json`
  - `hard_negatives/`, `neighbor_examples.json`, `pca_phase.png`
  - `watchdog.sh`, `watchdog.log`, `nohup.out`, `code_diff.patch`
- `run_perceiver6m/`
  - collapsed run `last.pt` at step 54,000
  - best checkpoint at step 20,000 (`val_loss=0.8588`)
  - full logs, eval history, hard negatives, watchdog script
- `run_perceiver6m_collapsed_20261109T232217Z/`
- `run_perceiver6m_latentcollapse_20261109T232839Z/`
- `run_perceiver6m_maxseg128_failed_20261109T230652Z/`
