# Perceiver content-embedding collapse (2026-11-09)

## Summary

Follow-up experiment: [Perceiver 6M: LR schedule collapse and validation drift](perceiver_6m_lr2e4_2026-11-10.md).

The first segment-aware Perceiver IO run trained MLM normally but its retrieval
embeddings collapsed to a nearly constant direction. InfoNCE stayed at roughly
`6.3` while the batch-256 chance level is `ln(256) = 5.55`. The loss functions
were computed correctly; the model found a degenerate solution that satisfied
the invariance and VICReg objectives while making cosine-based retrieval
meaningless.

Two changes fixed it:

1. Initialize the latent segment embedding with a small standard deviation.
2. Replace the shared-query latent readout with attention pooling over decoded
   content slots.

After the fix, at step 1,000:

```text
nce_loss=0.9198
mean_top10_cosine=0.5566
mean_random_pair_cosine=0.1465
test_r2_material=-0.0974
test_acc_side_to_move=0.4878
```

The collapsed run had `mean_top10_cosine=1.0000` and
`mean_random_pair_cosine=0.9997` at the same step.

## Run configuration

- Run directory: `data/models/run_perceiver6m` (fixed run)
- Failed run directories:
  - `data/models/run_perceiver6m_collapsed_*`
  - `data/models/run_perceiver6m_latentcollapse_*`
- Model: 5.96M parameters after the fix
- Architecture:
  - `d_model=256`
  - `n_heads=16`
  - `d_ff=832`
  - `n_layers=4`
  - `n_content_latents=16`
  - `n_state_latents=1`
  - `n_local_latent_layers=0`
  - `n_cross_segment_layers=1`
  - `n_decoder_layers=1`
  - `max_seq_len=2048`
  - `max_segments=30` (now resolved automatically from `max_seq_len`)
- Batch: 256
- Masking: 50% rectangle, 50% skip-gram
- Objectives: MLM + invariance + VICReg + InfoNCE

## Symptoms

At step 1,000 of the collapsed run, MLM and validation loss were improving, but
retrieval geometry was degenerate:

| Metric | Collapsed run | Fixed run |
|---|---:|---:|
| `nce_loss` | ~6.32 | 0.9198 |
| `mean_top10_cosine` | 1.0000 | 0.5566 |
| `mean_random_pair_cosine` | 0.9997 | 0.1465 |
| `cos_to_random_position` | 0.9997 | 0.1486 |
| `test_r2_material` | -0.0168 | -0.0974 |
| `test_acc_side_to_move` | 0.5463 | 0.4878 |
| `pc1_piece_corr` | 0.3278 | 0.6275 |

The collapsed run’s NCE was not merely flat; it was slightly **worse than
chance**. All content embeddings had nearly identical normalized directions, so
positives were no closer than negatives.

## Diagnostic chain

The collapse was traced by running a forward pass and measuring the content
representation after each stage. The values below are from the step-1,000
collapsed checkpoint on a batch of 64 boards.

| Stage | Mean off-diagonal cosine | Mean norm |
|---|---:|---:|
| Content queries | 0.1806 | 0.33 |
| After content cross-attention | 0.8332 | 11.26 |
| After local latent layers | 0.8332 | 11.26 |
| After cross-segment layer | 0.9842 | 72.73 |
| After latent readout | 0.9999 | 54.03 |
| Decoded content-slot mean | 0.8769 | 16.55 |
| Decoded state-slot mean | 0.4791 | 23.08 |

The decisive jump is the cross-segment layer:

```text
cosine 0.833 -> 0.984
norm    11.26 -> 72.73
```

The latent segment embedding had mean norm **15.95**, while the learned content
queries had mean norm **0.33**. Adding the segment embedding before the
cross-segment residual/FFN layer injected a large shared vector. The layer then
amplified that shared component relative to the board-specific component.

The content readout’s shared query added a second common component, but removing
it alone did not fix the collapse. A diagnostic that subtracted the readout query
still produced mean off-diagonal cosine `0.9999`. The dominant cause was the
latent segment embedding plus cross-segment amplification. The decoded content
slots were less collapsed (`0.8769`) and were therefore used for the new readout.

## Why the objective did not prevent collapse

The objective was not bypassed by a code bug. The loss terms were correct, but
they admit a degenerate solution in the **mean/common-vector direction**.

### InfoNCE

`info_nce_loss` normalizes anchors and positives and applies an in-batch softmax.
If every embedding is

```text
e_i = c + small_residual_i
```

with a large shared vector `c`, then all pairwise similarities are approximately
equal. The logits become nearly uniform, the loss approaches `ln(batch_size)`,
and the gradient with respect to `c` goes to zero. InfoNCE can organize the small
residuals, but it cannot penalize the common component. With `nce_weight=0.1`, it
also had little influence compared with MLM and invariance.

### VICReg

VICReg centers the embeddings before computing variance and covariance:

```text
z = embeddings - mean(embeddings)
```

Centering removes `c` completely. The variance and covariance terms therefore saw
only the small residuals, which still had healthy per-dimension variance and low
off-diagonal covariance. The collapsed run had `var_loss` near `4e-4` and
`cov_loss` near `0.07`. VICReg has no penalty on a large mean or a common
direction.

### Invariance

`alignment_loss` compares the cosine similarity between the window view and the
alone view of the same position. Collapse makes those views maximally similar, so
`inv_loss` goes to zero. The invariance objective therefore rewards the shared
component when it is not paired with a true uniformity constraint.

### MLM

MLM is a per-token objective. The decoder can learn to reconstruct masked tokens
from the content latents even when the pooled content readout is nearly constant.
The retrieval head is not required for the dominant loss, so it was free to
collapse without hurting MLM.

In short:

```text
large shared content vector
  -> invariance loss becomes easy
  -> VICReg sees only the centered residual and is satisfied
  -> InfoNCE cannot penalize the shared vector
  -> MLM trains through the decoder independently
  -> retrieval embedding is constant
```

## Fix

### 1. Small latent segment embedding

`src/modeling/perceiver/latent.py` now initializes the latent segment embedding
with `std=0.02` instead of the `nn.Embedding` default `N(0, 1)`:

```python
self.segment_embedding = nn.Embedding(config.max_segments, config.d_model)
nn.init.normal_(self.segment_embedding.weight, mean=0.0, std=0.02)
```

The initialized norm dropped from ~15.95 to ~0.32. The model can still learn
segment order, but it no longer starts with a late shared vector that dominates
the content latents.

### 2. Decoded-slot attention readout

The retrieval embedding no longer comes from a shared learned query added to the
content latents. `src/modeling/perceiver/model.py` now pools **decoded content
slots** with a learned attention pool:

- `SlotAttentionPool` computes a softmax over valid content/state slots and
  returns a convex combination of decoded slot states.
- Content slots are masked to visible content tokens, matching the previous
  content-pooling semantics.
- State slots are pooled separately from state latents.
- Content and state banks remain isolated.

The readout therefore no longer adds a shared query vector to every segment.

## Validation after the fix

NCE on the restarted run:

| Step | NCE |
|---:|---:|
| 20 | 6.3064 |
| 140 | 6.1331 |
| 300 | 5.3840 |
| 320 | 2.9850 |
| 360 | 1.7028 |
| 680 | 0.9780 |
| 1,000 | 0.9198 |

At step 1,000 the cosine diagnostics returned to a healthy range:

```text
mean_top10_cosine=0.5566
mean_random_pair_cosine=0.1465
cos_to_move_child=0.9318
cos_to_second_child=0.8589
cos_to_corrupted=0.9205
cos_to_random_position=0.1486
```

## Prevention / monitoring

Add collapse checks to training evaluation:

- Warn if `mean_top10_cosine > 0.95`.
- Warn if `mean_random_pair_cosine > 0.8`.
- Warn if `nce_loss > ln(batch_size)` after the warmup period.
- Track the mean norm of the content embedding in addition to centered variance;
  VICReg cannot detect a large shared offset.
- Track the effective rank of normalized embeddings, not only centered
  embeddings.
- Treat NCE near chance as a failure signal even when MLM and validation loss are
  improving.

Architectural guidelines:

- Avoid adding large learned embeddings late in the residual stream; initialize
  position/segment embeddings small or normalize them.
- Avoid shared additive readout queries for retrieval embeddings. Prefer pooling
  over token/slot states or a non-residual attention pool.
- Do not rely on invariance alone to shape retrieval geometry; it is minimized by
  collapse. Pair it with metrics or losses that constrain the mean/uniformity.
- Keep a retrieval-specific diagnostic path independent of MLM, since MLM can
  remain healthy while the retrieval embedding collapses.

## Code references

- `src/modeling/perceiver/latent.py`
  - `LatentBottleneck.segment_embedding` initialization
  - `SlotAttentionPool`
- `src/modeling/perceiver/model.py`
  - `_views_from_decoded`
  - `encode_views`, `encode_views_alone`, `embed`
- `src/training/objectives.py`
  - `compute_training_loss` reuses views from the main forward pass
- `src/evaluation/eval_suite.py`
  - cosine, material, side-to-move, and phase probes used above

## Artifacts

- Collapsed checkpoints/logs: `data/models/run_perceiver6m_collapsed_*`,
  `data/models/run_perceiver6m_latentcollapse_*`
- Fixed run: `data/models/run_perceiver6m`
- Fixed-run log: `data/models/run_perceiver6m/train.log`
- Evaluation history: `data/models/run_perceiver6m/eval_history.jsonl`
