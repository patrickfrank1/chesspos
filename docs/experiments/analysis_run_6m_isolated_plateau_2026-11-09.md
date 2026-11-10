# run_6m_isolated plateau analysis (2026-11-09)

## Run summary

- Run: `run_6m_isolated`
- Model: 8 layers, `d_model=256`, 16 heads, `d_ff=1024`, 6.40M parameters
- Batch: 128
- Objective: MLM + content-isolated InfoNCE + invariance + VICReg
- Masking: 50% rect, 50% skip-gram
- Stopped at step **72,000** of 200,000
- Checkpoint: `hf://buckets/patrickfrank1/chesspos-checkpoints/run_6m_isolated/last.pt`
- Best checkpoint: `hf://buckets/patrickfrank1/chesspos-checkpoints/run_6m_isolated/best.pt`

The run was stopped because semantic evaluation metrics had plateaued while training
loss was still creeping down. The GPU pod was destroyed after syncing and verifying
the checkpoint in the Hugging Face bucket.

## Training loss behavior

Loss components remained nearly flat after roughly 30k steps:

| Step | Total | MLM | NCE | Inv | Var | Cov | Alone cov |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 30k | 0.513 | 0.201 | 0.435 | 0.132 | 0.0009 | 0.182 | 1.518 |
| 50k | 0.486 | 0.183 | 0.421 | 0.128 | 0.0009 | 0.173 | 1.492 |
| 70k | 0.471 | 0.173 | 0.415 | 0.127 | 0.0008 | 0.169 | 1.474 |

Validation loss was noisy but flat: roughly 0.464–0.500 from 64k–71k. The best
validation checkpoint was at 69k (`val_loss=0.4646`). There was no overfitting:
train and validation loss were close.

The NCE fix from step 11k worked as intended: NCE dropped from ~3.55 to ~0.5
immediately and invariance dropped from ~0.32 to ~0.15. That did not improve the
semantic probes, which is the important signal.

## Evaluation plateau

Key metrics from `eval_history.jsonl`:

| Step | Material R² | Side-to-move acc | PC1 piece corr | Top-5 material delta |
|---:|---:|---:|---:|---:|
| 1k | -0.13 | 0.527 | 0.515 | 1.939 |
| 13k | -0.37 | 0.459 | 0.582 | 1.910 |
| 30k | -0.44 | 0.542 | 0.591 | 1.929 |
| 50k | -0.39 | 0.522 | 0.614 | 1.895 |
| 70k | -0.54 | 0.502 | 0.594 | 1.905 |

Material R² **worsened** over training. Top-5 material delta stayed near the
random-pair delta (`random_pair_material_delta=2.229`), so retrieval was barely
material-aware. Phase/piece-count information was captured (`pc1_piece_corr~0.59`),
but signed material balance was not.

## Layerwise probe

I probed the final checkpoint at several depths using the same validation pool
(1,024 segments, 2,048 validation games). Embeddings were L2-normalized and scored
with the same ridge material probe used by the eval suite.

| Pool | Layer 1 | Layer 2 | Layer 3 | Layer 4 | Layer 6 | Layer 8 |
|---|---:|---:|---:|---:|---:|---:|
| Content mean | **0.57** | 0.45 | 0.08 | -0.10 | -0.40 | **-0.56** |
| Piece mean | **0.67** | 0.44 | 0.06 | -0.07 | -0.28 | **-0.42** |
| CLS | 0.26 | 0.29 | 0.31 | 0.11 | -0.26 | **-0.34** |
| State mean | 0.35 | 0.32 | 0.23 | 0.11 | -0.23 | -0.32 |

Material is linearly present in layers 1–2 and is gone by layers 4–8. A nonlinear
MLP probe on the final content mean also failed (R² -0.16 to -0.55 across seeds),
so this is not merely a weak linear probe.

## Root cause

`pool_content` averages only **visible, non-MASK content tokens**
(`src/modeling/transformer.py:498`). During training, only **masked** positions
receive token-level supervision through the MLM head. Visible positions are trained
as context for predicting masked tokens, not as descriptions of the token at that
position.

As a result, deeper layers progressively replace intrinsic piece identity at
visible positions with context features. The final content pool is therefore an
average of context features, not a summary of the board.

The strongest direct evidence: applying the MLM head to fully visible content
positions predicts their own token only **11.6%** of the time (piece tokens also
11.6%); chance for 13 content classes is ~7.7%. The final hidden states at visible
positions do not encode their own content.

NCE and invariance do not fix this because they align the same kind of visible
context features across the masked-window and alone views. They can make the two
views agree without forcing the pooled vector to preserve material.

Content isolation is not the cause. The problem is the lack of supervision on
visible content positions used by the pooling operation.

## Why the run looked "plateaued"

- MLM and validation loss were approaching their objective floor, not a numerical
  bug.
- The auxiliary losses were also near their floors.
- The representation was still slowly moving away from a linear material axis.
- Continuing to 200k steps was unlikely to recover material without changing the
  objective.

## Recommended next steps

1. **Visible-content identity loss (preferred).** Add a small auxiliary loss that
   makes the MLM head predict the original token from its own final hidden state at
   visible content positions. This gives visible positions a reason to preserve
   piece identity and should restore linear material.
2. **Content auxiliary heads.** Enable the existing material and piece-count heads
   on the pooled content embedding (`src/training/aux_heads.py`). This directly
   targets the failing probes but adds head parameters and requires a fresh or
   partially loaded checkpoint.
3. **Early/multi-layer pooling (fallback).** Material is strong in layers 1–2.
   Concatenating early-layer and final-layer pooled features would recover material
   but is a band-aid rather than a fix for the underlying supervision gap.
4. **Do not increase NCE weight as a fix.** NCE is already low and is not the
   bottleneck; the missing signal is token identity at visible positions.

## Artifacts

- Training log: `run_6m_isolated/train.log`
- Evaluation history: `run_6m_isolated/eval_history.jsonl`
- Latest checkpoint: step 72,000
- Best checkpoint: step 69,000 (`val_loss=0.4646`)
- Analysis scripts used locally:
  - `/tmp/opencode/probe_pooling.py`
  - `/tmp/opencode/probe_nonlinear.py`
  - `/tmp/opencode/probe_visible_identity.py`

These scripts are ad hoc and were not committed. The probe methodology is
reproducible with the checkpoint and the validation snapshot.
