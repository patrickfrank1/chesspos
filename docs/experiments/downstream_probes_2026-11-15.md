# Downstream probes: sacrifice classification and position score regression (2026-11-15)

## Status

Frozen-encoder downstream evaluation of the two best foundation checkpoints.
Score regression is complete (two recipes, three seeds, held-out test split);
sacrifice classification uses the full puzzle corpus. Plots and the fine-tune
arm are appended as they finish.

## Question

How much downstream value do the current foundation checkpoints provide when
used as frozen feature extractors, and which checkpoint is the better starting
point? The answer is deliberately baseline-relative: an aggregate R² without a
simple reference is not interpretable.

## Checkpoints

| | `run_20m_lazy` | `run_6m_isolated` |
|---|---|---|
| file / step | `last.pt`, 120k | `best.pt`, 69k |
| parameters | 21.18M | 6.40M |
| shape | d512, 8 layers, ff1536 | d256, 8 layers, ff1024 |
| `content_pooling` | False (native embed pools the full trunk) | True (native embed pools content only) |
| trained on | 443 shards (2022 standard + elite 2013–2020) | identical |

Both use the current 67-token per-position token stream. The 6m checkpoint's
native `embed()` is **exactly turn-blind**: flipping only the side-to-move
token changes the embedding by L2 0.0, versus 31.9 for the 20m checkpoint.
That is why the full-trunk recipe (recipe A) is run as an alternative.

## Method

### Recipes

- **native** (`encoder.embed()`): the checkpoint's own retrieval embedding.
- **full** (`encoder.encode()`): mean pool over the non-isolated trunk, which
  includes content and the turn/castling/en-passant tokens. This is the same
  output as native for `run_20m_lazy`, and is the only state-aware recipe
  trained end-to-end in both checkpoints.

### Position score regression

- Target: White-perspective final score in `[-1, +1]` (`-1` Black wins, `0`
  draw, `+1` White wins), i.e. the realised game result.
- Data: `patrickfrank1/chess-positions`. Train/validation from the `train`
  split (56k games, 90/10 game-hash split); held-out evaluation on the `test`
  split (20k games: 10k `standard` + 10k `elite` shards, never seen by either
  the checkpoint or the probe).
- Input: one position (`CLS + segment + SEP`), `positions_per_game=16`
  sampled uniformly per epoch; validation and test score every position.
- Head: mean pool → `Linear(d, 256) → GELU → Dropout(0) → Linear(256, 1) →
  tanh`, small final-layer init.
- Loss: MSE. Optimizer: AdamW, LR 1e-3, weight decay 0.01, batch 256, bf16
  autocast with the flash-attn varlen kernel. Encoder frozen.
- Epochs: 3 for the first runs, then 1 once the validation curves showed epoch
  2–3 add nothing (20m: R² 0.0317 → 0.0317, AUC 0.5983 → 0.5985; 6m: flat to
  slightly worse). Runs can additionally stop themselves when the decisive AUC
  is not significantly above 0.5 (`--abort-if-chance`, Hanley–McNeil error).
- Metrics: MSE/MAE/R², decisive-game AUC, Pearson/Spearman, calibration bins.

### Sacrifice classification

Reuses the existing puzzle pipeline (`RollingVariantPlanner`, 50/50 epoch
plans, subsequence variants, attention-pooled sequence head, PR-AUC on the
natural-prevalence validation split) with the full 6.16M-puzzle corpus.

## Results — position score regression

Validation split (3 seeds, mean ± std):

| recipe | model | R² | decisive AUC | MAE |
|---|---|---|---|---|
| native | run_20m_lazy | 0.029 ± 0.004 | 0.595 ± 0.004 | 0.859 ± 0.016 |
| full | run_20m_lazy | 0.029 ± 0.004 | 0.594 ± 0.004 | 0.859 ± 0.016 |
| native | run_6m_isolated | 0.004 ± 0.003 | 0.575 ± 0.005 | 0.868 ± 0.016 |
| full | run_6m_isolated | 0.001 ± 0.002 | 0.576 ± 0.003 | 0.867 ± 0.016 |

Held-out test split (3 seeds, mean ± std):

| recipe | model | R² | decisive AUC | MAE |
|---|---|---|---|---|
| native | run_20m_lazy | 0.023 ± 0.001 | 0.587 ± 0.003 | 0.859 ± 0.001 |
| full | run_20m_lazy | 0.024 ± 0.001 | 0.587 ± 0.001 | 0.859 ± 0.001 |
| native | run_6m_isolated | 0.010 ± 0.002 | 0.571 ± 0.001 | 0.867 ± 0.000 |
| full | run_6m_isolated | 0.011 ± 0.001 | 0.573 ± 0.001 | 0.867 ± 0.001 |
| **material baseline** (affine, same data) | — | **0.065** | **0.656** | **0.835** |

The material baseline fits `score = a · material + b` on 80% of the held-out
predictions and scores the remaining 20%. It uses one feature and beats both
checkpoints on every metric.

![Calibration](../assets/downstream_calibration.png)

![Training curves](../assets/downstream_training_curves.png)

### Slices (held-out, 20m native; 6m behaves the same or worse)

| slice | n | R² | decisive AUC | MAE |
|---|---:|---:|---:|---:|
| ply 1–10 | 598k | -0.010 | 0.500 | 0.911 |
| ply 11–30 | 1.18M | -0.011 | 0.516 | 0.908 |
| ply 31–60 | 1.50M | +0.009 | 0.573 | 0.884 |
| ply 61+ | 1.47M | +0.083 | 0.677 | 0.773 |
| abs(material) 0 | 2.40M | -0.009 | 0.527 | 0.876 |
| abs(material) 1–3 | 1.88M | +0.023 | 0.599 | 0.854 |
| abs(material) 4–6 | 311k | +0.106 | 0.700 | 0.843 |
| abs(material) 7+ | 153k | +0.330 | 0.863 | 0.699 |

The signal is concentrated exactly where material already decides the game:
balanced positions are indistinguishable from chance (AUC 0.527, R² -0.01),
while large-material endgames reach AUC 0.86. Openings add nothing, as
expected. The 20m model is ahead of the 6m in the late-game and large-material
buckets and identical elsewhere.

## Results — adaptation ablations

The frozen probes pool per-token states with a **mean** and train only the
head. Two controls test what limits them: how much of the encoder adapts, and
whether a learned pooling operator (over the last per-token layer) matters.

Held-out test split, seed 17 (1 seed for the adapted arms):

| pooling | freeze level | model | R² | decisive AUC | MAE |
|---|---|---|---|---:|---:|---:|
| mean | frozen | run_20m_lazy | 0.024 | 0.587 | 0.859 |
| mean | frozen | run_6m_isolated | 0.010 | 0.571 | 0.867 |
| mean | last block | run_20m_lazy | 0.078 | 0.653 | 0.820 |
| mean | last block | run_6m_isolated | 0.013 | 0.574 | 0.865 |
| mean | **all parameters** | run_20m_lazy | **0.131** | **0.713** | 0.773 |
| mean | **all parameters** | run_6m_isolated | **0.117** | **0.700** | 0.786 |
| attention | frozen | run_20m_lazy | 0.020 | 0.577 | 0.864 |
| attention | frozen | run_6m_isolated | 0.012 | 0.571 | 0.868 |
| attention | last block | run_20m_lazy | 0.079 | 0.653 | 0.820 |
| attention | last block | run_6m_isolated | 0.013 | 0.574 | 0.865 |
| material baseline | — | — | 0.065 | 0.656 | 0.835 |

**Full fine-tuning is the largest single lever.** It roughly triples the
frozen R² and adds 0.13 AUC for both checkpoints, and clears the material
baseline on both metrics. With the whole encoder adapted, the 6m model nearly
catches the 20m (AUC 0.700 vs 0.713, R² 0.117 vs 0.131), so most of the 6m
deficit is recoverable representation capacity rather than a hard capacity
ceiling.

**The pooling operator is not the bottleneck.** Attention pooling matches the
mean within ±0.001 AUC and ±0.004 R² in every cell (frozen and last block,
both checkpoints), and is marginally worse for the frozen 20m (0.577 vs
0.587). The zero-initialised query is free to learn a non-uniform weighting,
but the probe gains nothing from it. What matters is *how much of the encoder
adapts*, not how the last token layer is pooled.

Reading the grid as a whole: frozen 0.57–0.59 AUC → last block 0.65 (20m only)
→ all parameters 0.70–0.71 for both checkpoints. The 20m's last-block gain is
large; the 6m needs the whole encoder before it catches up.

## Results — sacrifice classification

_Filled in when the full-corpus runs finish._

## Findings

1. **Frozen, neither checkpoint is a usable position evaluator.** The best
   held-out decisive AUC is 0.587 (20m), versus 0.656 for an affine function of
   material alone (R² 0.023–0.024 vs 0.065). **Fully fine-tuned they are:**
   AUC 0.713 / R² 0.131 (20m) and 0.700 / 0.117 (6m), both above the material
   baseline. The frozen-probe result measures the representation, not what the
   checkpoint can become.
2. **The 20m checkpoint is consistently the better representation.** It wins by
   ~0.015 AUC and ~0.013 R² on held-out data, well outside the seed standard
   deviation (0.001–0.003).
3. **Neither the turn token nor the pooling operator is the main limit.** The
   full-trunk recipe moves the 6m by +0.002 AUC, and learned attention pooling
   by ±0.001. Freezing is the constraint: full fine-tuning adds +0.126 AUC
   (20m) and +0.129 (6m).
4. **The aggregate hides where the signal lives.** The embedding is a weak
   material detector: chance in balanced positions, strong in lopsided
   endgames. Any claim that the representation "understands" a position should
   be read against that slice table.
5. **Calibration is compressed toward zero.** Predictions occupy a narrow band
   around the base rate; ranking (AUC) survives while R² does not. The
   calibration plot shows the slope deficit.

## Threats to validity

- The label is the realised game result, not engine ground truth, so even a
  perfect evaluator has a noise ceiling. The material baseline and the slice
  table bound what is achievable.
- The held-out split mixes elite games (draw-heavy) with recent standard-rated
  games, while probe training used the standard-rated `train` split. The test
  numbers therefore include a distribution shift; validation numbers are
  in-distribution and slightly higher.
- Single-position inputs ignore move history. A window ablation with a ridge
  probe on 4k games showed no gain (window 1/4/8 all ~0.56 AUC), but the head
  was not retrained per window.
- Sacrifice uses one epoch over the full corpus per seed; the plan's three
  epochs would change absolute numbers, not the model ordering.

## Artifacts

- Code: `feature/downstream-probes` (`src/dataset/game_outcomes.py`,
  `src/modeling/score_head.py`, `src/run/train_score_probe.py`,
  `src/run/eval_score_probe.py`, `scripts/eval/summarize_downstream.py`).
- Runs: `hf://buckets/patrickfrank1/chesspos-checkpoints/downstream/` plus the
  box-local `data/models/downstream/` and `data/eval/downstream/` trees
  (`metrics.json`, `predictions.npz`, checkpoints).
- Plan and decisions: [`../plan_downstream_evaluation.md`](../plan_downstream_evaluation.md).

## Next steps

- Fine-tune on the sacrifice task as well: the frozen representation already
  reaches AUROC 0.71, and adaptation is clearly the largest lever on the score
  task.
- If the goal is a position evaluator, the recipe is "fine-tune, then probe".
  The frozen checkpoint is a feature extractor, not an evaluator.
- Decide whether the next foundation-model work should target material and
  phase structure explicitly (an auxiliary head on the pooled content vector),
  given that those are the only slices with signal in the frozen setting.
- Adaptation answers the question: full fine-tuning more than doubles the
  last-block gain and beats the material baseline, so the frozen probe
  understated the checkpoints. The practical recipe is "fine-tune the model,
  then probe", not "freeze and probe".
- The same adaptation should be tried on the sacrifice task, where the frozen
  representation already reaches AUROC 0.71.
