# Plan: downstream evaluation of the two best foundation checkpoints

- **Status:** Complete — results in [`experiments/downstream_probes_2026-11-15.md`](experiments/downstream_probes_2026-11-15.md)
- **Date:** 2026-11-15
- **Scope:** Train frozen-encoder downstream heads on the two best foundation
  checkpoints and compare them on (a) sacrifice-theme classification and
  (b) position score regression (White POV, `[-1, +1]`).
- **Out of scope:** new foundation-model objectives, contrastive baselines,
  hard-negative mining, and the simplification plan in
  [`experiments/text_embedding_survey_and_simplification_2026-11-10.md`](experiments/text_embedding_survey_and_simplification_2026-11-10.md).

---

## 1. Checkpoints under test

| | `run_20m_lazy` | `run_6m_isolated` |
|---|---|---|
| File | `data/models/run_20m_lazy/last.pt` (step 120k) | `data/models/run_6m_isolated/best.pt` (step 69k; `best_step*.pt` 1k–69k also present) |
| Params | 21.18M | 6.40M |
| Shape | `d_model=512`, 16 heads, 8 layers, `d_ff=1536` | `d_model=256`, 16 heads, 8 layers, `d_ff=1024` |
| `content_pooling` | `False` | `True` |
| Objective | MLM + InfoNCE + invariance + VICReg | same + content-isolated views |
| Training data | 443 shards (2022 standard-rated + elite 2013–2020, 42 GB) | identical shard list |
| Val | `batch_0000` of every month 2022-01…2026-09 | identical |
| Token format | 67-token segments, vocab 52, `max_seq_len=2048`, `max_segments=128` | identical |

Both checkpoints load unchanged with the current `build_model` /
`model_config_from_dict` and the current `TokenStreamEncoder` (verified).

### Verified pooling asymmetry (important for the whole plan)

Changing only the side-to-move token on an otherwise identical board gives:

| Feature | `run_20m_lazy` | `run_6m_isolated` |
|---|---|---|
| `embed()` (native retrieval embedding) | L2 Δ = **31.9** (‖e‖ ≈ 22) | L2 Δ = **0.0** |
| `encode_content()` (`encode_views` view 1) | 0.0 | 0.0 |
| `encode_state()` (`encode_views` view 2) | 6.8 | 8.3 |
| `encode()` (full MLM trunk, all non-structural tokens) | 31.9 | turn-aware (state tokens pooled) |

Interpretation:

- `run_6m_isolated.embed()` is **exactly turn-blind by construction**
  (`content_isolation_mask` prevents content queries from attending to
  turn/castle/en-passant tokens). It cannot solve either task alone: sacrifice
  detection needs to know whose piece moved, and win probability needs whose
  turn it is.
- `run_20m_lazy.embed()` is state-aware because its config pools the full
  trunk (`content_pooling=False`), which was also its training path.
- `encode_views()`, the content+state recipe, was **never used during
  `run_20m_lazy` training** (`objectives.py` only calls it when
  `content_pooling or use_supcon`), so its isolated path is untrained for
  that checkpoint and cannot be the primary recipe.

This forces an explicit feature-recipe decision (section 5) instead of
"just use `embed()`".

---

## 2. Question and success criteria

Question: **how much downstream value can the frozen foundation checkpoints
provide today, and which of the two is the better starting point?**

Answer with a controlled comparison, not a single number:

1. Absolute task quality (PR-AUC for sacrifice; MSE/R² for score regression).
2. Advantage over cheap non-neural baselines (majority class; material + turn).
3. Gap between the two checkpoints, with seed variance so the gap is credible.
4. Failure slices (turn-blindness, material, phase, rating) that explain the gap
   and feed the next foundation-model decision.

"Good enough" is deliberately not pre-declared: the report states what each
model achieves and whether it beats the baselines.

---

## 3. Task A — sacrifice theme classification (existing design, new checkpoints)

Reuse [`design_puzzle_classification.md`](design_puzzle_classification.md)
unchanged in spirit:

- Data: Lichess puzzle corpus from `data/raw/lichess_db_puzzle.csv` (6.16M
  puzzles, 7.54% `sacrifice`), materialised locally to
  `data/processed/puzzles` with `src/run/generate_puzzle_dataset.py`.
- Unit: contiguous subsequence variants of a puzzle's position sequence
  (`RollingVariantPlanner`), 50/50 positive/negative epoch plans, single
  positions included so training matches deployment.
- Head: `PoolingClassifier` (attention pooling, one hidden layer, GELU,
  dropout 0) on frozen per-segment embeddings.
- Validation: every position of every validation puzzle once, natural
  prevalence; game/puzzle-level split by `sha1(puzzle_id)`.
- Metrics: PR-AUC (primary), AUROC, F1 at the validation-optimal threshold.

Changes required for this comparison:

1. `FrozenEncoderClassifier` currently always calls `encoder.embed()`. Add a
   `feature` option so the per-segment pooling can use the full MLM trunk
   (`encoder.encode()`) as well. Default stays `embed` (no behaviour change
   for existing checkpoints).
2. Run both checkpoints with the same head, plan seed, and hyperparameters.

Baselines: majority class (PR-AUC = prevalence), and mean-pooling ablation of
the same head.

There is **no recorded sacrifice-classifier result** in `docs/` or
`data/models/` (only `run_d512l8_20261021_streaming` checkpoints exist for
reference), so this comparison establishes the first number rather than
reproducing one.

---

## 4. Task B — position score regression

**Decision: regression, not classification.** The head predicts a bounded
score in `[-1, +1]` from White's perspective: `-1` = Black wins, `0` = draw,
`+1` = White wins. The realised game result is the label.

### 4.1 Data

- Source: `patrickfrank1/chess-positions` — 33.6M train games / 1.74M test
  games (44 GB parquet, game-level hash split, so no game appears in both).
  Each row carries `packed`, `n_positions`, `game_id`, `white_elo`,
  `black_elo`, `result`.
- The puzzle corpus has **no game score**, so this task uses the games dataset
  and cannot share rows with Task A. Only the frozen encoder and the head
  infrastructure are shared.
- Target: `score_from_result` → `"1-0"` `+1.0`, `"1/2-1/2"` `0.0`,
  `"0-1"` `-1.0` (White POV). Rows with an empty/unknown result are dropped.
- Position sampling: a fixed pool of games (section 6), plies up to a cap,
  one row per (game, ply).

### 4.2 Head and loss

```
tokens (CLS + 1 segment + SEP) or a window of k segments
    -> frozen encoder -> pooled vector (+ optional explicit state features)
    -> Linear(d, hidden) -> GELU -> Dropout -> Linear(hidden, 1)
    -> tanh -> score in [-1, 1] for White
```

- Loss: MSE on the bounded output (default); `--loss huber` as an ablation.
- Final layer is small-initialised so training starts near the base rate and
  does not begin inside the tanh saturation region.
- Because the target is White-POV, the head must consume side-to-move
  information; this is exactly where the turn-blind `run_6m_isolated.embed()`
  will fail (a reportable result, not a bug in the probe).
- Optional alternative arm: last-block fine-tuning (`--unfreeze-last-block`)
  to bound the ceiling.

### 4.3 Metrics and slices

- Primary: **MSE** and **R²** of the score; MAE.
- Secondary: Pearson and Spearman correlation; AUC on decisive games only
  (White wins vs Black wins, draws excluded); calibration over 10 bins of
  predicted score (bin mean prediction vs mean realised score).
- Slices: ply buckets (1–10, 11–30, 31–60, 61+), |material imbalance| buckets
  (0, 1–3, 4–6, 7+), pieces remaining (phase), Elo band, result class, and
  source family (elite vs standard).
- Symmetry test: colour-flipping a position and turning it must **negate** the
  White-POV prediction. Implemented as `flip_segment()` + a mean
  `|pred(p) + pred(flip(p))|` metric; also a unit property test
  `decode(flip(encode(b))) == b.mirror()`.
- Baselines: constant `0` (variance-only reference) and a ridge/logistic
  regression on `(material balance, turn, piece counts)` as the
  "no deep learning" reference; both fit on the same train plan.

### 4.4 Known noise

The label is the realised game result, not engine ground truth, so the ceiling
is below 1 and opening positions are near the base rate. A useful midgame
signal is expected to be modest; slices and the material baseline matter more
than one aggregate number.

---

## 5. Feature recipes

| Recipe | `run_20m_lazy` | `run_6m_isolated` | Trained in both? | Turn-aware? |
|---|---|---|---|---|
| **B. Native embed** — `encoder.embed()` | native (full trunk) | content-only | yes | 20m only |
| **A. Full-trunk pool** — `encoder.encode()` | = its native embed | MLM trunk (never used for retrieval) | yes | yes |
| C. Native embed + explicit state features (turn, castling mask, ep present) | native + 3 floats | content + 3 floats | yes | yes (via the bits) |
| D. `encode_views()` content+state concat | untrained isolated path | trained | no | yes |

**Primary: B (native `embed()`), as decided.** This is the model's intended
retrieval embedding and the honest "use the foundation model as-is" baseline.
The consequence is known and will be reported rather than hidden:
`run_6m_isolated` is turn-blind there, so its score regressor cannot do better
than the base rate on average.

**Alternative experiment: A (full-trunk pool).** The only recipe trained
end-to-end in both checkpoints and state-aware in both. If A is dramatically
better for `run_6m_isolated`, that quantifies exactly what content-only pooling
costs.

C is the deployment-friendly variant (native embed plus deterministic derived
state bits) and D is a `run_6m_isolated`-only reference. C and D are optional
and only run if the primary grid leaves the question open.

---

## 6. Protocol

- Frozen encoder in `eval()` mode; only heads train. This is the primary
  configuration. **Alternative arm:** last-block fine-tuning
  (`--unfreeze-last-block`, LR ≈ 1e-5) to bound the ceiling.
- Identical head width, dropout, LR, batch size, epochs, and sampling seed for
  both checkpoints; 3 seeds per (model, task, recipe) cell; mean ± std.
- Fixed probe pools, identical for both models (same games/puzzles, same plan
  seed):
  - Sacrifice: **the full 6.16M-puzzle corpus** (all negatives used once per
    epoch, positives drawn to 50/50 by `RollingVariantPlanner`); validate on
    every position of the held-out 10% of puzzles.
  - Score regression: **50k games** for train, **20k held-out games** from the
    HF `test` split for eval (stratified by source/Elo), all plies up to 150.
- Data hygiene: eval games/puzzles are never in the train pool; the HF `test`
  split was never seen by either checkpoint.
- Hyperparameter search is limited to a small LR/dropout grid on one seed for
  the primary recipe only; the rest use the fixed defaults.
- **Training length: one epoch with a chance guard.** Validation metrics are
  flat after the first epoch (20m: R² 0.0317 → 0.0317, AUC 0.5983 → 0.5985;
  6m: R² 0.0063, AUC 0.5791 with a slight drift down), so later runs train one
  epoch. `--abort-if-chance` stops any run whose AUC upper confidence bound
  (Hanley–McNeil, 95% one-sided) does not clear 0.5.

### Compute

No local GPU on the head node. Provision one cheap vast.ai box
(**< $0.5/h**, ≥24 GB, 4090/3090 class is sufficient for a frozen 20M model in
bf16) via the `vastai-gpu` skill, run everything there, sync
checkpoints/metrics/plots to
`hf://buckets/patrickfrank1/chesspos-checkpoints/downstream/`, destroy the box.

Estimated wall clock (frozen forward, bf16, batch 64–128):

| Step | One model |
|---|---|
| Puzzle dataset build (local, 16 CPU) | 20–40 min once |
| Sacrifice train 3 epochs on the full corpus + eval | 2–3 h |
| Score regression train 3 epochs + eval | 25–40 min |
| Alternative recipe A × 3 seeds | + the same again per recipe |
| Last-block fine-tune arm | + 2–4× the frozen time |

The primary grid (B, both models, both tasks, 3 seeds) is ≈ 5–7 GPU-hours on
one box; with recipe A and the fine-tune arm the total is ≈ 15–20 GPU-hours.

---

## 7. Implementation steps (tests first)

New modules, following repo conventions (BDD Given/When/Then, dataclass
configs, no ABCs unless pluggable):

| Path | Responsibility |
|---|---|
| `src/dataset/game_outcomes.py` | `decode_packed`; `GameOutcome` with result/Elo; `score_from_result`; `flip_segment`; `material_balance`; `build_score_plan`, `ScoreDataset`, `collate_score_samples`; local + remote-shard loaders; ply filtering; train/val split by `game_id` |
| `src/modeling/score_head.py` | bounded tanh score head with mean/attention pooling and optional explicit state features |
| `src/training/score_metrics.py` | MSE/MAE/R²/Pearson/Spearman, calibration bins, decisive-game AUC, material baseline |
| `src/run/train_score_probe.py` | CLI mirroring `train_puzzle_classifier.py`; checkpoints, `metrics.json`, `predictions.npz` |
| `src/modeling/puzzle_classifier.py` | `feature: Literal["native","full"]` on `FrozenEncoderClassifier`; shared pooling helper |
| `src/run/train_puzzle_classifier.py` | expose `--feature`; unchanged defaults |
| `src/dataset/token_stream.py` | move `_split_stream` here as public `split_stream` (used by both dataset loaders) |
| `scripts/eval/compare_downstream.py` | aggregate JSON → markdown comparison tables + calibration/score plots |
| `docs/experiments/downstream_probes_<date>.md` | sharable writeup with plots, tables, and the go/no-go recommendation |

Test-first order (per `AGENTS.md`):

1. `test/unit/test_game_outcomes.py` — result mapping, packed decoding,
   colour-flip round-trip against `Board.mirror()`, material balance, plan
   sampling/ply cap, dataset/collate shapes, parquet loader, split.
2. `test/unit/test_score_metrics.py` — known-value MSE/MAE/R²/Pearson/
   Spearman, perfect and constant predictions, calibration bins, decisive AUC.
3. `test/unit/test_score_head.py` — bounded output, pooling masking,
   explicit-state concat, gradient through a frozen encoder.
4. `test/unit/test_train_score_probe.py` — plan/loader smoke run, one
   optimizer step, checkpoint round-trip, CLI defaults.
5. Extend `test_puzzle_classifier.py` / `test_train_puzzle_classifier.py` for
   `feature="full"`.
6. `test/unit/test_token_stream.py` — `split_stream` after the move.

Then implement until green: `uv sync --extra cpu`, `uv run ruff format`,
`uv run ruff check`, `uv run pytest`.

Branch: `feature/downstream-probes`, PR into `main` (stacked on
`feature/practical-alternative-visible-content` so the survey link resolves).

### Implementation status (2026-11-15)

Done and green (`ruff format`, `ruff check`, full `pytest` — 778 passed):

- `src/dataset/game_outcomes.py`, `src/modeling/score_head.py`,
  `src/modeling/frozen.py`, `src/training/score_metrics.py`,
  `src/run/train_score_probe.py`.
- `FrozenEncoderClassifier(feature=...)` plus `--feature` in
  `train_puzzle_classifier.py`; `classify_position.py` honours the stored
  recipe.
- `split_stream` made public (used by both dataset loaders).
- Verified end-to-end on both real checkpoints on CPU: native and full recipes
  forward, backward, and predict.
- Two test-driven clarifications: plies are 1-based, and the last-block
  fine-tune arm unfreezes `blocks[-1]` + `ln_final` + head (and requires
  `feature="full"` because `embed()` runs under `torch.no_grad()`).

Finished: full puzzle corpus built (62 shards, `downstream/puzzles` in the
bucket), all runs completed on vast.ai (3 seeds for the frozen grid, seed 17
for the adapted arms), and the report with calibration, training curves,
slices, and the material baseline is written. The GPU box was destroyed after
the artifacts were verified in the bucket.

Headline: frozen AUC 0.587 (20m) vs 0.656 for a material baseline; last-block
fine-tuning 0.653; full fine-tuning **0.713** (20m) and **0.700** (6m). Turn
tokens and attention pooling are not the bottleneck; adaptation is.

Follow-up (2026-11-21): the sacrifice classifier now supports last-block and
full-encoder fine-tuning (`--unfreeze-last-block`, `--unfreeze-all`). On a 1M
puzzle subset, full fine-tuning lifts PR-AUC from ~0.20 to **0.455** (20m) and
**0.387** (6m); mean vs attention pooling is again irrelevant.

---

## 8. Risks and mitigations

1. **The 6m content embedding is turn-blind.** Quantified above; it is the
   expected headline finding for the native-`embed()` primary recipe, and
   recipe A/C quantify the cost.
2. **Native embed may understate `run_6m_isolated`.** Accepted by design; the
   alternative recipe A provides the ceiling comparison.
3. **Score regression on elite test games is draw-heavy** (~50% draws) and
   differs from the 2022 amateur training pool. Slices by source/Elo plus
   calibration make the shift visible; no metric is reported without the base
   rate.
4. **Result noise.** Realised outcomes are a noisy label; the constant-0 and
   material baselines bound how much signal is real.
5. **Puzzle sequences can exceed `max_seq_len=2048`** (34 positions × 68 = 2313).
   Variant sampling usually keeps inputs short; add an explicit length guard
   and count skipped variants in the log.
6. **Data volume / download cost.** Stream shards and cache only the sampled
   games; do not mirror 44 GB on the box.
7. **`run_20m_lazy` is a `last.pt` at 120k with no `best.pt`.** If a better
   checkpoint exists in the bucket, it should be used instead.
8. **Tanh saturation.** Small final-layer init and LR selection; monitor the
   prediction histogram in the first epoch.

---

## 9. Resolved decisions

| # | Decision |
|---|---|
| 1 | Score regression in `[-1, +1]`, White POV (`-1` Black wins, `0` draw, `+1` White wins), MSE loss |
| 2 | Primary feature recipe = native `encoder.embed()`; full-trunk `encode()` is the alternative experiment |
| 3 | Two datasets are fine: puzzles for sacrifice, games for score regression |
| 4 | Full 6.16M-puzzle corpus for sacrifice; fixed 50k/20k game pools for score regression |
| 5 | Independent heads / independent downstream models (no joint multi-task run) |
| 6 | Frozen encoder primary; last-block fine-tune as an alternative experiment |
| 7 | One vast.ai box at **< $0.5/h** |
| 8 | `run_20m_lazy/last.pt` (120k) and `run_6m_isolated/best.pt` (69k) |
| 9 | Sharable writeup with plots and evaluation tables in `docs/experiments/` |

### Added to the curriculum: token-level pooling ablation

The fine-tune arm pools the last per-token layer with a fixed **mean**. A mean
pool cannot reweight tokens as the last block adapts, so it conflates the
pooling operator with the adaptation of the block. The ablation crosses the two
pooling operators with the two freeze levels:

| | mean pooling | attention pooling |
|---|---|---|
| frozen encoder | existing `native` / `full` runs | **new** |
| last block fine-tuned | existing `ftlast` arm | **new** |

- Attention pool: one learned `d_model` query over the last token layer, a
  masked softmax per segment (structural tokens excluded; content-only for the
  content-isolated native recipe), zero-initialised so training starts at the
  mean. The query is a probe parameter and trains even when the encoder is
  frozen.
- A third freeze level is also measured: **full fine-tuning** (every encoder
  parameter, encoder LR 1e-5, head LR 1e-3), with mean pooling. It bounds how
  much of the gap is adaptation capacity versus the pooling operator.
- Runs: 4 pooling cells + 2 full fine-tuning cells, × 2 checkpoints × seed 17,
  one epoch each, same data and head.
- Metrics: held-out R²/AUC and calibration versus the mean cells and the
  material baseline.
- Implementation: `src/modeling/token_pool.py`, `--token-pooling
  mean|attention`, `MaskedStreamTransformer.token_states()`.

Remaining implementation-level details (not blocking, defaults proposed):
`positions_per_game=16`, `max_ply=150`, `val_ratio=0.1`, `head_hidden_dim=256`,
`dropout=0.0`, `lr=1e-3`, `batch_size=64`, 3 epochs, 3 seeds.
