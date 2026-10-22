# Design: Puzzle Theme Classification

- **Status:** Implemented
- **Date:** 2026-10-21
- **Scope:** Binary classification of a chess position sequence for a puzzle
  theme (`sacrifice` as the first target), using a frozen foundation encoder
  plus a trainable sequence head.
- **Out of scope:** New token encodings, foundation-model training, span-level
  theme annotations, and multi-theme/multi-task training.

---

## 1. Context

The puzzle dataset produced by `generate_puzzle_dataset.py` has one row per
Lichess puzzle:

| Column | Meaning |
|---|---|
| `puzzle_id` | Lichess puzzle id |
| `previous_position` | Encoded FEN position before the puzzle's first move |
| `packed` | `CLS` + one 67-token segment per position after each move + `SEP` |
| `n_positions` | Number of positions in `packed` (number of moves) |
| `rating` | Puzzle rating |
| `<theme>` | One `int8` column per theme, 0 or 1 |

For `sacrifice`, the full Lichess dump contains 6,157,341 puzzles, of which
464,256 are positive (7.54%). Position sequences are short: mean 4.63 moves,
99th percentile 10, maximum 34.

The task is: given a position sequence, predict whether the theme is present.
The semantic unit is the entire supplied sequence; the label applies to all
positions in it. The model must also work on the deployment input described
below: a single position, or a short history window ending at a position.

`previous_position` is deliberately **not** used as model input. It is the
position before the puzzle starts, and the sequence itself is the unit of
interest.

The foundation model is `MaskedStreamTransformer`, trained with masked stream
modeling on the current 52-token, 67-token-per-board representation. The
classifier uses:

```
data/models/run_d512l8_20261021_streaming/best_step40000.pt
```

with `d_model=512`, `n_heads=16`, `n_layers=8`, `d_ff=2048`,
`max_seq_len=4096`, `max_segments=128`, and slot embeddings.

---

## 2. Decision

### 2.1 Architecture

```
position sequence
    -> TokenStreamEncoder (67 tokens per position)
    -> CLS + segments + SEP
    -> frozen MaskedStreamTransformer
    -> per-position embeddings (model.embed, mean over each 67-token segment)
    -> attention pooling over positions in the sampled sequence
       (learned query; mean pooling available via `--pooling mean`)
    -> Linear(d_model, hidden_dim) -> GELU -> Dropout
    -> Linear(hidden_dim, 1)
    -> sigmoid -> P(theme present)
```

The encoder is frozen and kept in evaluation mode. Only the head is trained.
Pooling is masked: padded positions and structural tokens never contribute.
The default is attention pooling: a learned `d_model` query scores each
position (`h · q / sqrt(d_model)`), a masked softmax normalizes the scores, and
the sequence vector is the weighted sum of the position embeddings. The query
is zero-initialized, so training starts at uniform (mean) attention. Mean
pooling remains available via `--pooling mean` for the ablation. A fully
masked row pools to zero. The head has one hidden layer with GELU activation;
its width is set by `--head-hidden-dim` (default 256). Passing `hidden_dim=None` in the
config restores the single-`Linear` form so checkpoints saved before the
hidden layer remain loadable. The head has no LayerNorm, so `d_model=1` is
well-defined for tests. Dropout is configurable and defaults to 0 and is
applied before the projection and before the final classifier.

The head is parameterized by `d_model`, so it works with any foundation
checkpoint whose embedding width differs from 512.

### 2.2 Why a sequence classifier rather than a span head

The label is sequence-level, and the deployment input is "input a position".
A per-position classifier is obtained by feeding a length-1 sequence
(`CLS + one segment + SEP`) or a short window of preceding positions. This
directly matches deployment and handles multiple or disjoint theme
occurrences. The model never emits per-position labels during training; mean
pooling produces one sequence vector and one logit.

A start/stop span head was considered and rejected; see §5.

### 2.3 Loss and metrics

- Loss: `BCEWithLogitsLoss`, evaluated on one logit per sampled sequence.
- Optional `pos_weight` is available but is not used by default because
  training plans are 50/50.
- `--loss focal` opts into binary focal loss
  (`src/training/losses.py`). When `--focal-gamma`/`--focal-alpha` are
  omitted they are calibrated from the training label distribution:
  `alpha = 1 - prevalence` balances the expected per-class contribution and
  `gamma = min(2, log2(imbalance_ratio))` maps imbalance severity onto the
  usual 0-2 range. Validation loss always remains BCE so values stay
  comparable across runs and loss settings.
- Metrics: AUROC, PR-AUC (average precision), F1/precision/recall/accuracy at
  the validation-optimal threshold. PR-AUC is the primary metric because the
  natural prevalence is low and accuracy is misleading.
- Validation scores every single position once, preserving natural class
  prevalence.

---

## 3. Data sampling

### 3.1 Variant space

A puzzle's stored sequence has `n` positions. Its **variant space** is every
contiguous subsequence:

```
(start, end) with 0 <= start <= end < n
```

There are `n(n+1)/2` variants. Single positions and the whole sequence are
members of this space, so there are no duplicate configurations by
construction.

The theme label applies to every position in the puzzle sequence, so every
variant of a positive puzzle is positive and every variant of a negative
puzzle is negative. Sampling variants is therefore a label-preserving
augmentation and an upsampling mechanism for the minority class.

For the full `sacrifice` corpus the positive variant space is 10,143,193
configurations.

### 3.2 Train/validation split

The split is deterministic and independent of run order:

```
sha1(puzzle_id)[:8] / 2^64 < val_ratio  ->  validation
```

Default `val_ratio=0.1`. No puzzle appears in both splits.

### 3.3 Epoch plan

Each epoch is materialized as a compact plan before training:

```
puzzle_index : int32
start        : int16   (inclusive)
end          : int16   (inclusive)
label        : int8
```

- **Negative side:** every negative puzzle is used exactly once per epoch,
  with one variant chosen by its cursor.
- **Positive side:** `n_pos = round(n_neg * p / (1-p))` unique variants are
  drawn from the positive variant space. With the default `p=0.5`,
  `n_pos = n_neg`.
- The combined plan is shuffled once. There is no per-batch resampling.
- The dataset slices the selected segments and packs them on the fly:
  `CLS + segment_1 + SEP + ... + segment_k + SEP`.
- `collate_variants` pads tokens to the batch maximum, computes `segment_ids`
  with `cumsum(tokens == SEP)`, and builds the padding mask and labels.

### 3.4 Rolling cursor: no repeats until exhaustion

`RollingVariantPlanner` is stateful across epochs:

- **Positives** draw from a global permutation of the positive variant space.
  The cursor advances; when the permutation is exhausted, a new cycle starts
  with a fresh permutation.
- **Negatives** use a per-puzzle cursor over that puzzle's variants. The
  variant order is a seeded permutation generated with a coprime multiplier
  and offset, so each cycle covers every variant exactly once. When a puzzle's
  cursor wraps, its cycle counter advances and a new permutation is used.
- The planner state (positive cursor/cycle, negative cursors/cycles, epoch) is
  saved in every classifier checkpoint.
- No `(puzzle, start, end)` configuration is repeated until the relevant
  variant space is exhausted.

### 3.5 Why a precomputed plan

Sampling one variant per batch would:

- require repeated random draws and risk duplicate configurations;
- make batch composition non-reproducible;
- add per-batch CPU work before the model sees data.

The plan avoids all three. Cost is small: about 9 bytes per sample, roughly
100 MB at 50/50 over the full corpus, plus the positive permutation and
negative cursor arrays.

### 3.6 Current limitations

`load_puzzle_records` loads all selected parquet rows into memory. This is
acceptable for samples and moderate subsets, but the full 6.1M-puzzle corpus
(~5 GB packed) may need a shard-streaming or index-backed store. That is a
storage-layer change, not an algorithmic one.

---

## 4. Inference

### 4.1 Single position

```
FEN -> TokenStreamEncoder -> one 67-token segment
    -> CLS + segment + SEP
    -> frozen encoder -> pooled sequence vector (attention by default)
    -> two-layer head -> sigmoid -> probability
```

This is the length-1 case of the training input.

### 4.2 Sliding window over a game

For each position in a game, the inference path builds a window of the last
`k` positions ending at that position (`k=1` means the position alone). The
window is encoded and scored exactly like a training variant.

`classify_position.py` supports:

```bash
# single FEN
uv run python -m src.run.classify_position \
  --checkpoint data/models/puzzle_classifier/best.pt \
  --fen "r6k/pp2r2p/4Rp1Q/3p4/8/1N1P2R1/PqP2bPP/7K b - - 0 24"

# every position in a PGN, with four-position history
uv run python -m src.run.classify_position \
  --checkpoint data/models/puzzle_classifier/best.pt \
  --pgn game.pgn --window 4 --threshold 0.5
```

PGN output includes the start position (`ply=0`) and every mainline move with
`ply`, `move`, `fen`, `probability`, and optionally `is_theme`.

Per-position probabilities can be post-processed into spans by thresholding
and grouping contiguous positives. This is a decoding step, not a model head.
It is not yet implemented as a separate utility.

### 4.3 Checkpoint format

Classifier checkpoints contain:

```python
{
    "encoder_config": ...,
    "encoder_state": ...,
    "head_config": ...,
    "head_state": ...,
    "planner_state": ...,
    "epoch": ...,
    "metrics": ...,
}
```

Inference loads the encoder and head strictly; no architecture compatibility
shims are used.

---

## 5. Alternatives considered and discarded

### 5.1 Start/stop span head over a whole game

**Idea:** encode a full game and predict `(start, stop)` positions for the
theme, or `(-1, -1)` when absent. One forward pass per game; contiguous output.

**Why rejected:**

- The available labels are puzzle-level, not move-level. The exact sacrifice
  position is not annotated. Training a span head would require inventing span
  labels.
- It assumes at most one contiguous occurrence. A game can contain multiple
  sacrifice motifs.
- Full games exceed `max_seq_len=4096` (roughly 59 positions), so chunking is
  still required, which complicates span labels and decoding.
- Start/end decoding (`start <= stop`, at most one span) adds complexity and
  can be unstable.

The per-position classifier is more robust, directly matches deployment, and
handles multiple occurrences. Spans can be derived afterwards by thresholding.

### 5.2 Sequence-level classifier without subsequence augmentation

**Idea:** train on the whole puzzle sequence only, then feed single positions
at inference.

**Why rejected:** this creates a train/deployment distribution shift. Training
would only ever see multi-position sequences, while deployment is often a
single position. Subsequence sampling includes singles, short windows, and
whole sequences, so the model sees the deployment input during training.

### 5.3 First-position-only classifier

**Idea:** classify only the position after the first move, which is often where
the motif is played.

**Why rejected:** it discards the rest of the sequence, contradicts the
sequence-level semantics, and cannot detect motifs that appear later.

### 5.4 Attention pooling

**Idea:** learn a query over position embeddings and pool with attention.

**Status:** adopted after the mean-pooling baseline underfit (loss stayed near
the base rate). The implementation uses a single learned query vector (only
`d_model` extra parameters), scaled dot-product scores, and a masked softmax.
Zero initialization reproduces uniform/mean attention at step 0, limiting the
overfitting risk that motivated the original rejection. Mean pooling remains
available through `--pooling mean` for a controlled ablation.

### 5.5 CLS-token pooling

**Idea:** use the `CLS` hidden state as the sequence vector.

**Why rejected:** the foundation model was trained with masked-token modeling,
not a sequence-level CLS objective. `embed()` excludes `CLS` from its pooling,
and there is no guarantee that the CLS state summarizes the sequence.

### 5.6 Mean + max pooling

**Idea:** concatenate mean and max over positions.

**Why rejected:** doubles the head input, and max pooling is sensitive to a
single noisy position. The extra capacity was not justified for the initial
baseline.

### 5.7 Full fine-tuning or unfreezing the last N layers

**Idea:** train the encoder jointly with the head, or unfreeze the top blocks.

**Why rejected for now:** frozen features are fast, stable, and avoid
catastrophic forgetting on a small downstream dataset. The frozen encoder is
the "foundation model" premise of the project. Fine-tuning remains a future
experiment.

### 5.8 Positive weighting and focal loss

**Idea:** counter the 7.5% positive prevalence with `pos_weight`, focal loss, or
weighted sampling.

**Status:** BCE remains the default because the rolling plan already produces
50/50 training batches, so plain BCE is balanced in expectation. Both
`--pos-weight` and `--loss focal` are exposed as opt-in experiments. Focal
parameters are calibrated from the training label distribution unless given
explicitly. Validation loss stays BCE for comparability. AUROC and PR-AUC are
reported so imbalance-sensitive behavior is visible.

### 5.9 Enumerating every positive variant per epoch

**Idea:** emit every single position, every contiguous subsequence, and the
whole sequence for every positive puzzle each epoch.

**Why rejected:** the full enumeration is large (10.1M positive variants for
`sacrifice`), makes epochs expensive and uneven, and repeatedly uses the same
configurations. The rolling cursor draws unique variants and covers the space
over multiple cycles instead.

### 5.10 Using `previous_position` as input

**Idea:** prepend the FEN position stored in `previous_position`.

**Why rejected:** the classifier's unit is the sequence itself. The previous
position is outside that sequence and would change the input distribution.
The column remains available in the dataset for other tasks.

### 5.11 Materializing tokenized samples or precomputed embeddings

**Idea:** precompute every variant's tokens or frozen position embeddings to
make training I/O-free.

**Why rejected:** the plan already avoids repeated sampling, and on-the-fly
packing is cheap. A materialized token set is several GB and less flexible;
precomputed embeddings are much larger (`d_model=512` per position) and would
make the head-only training memory-bound. On-the-fly encoding keeps the
pipeline simple.

### 5.12 Multi-class theme classification

**Idea:** one classifier over all 73 themes.

**Why rejected for now:** themes are multi-label and imbalanced; a separate
binary head per selected theme is simpler to train, evaluate, and calibrate.
The architecture and sampling machinery are theme-agnostic (`--theme`), so
multiple binary models are straightforward.

---

## 6. Implementation map

| Path | Responsibility |
|---|---|
| `src/dataset/puzzle_variants.py` | Variant space, `RollingVariantPlanner`, epoch plans, split, dataset, collate, parquet loader |
| `src/modeling/puzzle_classifier.py` | `PoolingClassifier`, `FrozenEncoderClassifier` |
| `src/training/losses.py` | Focal loss and label-distribution calibration |
| `src/training/puzzle_metrics.py` | AUROC, PR-AUC, threshold metrics |
| `src/run/train_puzzle_classifier.py` | Frozen-encoder training CLI and checkpointing |
| `src/run/classify_position.py` | FEN and per-position PGN inference |
| `test/unit/test_puzzle_variants.py` | Variant space, rolling cursor, plans, dataset, collate, split |
| `test/unit/test_puzzle_classifier.py` | Mean/attention pooling, masking, `d_model` parameterization, frozen encoder |
| `test/unit/test_focal_loss.py` | Focal loss values, reductions, calibration, overrides |
| `test/unit/test_puzzle_metrics.py` | Ranking and threshold metrics |
| `test/unit/test_train_puzzle_classifier.py` | Loader, validation plan, training smoke test |
| `test/unit/test_classify_position.py` | FEN, window, and PGN inference |

### Training

```bash
uv run python -m src.run.train_puzzle_classifier \
  --data-dir data/processed/puzzles \
  --theme sacrifice \
  --checkpoint data/models/run_d512l8_20261021_streaming/best_step40000.pt \
  --output-dir data/models/puzzle_classifier \
  --epochs 3 --batch-size 64 --positive-fraction 0.5 --val-ratio 0.1
```

Outputs `last.pt` after every epoch and `best.pt` on the best validation
PR-AUC.

---

## 7. Consequences

**Positive**

- Directly trainable from the puzzle-level labels already produced by the
  dataset pipeline.
- Deployment matches training: single positions and short windows are valid
  variants.
- Rolling plans avoid duplicate configurations and per-batch sampling cost.
- Frozen encoder makes head-only training inexpensive and stable.
- The head is parameterized by `d_model` and requires no architecture-specific
  changes for other foundation checkpoints.

**Negative / open items**

- In-memory record loading does not scale to the full corpus without a
  streaming/indexed datastore.
- The model is a detector, not a precise localizer; spans are derived by
  thresholding.
- One model is trained per theme; multi-theme training is future work.
- The foundation checkpoint must use the current token encoding and include
  slot embeddings. Legacy checkpoints are not supported.
- Per-position validation is natural-prevalence but not yet evaluated on
  full games; the PGN inference path supports such an evaluation.
- Threshold selection currently maximizes F1 on validation; a deployment
  precision/recall target may be preferable depending on the application.

---

## 8. Future work

1. Add a shard-streaming record store for full-corpus training.
2. Add span post-processing (threshold + contiguous grouping) and span-level
   metrics for game scanning.
3. Compare the attention head against mean pooling and full fine-tuning.
4. Evaluate per-position precision/recall on held-out full games.
5. Add probability calibration if decisions need calibrated scores.
6. Extend to multi-theme or multi-task heads once the binary baseline is
   established.
