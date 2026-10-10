# Idea: Staged Masking Curriculum

> Stashed 2026-10-10. Parked until the static-mixture baseline (see
> `src/run/train_masked_transformer.py`) has been trained and evaluated
> end to end.

## Proposal

Instead of a static mixture of masking modes
(`MaskingConfig`: 50% random / 25% board / 25% span), schedule the
mixture over training in three stages:

1. **Random-piece heavy** — mask mostly individual piece-square tokens
   on random boards. Rich context per prediction; teaches the model
   what *legal, consistent* positions look like.
2. **Board heavy** — mask mostly entire positions. Forces prediction of
   a whole board from its neighbours; teaches the semantics of
   neighbouring positions (game dynamics, transpositions).
3. **Span heavy** — mask mostly sequences of 2–4 consecutive boards.
   Longest-range objective; helps generalization to sequence-level
   embeddings.

Classic easy→hard curriculum: each stage's skill plausibly builds on
the previous one.

## Why we parked it

- **No baseline yet.** Curriculum gains in the MLM literature (masking
  rate / span-length schedules) are mixed and often within noise. The
  static-mixture run must exist first so the curriculum has something
  to beat.
- **Multi-task gradients likely help.** All three modes share one
  trunk and are seen every step. The final embedding quality comes
  from the *last* distribution seen; a span-heavy final stage may
  erode the piece-level precision that single-position kNN search
  depends on. A static mixture keeps every skill live at the end.
- **Cost.** Multiplies hyperparameters (stage boundaries, transition
  sharpness, LR per stage?) and requires per-mode eval bookkeeping —
  board/span losses are inherently higher than random-piece losses, so
  val loss is not comparable across stages.

## How to implement it later

Cheap: `MaskingConfig` probabilities are plain parameters. A schedule
is just `p_random / p_board / p_span` as functions of step in the
training loop; no architecture changes needed.

Prerequisites to add first:

- Log which masking mode each sample used (return it from
  `mask_stream`), and track **val loss per masking mode** so results
  are comparable across stages.
- Decide how stages transition (hard switch vs. linear interpolation
  of the mixture over N steps).
