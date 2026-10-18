# Plan: 64-Slot Token Encoding Migration

Status: **in progress** (2026-10-18)

Switch the position encoding from set-of-piece-square tokens (vocab 806) to a
fixed 64-slot board representation. See
[ADR-0001](adr/0001-64-slot-token-encoding.md) for the rationale.

## Decisions

- 64 content slots (a1..h8) + 3 fixed scalar slots per segment: `[64 slots][turn][castle][ep]`,
  always 67 `int16` tokens.
- Content-only slot vocabulary (~52 ids): `EMPTY` or one of 12 piece types; square identity
  comes from the slot index, not the token.
- En passant becomes an always-present scalar slot with an explicit `EP_NONE` value (fixed
  segment length requires it; today the ep token is omitted when absent).
- Positional encoding: learned 1D within-segment embedding (67 entries). 1D over a fixed
  canonical layout is exactly as expressive as a rank/file factorization; the 2D variant is
  a later ablation, not a requirement.
- Masking is uniform over slots, **including empties** — emptiness becomes a first-class
  prediction. Empty dominance (P(empty) ≈ 0.5) is handled by the inverse-unigram-frequency
  loss weighting.
- Context length: `max_seq_len=4096` (~59 boards per window at 67 tokens + framing).
- Loss machinery from the 2026-10 masking overhaul carries over unchanged: balanced mode
  pattern (random/board/span), per-mode weights, inverse-frequency weights, weighted-mean
  loss.
- Old artifacts are **invalidated, not migrated**: dataset repo deleted and recreated;
  bucket prefix `run_full_v2` deleted; old checkpoints/eval results removed.

## Phases

### Phase 1 — Encoding (`src/dataset/token_stream.py`) — **done 2026-10-18**
- New vocabulary constants; `TokenStreamEncoder.encode/decode` emit/parse the 67-token
  slot format; roundtrip tests (start position, ep, promotions, castling masks, random
  games).
- All modules referencing the old constants updated so the suite stays green:
  `masking.py` (token class ranges), `transformer.py` (plain token embedding over the
  52-id vocab; positional embedding lands in Phase 4), `eval_suite.py` (scalar bounds,
  `token_prior`, corruption), `src/dataset/__init__.py` re-exports.
- Bonus fix: `_sample_few` now clamps few-mode windows to the token budget (previously
  latent; exposed by the longer slot segments).

### Phase 2 — ETL emits new format
- `etl.py` / `PGNProcessor` need no logic change (they call the encoder); batch metadata
  and dataset card text updated to describe the slot format.
- Dry-run one small archive, validate parquet output, then full reprocess.

### Phase 3 — Dataset reprocess (Ray job, local cluster)
- Repo `patrickfrank1/chess-positions` was deleted and recreated empty on 2026-10-18.
- Manifest of the 89 archives to reprocess (82 elite + 7 rated):
  `data/processed/dataset_manifest_v2.txt`.
- Run the archive pipeline with the same sampling tiers / time-control filter; shards
  push to the Hub incrementally (per-archive resume).

### Phase 4 — Model changes
- Within-segment positional embedding (67) + segment embedding; head over ~52 ids.
- Regression test: two MASK tokens at different slots of the same board must receive
  different predictions (the defect this representation kills by construction).
- Re-train as `run_slots_v1` with all loss improvements; upload prefix `run_slots_v1`.

### Phase 5 — Eval redefinition (`src/evaluation/eval_suite.py`)
- `slot_acc` (all masked slots), `occupied_acc` (12-way piece given occupied), `empty_acc`,
  board exact-match; scalar metrics unchanged in spirit.
- Neighbor/counterfactual/probes/PCA unchanged (they consume `embed()`).
- Optional: per-square occupancy probe to verify spatial alignment.

## Not in scope (later experiments)
- 2D (rank/file) positional ablation; occupied-only masking variant; hierarchical
  board-encoder + cross-board attention.
