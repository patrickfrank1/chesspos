# ADR-0001: 64-Slot Token Encoding for Chess Positions

- **Status:** Accepted
- **Date:** 2026-10-18
- **Supersedes:** the set-of-piece-square token format (vocab 806) used by
  `run_full_v2` and earlier

## Context

Positions are encoded as token streams for a masked-sequence transformer. The
original encoding represented a board as a *set of piece-square tokens*: each
occupied square contributed one token from a 12×64 = 768-entry joint
piece×square vocabulary (plus turn/castling/en-passant scalars), ordered by
square. Empty squares contributed nothing; absence of a token meant "empty".

Training the `run_full_v2` model exposed three structural problems:

1. **Mask localization is impossible.** A masked piece-square token is replaced
   by a single MASK id, and the model had no within-segment positional signal.
   All MASK tokens in a board received identical embeddings and — under
   bidirectional attention — identical outputs. The model therefore could not
   know *which square* it was predicting and capped at the unigram piece prior
   (measured: random-mode piece accuracy 0.341 vs prior 0.349 after 23k steps).
2. **"Empty" is not representable.** Roughly half the information of a chess
   position (which squares are vacant) exists only implicitly. Board/span
   reconstruction must infer vacancy from absence rather than classify it.
3. **Board constraints are inexpressible.** The joint softmax can output two
   pieces claiming one square; the 64-slot, one-piece-per-square structure of a
   chessboard is not enforced or exploited by the target parameterization.

Additionally, the set encoding cannot express partial observations where
*occupancy itself* is uncertain (e.g. "the pawn structure is known, the knights'
positions are not"): deleting tokens asserts emptiness, masking asserts
occupancy.

## Decision

Encode each board as a **fixed-length 67-token segment of 64 content slots plus
3 scalar slots**:

```
[64 slot tokens, a1..h8] [turn] [castle] [ep]
```

- Slot tokens are **content-only** (`EMPTY` or one of 12 piece types, 13
  values). Square identity comes from the slot index.
- Scalars are fixed slots: turn (2 values), castle (16 values), en passant
  (17 values: `EP_NONE` + 16 target squares). En passant is always present to
  keep segments fixed-length.
- Total vocabulary ≈ 52 ids (down from 806).
- The model adds a learned 1D within-segment positional embedding (67
  entries); occupied, empty, and masked slots all receive square identity from
  it. Masking replaces a slot's content with MASK while the slot index conveys
  the square.

The dataset repo is reprocessed from the PGN archives into this format; old
checkpoints and eval results are deleted, not migrated.

## Alternatives considered

- **`MASK@square` vocabulary (64 mask ids), keep the set encoding.** Smallest
  change; fixes mask localization. Rejected as the target representation
  because constraints stay inexpressible, empties stay implicit, and
  occupancy-uncertain states remain unrepresentable.
- **Square side-input** (per-position square ids added as embeddings). More
  general than `MASK@square` but changes the model's input signature and keeps
  the variable-length set semantics.
- **Within-segment positional embedding over the *old* set encoding.** Works
  only because piece tokens are emitted in sorted-square order behind a
  variable-length 2–3 token prefix; indirect and fragile.
- **2D (rank×file) factorized positional embedding.** Equivalent in
  expressiveness to a learned 1D embedding over 64 fixed squares; kept as a
  possible inductive-bias ablation, not part of this decision.
- **Hierarchical architecture** (local board encoder over the slot grid +
  cross-board attention). Attractive long-term (within-board permutation is
  handled locally, cheaper attention); deferred — this ADR changes the
  representation, not the transformer topology.

## Consequences

**Positive**

- Masked reconstruction is well-posed by construction: random, board, and span
  modes all reduce to "predict the content of known squares given context".
- Emptiness is a first-class prediction; space/mobility features become
  directly learnable.
- Pooled embeddings average over a fixed aligned basis, and per-slot hidden
  states are spatially aligned (hidden state *i* ↔ square *i*), so dense
  per-square downstream heads become trivial — the property a foundation model
  needs.
- Masking is a universal interface for partial observations, including
  occupancy-uncertain states (mask any slot subset; the head yields a full
  posterior incl. `EMPTY`).
- Output head shrinks (806 → ~52 logits); the 806-way piece-square prior that
  dominated v2 training no longer exists as a shortcut.

**Negative / costs**

- ~2.2× tokens per board (empties included): ~30 boards per 2048-token window,
  or ~59 at the chosen `max_seq_len=4096`, with proportionally higher
  attention cost.
- Empty dominance in the loss (P(empty) ≈ 0.5); mitigated by
  inverse-unigram-frequency weighting.
- Metrics are redefined (`slot_acc` / `occupied_acc` / `empty_acc` / board
  exact-match) and are not comparable to pre-migration runs.
- Old checkpoints, eval artifacts, and the published dataset are invalidated;
  a full ETL reprocess is required.
- Global legality (exactly one king per side, ≤16 pieces, pawn ranks) is still
  unenforced — accepted, as it is orthogonal to the representation-learning
  objective.
