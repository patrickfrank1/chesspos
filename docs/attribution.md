# Position attribution

Per-square explanations for the two downstream probes:

- **sacrifice classifier** — probability that the position belongs to a
  sacrifice puzzle.
- **score probe** — White-POV expected points in `[-1, +1]`.

The same code path serves both because both models read the same token stream
(`src/dataset/token_stream.py`): 64 square tokens, then turn, castling, and
en-passant. Only the output head and the optional window differ.

Implementation: [`src/chess_eval/attribution.py`](../src/chess_eval/attribution.py).

## Value function

Every method asks "what happens if this square's piece is not there?" and
replaces the square token with either:

| replacement | meaning |
| --- | --- |
| `empty` | the square is empty — a piece is removed |
| `mask` | the square's content is unknown (the `MASK` token) |

`mask` matches the encoder's BERT-style pretraining distribution; `empty` is
the cleaner chess counterfactual, and empty squares become exact no-ops.

Attribution signs are in output space:

- **positive** = the square raises the output (sacrifice probability up, or
  White-POV score up);
- **negative** = the square lowers it.

For the score probe, a Black piece with a positive value is one that helps
White.

## Occlusion

`occlusion` replaces one square at a time and reports

```text
delta(square) = f(board) - f(board with that square replaced)
```

All 64 variants are evaluated in one batched forward. Occlusion deltas are
**not additive** — interactions mean the parts do not sum to the whole.

## KernelSHAP

`shap` treats every **occupied** square as a player in a cooperative game whose
value is the model output with only the coalition's pieces kept. The baseline
is the empty board (same side to move, castling rights, and en-passant square,
but no pieces), so

```text
sum(phi) = f(board) - f(empty board)
```

exactly; the implementation shifts the least-squares solution to enforce it.

Empty squares are excluded from the game: calling an empty square present or
absent produces the same token, so its Shapley value is exactly zero. That also
halves the number of players and makes the regression much better determined.

Coalitions are sampled uniformly, weighted by the Shapley kernel
`(n-1) / (C(n,s)·s·(n-s))`, and fit with weighted least squares. `samples`
controls the variance: the API/UI default is 128, which is enough for an
opening board's 32 pieces. The empty and full coalitions are always included.

## Windows

The score probe may consume a window of preceding positions. Attribution then
covers every `(position, square)` slot in the window:

- `squares` — the current board (last position), 64 values;
- `segment_squares` — one 64-value row per position, oldest first;
- `segment_totals` — the per-position sum (additive for KernelSHAP).

The sacrifice classifier always uses a single position.

## CLI

```bash
uv run python -m src.run.explain_position \
  --checkpoint data/models/downstream/sacrifice/unfreeze_all_mean/run_20m_lazy/seed17/best.pt \
  --fen "r1q3r1/ppk5/1n1p1n2/2p1p2p/4P3/1Q1P4/PPP2PPP/RNB2RK1 w - - 0 1" \
  --method shap --samples 128 --top 8
```

The task (score vs sacrifice) is inferred from the checkpoint's head config;
override it with `--task`. Output is one JSON object with the full 64-square
vector, the piece-group sums, and the ranked `top` squares. `--moves` accepts a
comma-separated UCI move list applied to `--fen` first.

## HTTP API

```http
POST /api/attribution
{
  "start_fen": "...",
  "moves": [],
  "model_id": "data/models/.../best.pt",
  "method": "occlusion",   // or "shap"
  "occlusion": "mask",     // or "empty"
  "samples": 128,
  "seed": 0
}
```

Returns `{"fen": ..., "attribution": {...}}`. Unknown models are `404`; unknown
methods or occlusion modes are `400`.

## Web UI

The board panel has an **Attribution** section: pick a selected checkpoint and
a method, then press **Explain**. The current board is tinted green (raises the
output) or red (lowers it), with the top squares listed beside the controls.
The overlay clears whenever the position changes.

The browser sends `occlusion: "mask"` and `samples: 128` by default. On an
8-core CPU with a 20M probe, occlusion takes about 2 s and KernelSHAP about
3 s.

## Faithfulness

An explanation is only useful if deleting the squares it ranks changes the
model output more than deleting random squares. The evaluation script measures
exactly that:

```bash
uv run python -m src.run.evaluate_attribution \
  --checkpoint data/models/downstream/sacrifice/unfreeze_all_mean/run_20m_lazy/seed17/best.pt \
  --limit 12 --samples 128 --random-trials 3 --steps 16 \
  --output data/eval/attribution/sacrifice_20m_ftmean.json
```

For each position it

1. ranks squares by `|attribution|`;
2. deletes the top squares one by one (each method under its own counterfactual:
   occlusion uses `--occlusion`, KernelSHAP uses the empty board);
3. computes the mean absolute output change over the first `--steps`
   deletions, normalised by the base output;
4. repeats with random permutations as the baseline.

`lift` is the mean per-position ratio of attributed to random impact;
`median_lift` is the same median. The median is the headline number because a
single low-probability board can produce a very large ratio.

### Measured on the 20M full-fine-tuned probes (seed 17, 12 positions)

| task | method | median lift | mean lift |
| --- | --- | --- | --- |
| sacrifice | occlusion | 2.43 | 4.32 |
| sacrifice | KernelSHAP | 2.75 | 7.90 |
| score | occlusion | 2.52 | 3.44 |
| score | KernelSHAP | 1.76 | 2.30 |

Both methods beat random deletion on both tasks. Occlusion's advantage is
partly self-consistent: it ranks by single-square removal and the deletion
curve performs single-square removals, so its lift is a sanity check. KernelSHAP
is the informative comparison, because its ranking comes from a different
estimator (marginal contributions over sampled coalitions).

Raw reports: `data/eval/attribution/*.json`.

## Caveats

- **State tokens are not attributed.** Turn, castling rights, and en-passant
  stay fixed and are part of the baseline. For `feature="native"` checkpoints
  the content-isolated trunk ignores them anyway; for `full` recipes they can
  matter, so a square-only explanation is incomplete by construction.
- **Score sign.** The score probe predicts the realised game result from
  White's point of view, not centipawns.
- **SHAP variance.** 128 samples is interactive-quality, not publication
  quality. Use 512–1024 for analysis.
- **Piece groups** are sums of the square values for each colour and piece
  type, not independent counterfactuals.
