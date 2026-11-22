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

Every method asks "what happens if this piece is not the piece I think it is?"
and replaces the square token with either:

| replacement | meaning |
| --- | --- |
| `empty` | the square is empty — the piece is removed |
| `mask` | the square's content is unknown (the `MASK` token) |

`mask` matches the encoder's BERT-style pretraining distribution; `empty` is
the cleaner chess counterfactual, and empty squares become exact no-ops.

**Kings are never removable.** A kingless board is not a legal position, and
removing one measures the model's reaction to garbage. King squares are held
fixed and reported as `0`.

Attribution signs are in output space:

- **positive** = the piece's presence pushes the output up (toward
  "sacrifice", or toward a better White score);
- **negative** = the piece's presence pushes it down.

For the score probe, a Black piece with a positive value is one that helps
White.

## Occlusion

`occlusion` replaces one non-king square at a time and reports

```text
delta(square) = f(board) - f(board with that square replaced)
```

All variants are evaluated in one batched forward. Occlusion deltas are **not
additive** — interactions mean the parts do not sum to the whole.

This is an exact single-piece counterfactual: on position `08RGa` the values
match removing the piece from a `chess.Board` to within `5e-4`.

## KernelSHAP

`shap` treats every **non-king piece** as a player in a cooperative game.

The absent state matters more than the estimator. With no background the
absent square is empty, which is far out of distribution. The default is
**interventional SHAP**: an absent square takes its content from a small set
of real background positions (four puzzle starts by default), and the value
function averages over them. That keeps the sampled boards plausible; on the
chess benchmark below it lifts top-3 from 15% to 25% and top-5 from 25% to
35%.

Empty squares are never players: in the empty-board game they are exact
no-ops, and in the background game they are held fixed.

Coalitions are sampled uniformly, weighted by the Shapley kernel
`(n-1) / (C(n,s)·s·(n-s))`, and fit with a weighted least-squares solution
whose efficiency constraint is enforced exactly. Values sum to
`model_value - baseline_value`, where the baseline is the empty coalition
(all players replaced).

The estimator itself is verified against brute-force Shapley enumeration: on
an eight-player game it agrees with the exact values to `0.015`.

## Windows

The score probe may consume a window of preceding positions. Attribution then
covers every `(position, square)` slot in the window:

- `squares` — the current board (last position), 64 values;
- `segment_squares` — one 64-value row per position, oldest first;
- `segment_totals` — the per-position sum (additive for KernelSHAP).

The sacrifice classifier always uses a single position. The background rows
for a windowed score probe repeat the same background position across the
window.

## CLI

```bash
uv run python -m src.run.explain_position \
  --checkpoint data/models/downstream/sacrifice/unfreeze_all_mean/run_20m_lazy/seed17/best.pt \
  --fen "r1q3r1/ppk5/1n1p1n2/2p1p2p/4P3/1Q1P4/PPP2PPP/RNB2RK1 w - - 0 1" \
  --method shap --samples 128 --background-count 4 --top 8
```

The task (score vs sacrifice) is inferred from the checkpoint's head config;
override it with `--task`. Output is one JSON object with the full 64-square
vector, the piece-group sums, the ranked `top` squares, and the number of
background positions. `--moves` accepts a comma-separated UCI move list
applied to `--fen` first. `--background-count 0` falls back to the empty-board
game.

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

Returns `{"fen": ..., "attribution": {...}}`. KernelSHAP automatically uses
four puzzle starts as its background. Unknown models are `404`; unknown methods
or occlusion modes are `400`.

## Web UI

The board panel has an **Attribution** section: pick a selected checkpoint and
a method, then press **Explain**. Green means the piece's presence pushes the
model toward its verdict; red means it argues against it. The top pieces are
listed beside the controls, and the overlay clears whenever the position
changes.

The browser sends `occlusion: "mask"` and `samples: 128`. On an 8-core CPU
with a 20M probe, occlusion takes about 2 s and KernelSHAP about 5 s with the
background.

## Faithfulness

The strongest test is chess-grounded: **does the explanation rank the piece
the solution actually sacrifices near the top?** The puzzle shards contain the
solution, so the sacrificed piece is recoverable: it is the solver's piece
captured on its destination square by the immediate reply
([`src/chess_eval/puzzles.py`](../src/chess_eval/puzzles.py)).

```bash
uv run python -m src.run.evaluate_attribution \
  --checkpoint data/models/downstream/sacrifice/unfreeze_all_mean/run_20m_lazy/seed17/best.pt \
  --limit 20 --samples 64 --background-count 4 \
  --output data/eval/attribution/sacrifice_20m.json
```

Rank of the sacrificed piece over 20 puzzles (20M full-fine-tuned sacrifice
probe, seed 17):

| method | top-1 | top-3 | top-5 | mean rank |
| --- | --- | --- | --- | --- |
| occlusion | 25% | **55%** | **70%** | 9.3 |
| KernelSHAP, empty board | 10% | 15% | 25% | 10.7 |
| KernelSHAP, 4 backgrounds | 10% | 25% | 35% | 9.7 |

Occlusion is the piece localizer; KernelSHAP is a global, interaction-aware
decomposition and is measurably weaker at pointing at one piece. The
background fixes most of the empty-board damage but does not close the gap.

The script also reports the older **deletion-curve** comparison (delete the
top-ranked squares and measure the output change against random deletion).
That metric is secondary: for occlusion it is partly self-consistent, because
the ranking and the deletion use the same intervention.

### Case study: `08RGa`

`08RGa` is a two-rook sacrifice: `1.Kf1 Rg1+ 2.Nxg1 Rxg1+ 3.Kxg1 Qh2+`, so the
piece given up is the rook on `g2`.

| method | rank of `g2` | what it ranks first |
| --- | --- | --- |
| occlusion | 11 | White rook `e2` (+0.213) |
| KernelSHAP (background, seed 0) | 4 | black pawn `c6` (+0.108) |

The model is confident (0.878) and correct by label, but its deletion
response is dominated by White material: removing White's king or rooks moves
the prediction more than removing the rook that is actually sacrificed. This
is a property of the probe, not of the estimator — occlusion matches true
piece removal to `5e-4`, and the Shapley estimator matches exact enumeration.

Moving the `g2` rook instead of deleting it drops the prediction from 0.878 to
about 0.58, so the model does encode its placement. It just does not attribute
"sacrifice" to it in a leave-one-out sense. No attribution method can invent
piece-local reasoning the probe does not have.

## Caveats

- **State tokens are not attributed.** Turn, castling rights, and en-passant
  stay fixed and are part of the baseline. For `feature="native"` checkpoints
  the content-isolated trunk ignores them anyway; for `full` recipes they can
  matter, so a square-only explanation is incomplete by construction.
- **Score sign.** The score probe predicts the realised game result from
  White's point of view, not centipawns.
- **SHAP variance.** 128 samples is interactive-quality; per-position values
  move. Use 512–1024 for analysis, and expect the ranked order to change.
- **Piece groups** are sums of the square values for each colour and piece
  type, not independent counterfactuals.
- **Probe quality, not explanation quality.** Both methods are faithful to the
  model. When the model keys on material or king safety, the explanation will
  say so, even if the puzzle's theme is a tactical sacrifice.
