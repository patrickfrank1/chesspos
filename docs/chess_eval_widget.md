# Interactive position evaluator

A small local web app for stepping through a chess position and comparing the
outputs of the trained downstream checkpoints.

- **Board:** click a piece, then click a destination. Legal destinations are
  highlighted; promotions use the selector under the board.
- **History:** the move chips let you jump back to any earlier ply; `Reset`
  returns to the initial position.
- **FEN:** paste any legal FEN and load it.
- **Models:** choose any combination of score regressors and sacrifice
  classifiers. The app auto-selects the best 20M score and sacrifice rows.
- **Score output:** White-perspective realised game result in `[-1, +1]`,
  where `+1` means White wins, `0` a draw, and `-1` Black wins. This is not a
  centipawn engine evaluation.
- **Sacrifice output:** the classifier's probability for the sacrifice theme,
  with the validation-optimal F1 threshold shown when the checkpoint has one.

## Run it

```bash
uv sync --extra cpu
uv run python -m src.chess_eval --host 127.0.0.1 --port 8000
```

Open <http://127.0.0.1:8000>.

The app scans these roots by default:

- `data/models/downstream/score`
- `data/models/downstream/sacrifice`
- `data/models/score_probe`
- `data/models/puzzle_classifier`

Pass one or more `--model-root` flags to override the search path. In each run
directory the app prefers `best.pt`, falls back to `last.pt`, and uses the
directory names for the model/recipe/seed label.

To use the GPU, pass `--device cuda` or set `CHESSPOS_DEVICE=cuda`. The app
loads checkpoints lazily and caches them in memory, so the first evaluation
with a new checkpoint is slower than subsequent ones.

## Fetching the downstream checkpoints

If the local downstream trees are empty:

```bash
hf buckets sync \
  hf://buckets/patrickfrank1/chesspos-checkpoints/downstream/runs/models/score \
  ./data/models/downstream/score/

hf buckets sync \
  hf://buckets/patrickfrank1/chesspos-checkpoints/downstream/runs/models/sacrifice \
  ./data/models/downstream/sacrifice/
```

## Limitations

- The score probe uses the model's saved window size; when the current move
  history is shorter than that window, it evaluates the available positions.
- The puzzle classifier was trained on single puzzle positions, so it evaluates
  the current position only.
- The 6M content-isolated native embedding is turn-blind by construction; the
  `full` feature recipes include the turn/castling/en-passant tokens.
- These are learned downstream probes, not an engine. A larger score or
  probability is a statement about the probe's training target, not a claim
  about best play.
