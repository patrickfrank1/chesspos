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
- **Stockfish 19:** the full Stockfish 19 WASM build runs locally in a Web
  Worker and evaluates the current position. Choose a search depth, or disable
  auto-evaluation and run on demand.
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

## Stockfish 19

The browser engine is the full `sf_19` build from
[`@lichess-org/stockfish-web`](https://github.com/lichess-org/stockfish-web)
version `0.6.1`, vendored under
`src/chess_eval/static/stockfish/`. It runs inside a dedicated Web
Worker, so searches do not block the board UI.

The NNUE file (`nn-1a298aa575a0.nnue`, about 94 MB) is downloaded on first use
from the official Stockfish NNUE host and cached at:

```
data/models/stockfish/nn-1a298aa575a0.nnue
```

The first page load therefore takes longer; subsequent loads use the cached
file and HTTP cache. The engine panel shows the download progress. If the
download fails, the panel reports the error but the model evaluations continue
to work.

Stockfish’s threaded WebAssembly build needs `SharedArrayBuffer`, which requires
cross-origin isolation. The app sends:

```
Cross-Origin-Opener-Policy: same-origin
Cross-Origin-Embedder-Policy: require-corp
Cross-Origin-Resource-Policy: same-origin
```

Use `http://127.0.0.1:8000`, `http://localhost:8000`, or serve the app over
HTTPS. A plain HTTP LAN address such as `http://192.168.1.10:8000` will not be a
secure context, and the engine panel will say that cross-origin isolation is
missing.

The engine evaluation is White-POV: centipawns are negated when Black is to
move, and mate scores are shown as `#N` for White or `#-N` for Black. The PV is
shown in UCI notation. Stockfish is GPLv3/AGPL-3.0; the vendored license is in
`src/chess_eval/static/stockfish/LICENSE`.

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
- Stockfish in the browser needs a modern browser with WebAssembly, module
  workers, and `SharedArrayBuffer` support.
