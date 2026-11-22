# Interactive position evaluator

A small local web app for stepping through a chess position and comparing the
outputs of the trained downstream checkpoints.

- **Board:** click a piece, then click a destination. Legal destinations are
  highlighted; promotions use the selector under the board.
- **History:** the move chips let you jump back to any earlier ply; `Reset`
  returns to the initial position.
- **FEN:** paste any legal FEN and load it.
- **Puzzle samples:** choose a sampled puzzle position from the parquet puzzle
  shards. The dropdown mixes sacrifice and non-sacrifice puzzles and includes
  each puzzle’s start position plus every position after a solution move.
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
- **Attribution:** explain one selected checkpoint per square. Occlusion asks
  "what if this piece were not there?"; KernelSHAP gives additive Shapley
  values against an empty-board baseline. Green raises the output, red lowers
  it. See [`docs/attribution.md`](attribution.md) for the method details, CLI,
  and faithfulness measurements.

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

## Puzzle samples

The sample dropdown reads the local puzzle parquet shards:

```
data/processed/puzzles/*.parquet
```

It samples 60 puzzles (balanced between sacrifice and non-sacrifice) from the
first five shards, then exposes each puzzle's start position and every position
after a solution move. Selecting an entry loads that FEN directly; the model
and Stockfish evaluations update immediately.

The sample set is cached in memory after the first request. Use
`--puzzle-data-dir` to point at a different shard directory.

## Attribution

The board panel's **Attribution** section explains the current position for a
selected checkpoint. Choose `Occlusion` or `KernelSHAP`, then press `Explain`:

- green squares push the model output up, red squares push it down;
- the ranked list shows the strongest squares and the piece on each;
- the overlay clears when the position or the model selection changes.

Occlusion uses the `MASK` token by default, matching the encoder's pretraining
distribution. KernelSHAP only treats occupied squares as players and uses the
empty board as its baseline, so its values sum to
`model_value - baseline_value`.

The same explanations are available from the CLI:

```bash
uv run python -m src.run.explain_position \
  --checkpoint data/models/downstream/sacrifice/unfreeze_all_mean/run_20m_lazy/seed17/best.pt \
  --fen "..." --method shap --top 8
```

On an 8-core CPU with a 20M probe, occlusion takes about 2 s and KernelSHAP
about 3 s.

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
