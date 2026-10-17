# data/

All products of the pipeline live here. Nothing in this directory is committed
to the repository (except this README).

```
data/
  raw/         # Lichess database dumps (.pgn, .csv.zst) — download these
  processed/   # Tokenized parquet shards produced by generate_hf_dataset.py
               #   train/ and test/ subdirectories
  models/      # Training checkpoints, one subdirectory per run
               #   (e.g. models/run_full/best.pt, last.pt, train.log)
  eval/        # Evaluation outputs from eval_model.py (plots, reports)
```

## Downloading raw data from Lichess

Database dumps are published monthly at <https://database.lichess.org>.

- Elite games (used for training):
  <https://database.lichess.org/#standard> — pick `lichess_elite_YYYY-MM.pgn.zst`
- Puzzles: <https://database.lichess.org/#puzzles> — `lichess_db_puzzle.csv.zst`

Download into `data/raw/`:

```sh
curl -o data/raw/lichess_elite_2014-02.pgn.zst \
  https://database.lichess.org/elite/lichess_elite_2014-02.pgn.zst
```

The files are zstandard-compressed; decompress before running the pipeline:

```sh
zstd -d data/raw/lichess_elite_2014-02.pgn.zst
```

## Reproducing downstream products

- `processed/`: `uv run python -m src.run.generate_hf_dataset --config configs/pipeline.yaml`
- `models/`: `uv run python -m src.run.train_masked_transformer`
- `eval/`: `uv run python -m src.run.eval_model`
