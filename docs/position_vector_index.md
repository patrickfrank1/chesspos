# Position Vector Index

A LanceDB index of chess-position embeddings for retrieving games with
positions similar to a query position. The index is built offline from the
packed token-stream shards and queried through a CLI.

## Layout

| Module | Responsibility |
|---|---|
| `src/search/codec.py` | 34-byte lossless position codec |
| `src/search/schema.py` | Arrow schema, compact value objects, manifest |
| `src/search/store.py` | filter/hit interfaces |
| `src/search/lancedb_store.py` | LanceDB table, search, indexes, manifest |
| `src/search/memory_store.py` | brute-force oracle for tests and ANN recall |
| `src/search/embedder.py` | checkpoint-backed `native_positions` embedder |
| `src/search/index.py` | packed game → per-position rows; parquet/Hub builds |
| `src/search/query.py` | group position hits into games |
| `src/run/build_position_index.py` | build CLI |
| `src/run/search_positions.py` | search CLI |

## Schema

One row per `(game, ply)`, stored as a single LanceDB table:

| Column | Type | Notes |
|---|---|---|
| `vector` | `float16[512]` | L2-normalized checkpoint embedding |
| `packed_position` | `binary[34]` | board + turn + castling + en passant |
| `game_id` | `uint64` | 16-hex game id parsed as an integer |
| `ply` | `uint16` | 1-based ply |
| `white_elo`, `black_elo` | `uint16` | 0 means missing |
| `utc_date` | `date32` | `1970-01-01` when the source has no date |
| `year` | `uint16` | exact game year, else archive year |
| `result` | `uint8` | 0 unknown, 1 white, 2 black, 3 draw |
| `split` | `uint8` | 0 train, 1 test |
| `n_pieces` | `uint8` | phase/material filtering without decoding |

The vector column dominates storage (~1 KB/row). The remaining columns are
narrow integers and fixed-size binary, so a billion positions is roughly
1 TB of vectors plus ~60 GB of metadata. The 34-byte position codec avoids
storing FEN strings.

## Build

From a local parquet directory:

```bash
uv run python -m src.run.build_position_index \
  --data-dir data/processed/hub/train \
  --checkpoint data/models/run_20m_lazy/last.pt \
  --output data/vector/positions \
  --device cuda
```

From a HuggingFace dataset repo (a bounded, evenly spread shard subset):

```bash
uv run python -m src.run.build_position_index \
  --repo patrickfrank1/chess-positions \
  --split train \
  --max-bytes 200000000 \
  --checkpoint data/models/run_20m_lazy/last.pt \
  --output data/vector/positions \
  --device cuda
```

Useful flags: `--limit-games`, `--batch-games`, `--no-index`,
`--index-type ivf_pq|ivf_hnsw_sq|ivf_flat`, `--num-partitions`,
`--num-sub-vectors`, `--overwrite`, `--revision`, `--cache-dir`.

The build writes `<output>/<table>.manifest.json` with the checkpoint hash,
feature recipe, dimension, dtype, row/game counts, date coverage and LanceDB
version.

## Search

```bash
uv run python -m src.run.search_positions \
  --index data/vector/positions \
  --checkpoint data/models/run_20m_lazy/last.pt \
  --fen "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1" \
  --k 200 --n-games 10 \
  --min-elo 2400 --min-year 2020 --result 1-0
```

The default output groups position hits by game and reports the best matching
ply/FEN per game. `--by position` shows raw position hits, `--json` prints the
structured payload, `--moves` applies UCI/SAN moves to the query board, and
`--pgn` replays a game (optionally up to `--ply`). `--exclude-game` removes a
game from the results.

Filters: `--min-elo`, `--max-elo`, `--min-year`, `--max-year`, `--min-date`,
`--max-date`, `--result`, `--split`, `--min-ply`, `--max-ply`,
`--min-pieces`, `--max-pieces`. `--nprobes`, `--refine-factor` and `--exact`
control the ANN/recall trade-off.

## Data Caveats

- **Vocabulary.** Shards written before the 52-token stream encoding contain
  tokens up to 805 and cannot be embedded by current checkpoints. The builder
  raises a clear error (`position token N exceeds the embedder vocabulary`)
  instead of failing inside PyTorch. Rebuild those shards with the current
  pipeline or use the newer archive shards under `train/{archive_id}/`.
- **Dates and splits.** Existing parquet shards do not carry per-game
  `utc_date` or `split` columns. The builder uses the path split
  (`train`/`test`), derives the year from the archive path (e.g.
  `lichess_elite_2013-09`), and stores `1970-01-01` when a date is missing.
  `--min-date`/`--max-date` therefore only filter rows whose shard carried
  dates; the manifest records `date_coverage`. `--min-year`/`--max-year` work
  for all existing shards.
- **Old checkpoints.** Use a checkpoint whose `vocab_size` matches the shard
  tokenizer (current checkpoints use 52).

## Scale Notes

- The first build is one shard; the target is 1B+ positions. LanceDB is a
  single-node embedded engine, so 1B is handled by partitioning (one table per
  source/year/split) and fanning out queries, not by horizontal clustering.
- Store `float16` vectors and build `IVF_PQ` (32 bytes/vector at 512-d); use
  `refine_factor` (10-50) to rerank candidates against the stored vectors.
- Embedding 1B positions is a GPU batch job. Build partitions independently,
  append batched rows, and run `optimize()` after large writes.
- A 1 TB index does not fit on the current 408 GB local disk; use an external
  NVMe volume or object storage for the full corpus.
- `.bypass_vector_index()` (the `--exact` flag) provides exact kNN for
  measuring ANN recall.
