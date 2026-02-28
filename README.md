# Chess Position Embeddings

## Setup

npm install @fission-ai/openspec@latest
npx openspec init
npx openspec update
Run /opsx:onboard in opencode or another agent

## Cheat Sheet

- Generate training positions

    python -m src.run.generate_positions

- Train a neural network

    python -m src.run.train

- Evaluate a trained network by starting the notebook: `src/run/evaluate.ipynb`

## Generate HuggingFace Dataset

Generates chess position datasets from PGN files and pushes them to HuggingFace Hub.

### Setup

1. **Authenticate with HuggingFace Hub:**

   ```bash
   uv run huggingface-cli login
   ```

   You'll need a HuggingFace access token (get one at https://huggingface.co/settings/tokens).

2. **Prepare PGN files:**

   Place your PGN files in `data/raw/`. The directory already contains Lichess elite game files.

### Usage

**Basic usage (dry run to test locally):**

```bash
uv run python -m src.run.generate_hf_dataset --repo your-username/test-dataset --dry-run --batches 1
```

**Push to HuggingFace Hub:**

```bash
uv run python -m src.run.generate_hf_dataset --repo your-username/chesspos-positions --batches 3
```

**Key options:**

| Option | Default | Description |
|--------|---------|-------------|
| `--repo` | required | HuggingFace Hub repository name |
| `--batches` | 3 | Number of batches to generate |
| `--batch-size` | 100000 | Positions per batch |
| `--dry-run` | false | Generate locally without pushing |
| `--resume` | false | Continue from last batch on Hub |
| `--encoding` | token_sequence | Encoding format (token_sequence/tensor/bitboard) |
| `--workers` | 4 | Ray workers for parallel processing |
| `--min-elo` | 2000 | Minimum ELO for position sampling |
| `--subsample` | 0.33 | Position subsampling rate |
| `--create-card` | false | Push dataset card to Hub |

**Example with custom settings:**

```bash
uv run python -m src.run.generate_hf_dataset \
  --repo your-username/chesspos-positions \
  --batches 5 \
  --batch-size 50000 \
  --min-elo 2200 \
  --subsample 0.5 \
  --create-card
```

## Tools

- Start ML Flow UI, in correct python venv

    mlflow ui

- Export dependencies to requirements.txt

    poetry export > requirements.txt

## Notes

### Milvus

Start local milvus db instance with:

`docker compose -f milvus-2-3-10-standalone-docker-compose.yml up -d`

Stop the instance with:

`docker compose -f milvus-2-3-10-standalone-docker-compose.yml down`

- could only get milvus 2.3.1 to work, so use that for now
- but had to downgrade python to 3.9, because of compatibility issues
- and only works with recent tensorflow version, so it's incompatible with aws sage maker
  - maybe I need to build a different toolchain for different python versions

## TODOs

- [ ] Write to db from .npy files
  - [ ] write tokenized positions with some metadata and id
  - [ ] write embeddings generated from a model
- [ ] Write to db from .pgn file
  - maybe some refactoring is needed
- make embeddings better for search
  - document approaches
  - make a plan