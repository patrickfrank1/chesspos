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

- [ ] Document findings of up to current model training
- [ ] Write to db from .npy files
  - [ ] write tokenized positions with some metadata and id
  - [ ] write embeddings generated from a model
- [ ] Write to db from .pgn file
  - maybe some refactoring is needed
- make embeddings better for search
  - document approaches
  - make a plan