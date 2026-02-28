# AGENTS.md - Instructions for Coding Agents

This document provides essential guidelines for agentic coding tools working in the chesspos repository.

## Project Overview

Chess position embedding learning system using TensorFlow/Keras with Milvus vector database for semantic search of chess positions.

## Critical Instructions

- **Always use the context7 MCP server** to research usage of libraries that are not well known (e.g., keras-nlp, python-chess, etc.)
- **Always use `uv` for package management** instead of pip
- **Always use `uv run python` instead of `python`** to run scripts

## Build/Lint/Test Commands

### Package Management

```bash
uv sync                    # Install dependencies
uv sync --dev              # Install dev dependencies
uv add <package>           # Add a new dependency
uv add --dev <package>     # Add a dev dependency
```

### Running Scripts

```bash
uv run python -m src.run.generate_positions   # Generate training positions
uv run python -m src.run.train                # Train a neural network
uv run python -m src.run.build_vector_db      # Build vector database
```

### Testing

```bash
uv run pytest test/                                    # Run all tests
uv run pytest test/test_vector_db.py                   # Run a single test file
uv run pytest test/test_vector_db.py::test_milvus_vector_store_insert_and_search   # Run specific test
uv run pytest -v test/                                 # Run with verbose output
uv run pytest -x test/                                 # Stop on first failure
```

### Linting and Formatting

```bash
uv run ruff format .        # Format all Python files
uv run ruff check .         # Lint all Python files
uv run ruff check --fix .   # Auto-fix linting issues
```

### Infrastructure

```bash
uv run mlflow ui                                                        # Start MLflow UI
```

## Code Style Guidelines

### Python Version

- Requires Python >=3.11.8

### Imports

- Group imports in the following order, separated by blank lines:
  1. Standard library imports (alphabetically sorted)
  2. Third-party imports (alphabetically sorted)
  3. Local imports (alphabetically sorted)
- Use explicit imports, avoid `from module import *`
- Example:
  ```python
  import os
  from typing import Tuple

  import numpy as np
  import tensorflow as tf
  from tensorflow import keras

  from src.types import IOTensorPair
  from src.modeling.model import get_model
  ```

### Formatting

- Use Ruff for formatting (configured via VS Code settings)
- Format on save is enabled
- Auto-organize imports on save

### Type Annotations

- Use type hints for function parameters and return types
- Use modern Python type syntax (e.g., `list[str]` instead of `List[str]`)
- Use `|` for union types instead of `Union`
- Example:
  ```python
  def get_model(model: str) -> dict[str, keras.Model]:
  def search_by_ids(self, query_ids: list[int]) -> list[dict]:
  def _piece_id_to_piece(id: int) -> chess.Piece | None:
  ```

### Naming Conventions

- Functions and variables: `snake_case`
- Classes: `PascalCase`
- Constants (module-level): `UPPER_SNAKE_CASE`
- Private methods: prefix with underscore (e.g., `_connect`, `_create_collection`)
- Example:
  ```python
  PIECE_ENCODING = {...}  # Constant
  EMBEDDING_SIZE = 256    # Constant

  def board_to_bitboard(board: chess.Board) -> np.ndarray:  # Function
  class MilvusVectorStore:  # Class
  def _connect(self):  # Private method
  ```

### Error Handling

- Raise specific exceptions with descriptive messages
- Use `ValueError` for invalid arguments
- Use `assert` for internal invariants and preconditions
- Example:
  ```python
  raise ValueError("The requested neural network architecture does not exist.")
  raise ValueError(f"Invalid embedding data type for collection {self.collection_name}.")
  assert bb.shape == (773,)
  assert tensor.shape == (8, 8, 15)
  ```

### Documentation

- Use inline comments sparingly for complex logic
- Use docstrings for public classes and methods
- Keep comments on the same line as code for short explanations
- Example:
  ```python
  def _connect(self):
      """Connect to the Milvus server."""
      ...

  # one plane per piece
  for color in [1, 0]:
  ```

### Code Organization

- Source code lives in `src/` with subdirectories by concern:
  - `src/modeling/` - Model definitions
  - `src/preprocessing/` - Data processing and board representations
  - `src/training/` - Data generators and training utilities
  - `src/run/` - Main executable scripts
  - `src/search/` - Vector database integration
  - `src/evaluation/` - Visualization and evaluation
  - `src/utils/` - File operations and utilities
- Tests live in `test/` directory
- Notebooks for debugging in `notebooks/`
- Saved models in `model/`

### TensorFlow/Keras Conventions

- Use `bfloat16` dtype for neural network computations
- Use `keras.Model` for model definitions
- Return model components as dict: `{'encoder': encoder, 'decoder': decoder, 'autoencoder': autoencoder}`
- Use `keras_nlp` layers for transformer components

## Project-Specific Patterns

### Board Representations

- **Token Sequence**: 69-token sequence for Transformers

### Model Architecture

- Models are defined in `src/modeling/model.py`
- Use factory pattern: `get_model(model_name: str) -> dict[str, keras.Model]`
- Available architectures: `vanilla_dense`, `skip_dense`, `skip_equi_dense`, `cnn_dense`, `trivial`, `encoder_decoder_transformer`

## Notes

- Run linting after making changes: `uv run ruff check .`
