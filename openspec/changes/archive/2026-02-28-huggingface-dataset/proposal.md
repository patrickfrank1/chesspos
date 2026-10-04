# Proposal: HuggingFace Dataset Generation

## Summary

Replace the current `.npz` file storage with HuggingFace Hub dataset versioning. Generate chess position datasets using the existing `extract_board()` and `board_to_token_sequence()` pipeline, and push them to HuggingFace Hub in batches for versioning and streaming.

## Motivation

- Previous data versioning used DVC (now removed)
- Need reproducible, versioned datasets for ML experiments
- HuggingFace Hub provides:
  - Built-in versioning via commits/tags
  - Streaming support for large datasets
  - Easy loading in PyTorch/TensorFlow/Keras
  - Public or private dataset hosting

## Scope

### In Scope
- Create new script `src/run/generate_hf_dataset.py`
- Generate positions using existing `extract_board()` and `board_to_token_sequence()`
- Push to HuggingFace Hub in batches (Parquet format)
- 95/5 train/test split per batch
- CLI interface for configuration

### Out of Scope
- Modifying existing `extract_board()` logic
- Adding position metadata (future enhancement)
- Temporal window extraction (currently duplicates single position 10x)

## Dataset Schema

```
window:  (10, 69) int8  — 10 copies of token_sequence
scalars: (3,) int8       — placeholder zeros
```

Each sample contains:
- `window`: 10 copies of the same position encoded as token sequence (69 tokens each)
- `scalars`: 3 placeholder zeros for future use

## Implementation Plan

### File Structure
```
src/run/generate_hf_dataset.py
```

### Batch Processing Flow

```
For each of 3 batches:

1. Generate 100k positions
   extract_board() → board_to_token_sequence()
   Duplicate 10x → window (10, 69)
   Add scalars (3,) zeros

2. Split 95/5
   train: 95k samples | test: 5k samples

3. Create Dataset and push
   train_batch_001.parquet → hub:train/
   test_batch_001.parquet  → hub:test/
```

### Result on HF Hub

```
train/
  ├── batch_001.parquet  (95k)
  ├── batch_002.parquet  (95k)
  └── batch_003.parquet  (95k)
test/
  ├── batch_001.parquet  (5k)
  ├── batch_002.parquet  (5k)
  └── batch_003.parquet  (5k)

Total: 285k train | 15k test
```

### Minimum sample

```python
# ===============================
# Large-scale dataset with streaming / append
# ===============================

from datasets import IterableDataset, DatasetDict
from huggingface_hub import Repository
import numpy as np
import os

# -------------------------------
# Parameters
# -------------------------------
repo_name = "my_integer_tensors"  # Hugging Face dataset repo
local_repo_path = "./my_integer_tensors_repo"
batch_size = 100_000  # number of samples per batch
num_batches_v1 = 10   # simulate v1
num_batches_v2 = 5    # simulate v2

# Tensor shapes
window_shape = (9, 32, 32)
special_shape = (32, 32)
scalars_shape = (3,)

# -------------------------------
# Helper function to generate a batch of samples
# -------------------------------
def generate_batch(batch_size):
    windows = [np.random.randint(0, 256, window_shape, dtype=np.int32) for _ in range(batch_size)]
    specials = [np.random.randint(0, 256, special_shape, dtype=np.int32) for _ in range(batch_size)]
    scalars = [np.random.randint(0, 10, scalars_shape, dtype=np.int32) for _ in range(batch_size)]
    return {
        "window": windows,
        "special": specials,
        "scalars": scalars
    }

# -------------------------------
# Step 1: Initialize HF repo locally
# -------------------------------
if not os.path.exists(local_repo_path):
    repo = Repository(local_repo_path, clone_from=f"your-username/{repo_name}", use_auth_token=True)
else:
    repo = Repository(local_repo_path, use_auth_token=True)

# -------------------------------
# Step 2: Create v1 batches
# -------------------------------
for i in range(num_batches_v1):
    print(f"Generating v1 batch {i+1}/{num_batches_v1}")
    batch_data = generate_batch(batch_size)
    
    # Convert to IterableDataset
    batch_dataset = IterableDataset.from_generator(
        lambda: (sample for sample in zip(batch_data["window"], batch_data["special"], batch_data["scalars"])),
        features={
            "window": np.array(window_shape, dtype=np.int32),
            "special": np.array(special_shape, dtype=np.int32),
            "scalars": np.array(scalars_shape, dtype=np.int32)
        }
    )

    # Save batch locally as an Arrow file
    batch_file = os.path.join(local_repo_path, f"train_batch_{i+1}.arrow")
    batch_dataset.save_to_disk(batch_file)

    # Push this batch to HF Hub
    batch_dataset.push_to_hub(
        repo_name,
        path_in_repo=f"train/train_batch_{i+1}.arrow",
        token=True,
        commit_message=f"Add v1 batch {i+1}"
    )

# -------------------------------
# Step 3: Append v2 batches
# -------------------------------
for i in range(num_batches_v2):
    print(f"Generating v2 batch {i+1}/{num_batches_v2}")
    batch_data = generate_batch(batch_size)
    
    batch_dataset = IterableDataset.from_generator(
        lambda: (sample for sample in zip(batch_data["window"], batch_data["special"], batch_data["scalars"])),
        features={
            "window": np.array(window_shape, dtype=np.int32),
            "special": np.array(special_shape, dtype=np.int32),
            "scalars": np.array(scalars_shape, dtype=np.int32)
        }
    )

    batch_file = os.path.join(local_repo_path, f"train_batch_v2_{i+1}.arrow")
    batch_dataset.save_to_disk(batch_file)

    # Push this batch to HF Hub
    batch_dataset.push_to_hub(
        repo_name,
        path_in_repo=f"train/train_batch_v2_{i+1}.arrow",
        token=True,
        commit_message=f"Add v2 batch {i+1}"
    )

# -------------------------------
# Step 4: Users can stream the dataset
# -------------------------------
from datasets import load_dataset

# Streaming the merged dataset without downloading everything
dataset = load_dataset(f"your-username/{repo_name}", split="train", streaming=True)

for idx, sample in enumerate(dataset):
    if idx < 5:
        print("Sample:", sample)
    else:
        break
```

### CLI Interface

```bash
python -m src.run.generate_hf_dataset \
    --repo your-username/chesspos-positions \
    --batches 3 \
    --batch-size 100000 \
    --train-ratio 0.95 \
    --data-path ./data/raw
```

### Key Decisions

| Aspect | Decision | Rationale |
|--------|----------|-----------|
| Format | Parquet | HF default, better compression, interoperable |
| Split | 95/5 per batch | Matches existing pipeline convention |
| Memory | Batch-at-a-time | Never load all data into memory |
| Auth | `huggingface-cli login` | Standard HF authentication flow |

### Dependencies

- `datasets>=4.5.0` — already installed
- `huggingface_hub` — comes with `datasets`

## Pre-requisites

User must authenticate before running:

```bash
huggingface-cli login
# Enter HF token when prompted
```

## Future Enhancements

- Real temporal window extraction (10 consecutive positions)
- Position metadata (source game, ply, ELOs)
- Append new batches for version 2, 3, etc.
- Configurable encoding (bitboard, tensor, token_sequence)
