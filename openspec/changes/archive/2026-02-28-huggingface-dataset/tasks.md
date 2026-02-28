## 1. Setup and Dependencies

- [x] 1.1 Add Ray dependencies to pyproject.toml (ray[data], datasets)
- [x] 1.2 Create src/dataset module directory structure
- [x] 1.3 Define type hints and protocols in src/dataset/types.py

## 2. Configuration Dataclasses

- [x] 2.1 Create DatasetConfig dataclass with validation (repo_name, batch_size, encoding, splits)
- [x] 2.2 Create PreprocessingConfig dataclass (worker_count, memory_limit, sampling_filters)
- [x] 2.3 Create EncoderConfig dataclass (encoding_format, window_size)
- [x] 2.4 Add JSON serialization methods to all config dataclasses

## 3. Position Encoder Classes

- [x] 3.1 Create PositionEncoder abstract base class with encode() and encode_batch() methods
- [x] 3.2 Implement TokenSequenceEncoder subclass
- [x] 3.3 Implement TensorEncoder subclass
- [x] 3.4 Implement get_encoder() factory function

## 4. PGN Processor Class

- [x] 4.1 Create PGNProcessor class with process_directory() method
- [x] 4.2 Implement extract_game() returning GameRecord with positions and metadata
- [x] 4.3 Add sampling filter methods (ELO threshold, ply range)
- [x] 4.4 Implement temporal window extraction for consecutive positions
- [x] 4.5 Create GameRecord dataclass (positions, metadata, ply_numbers)

## 5. HuggingFace Client Class

- [x] 5.1 Create HuggingFaceClient class with authentication check
- [x] 5.2 Implement get_repository() for create/clone operations
- [x] 5.3 Implement push_batch() for Parquet upload with path_in_repo
- [x] 5.4 Implement create_version_tag() for dataset versioning
- [x] 5.5 Add resume detection by checking existing batches on Hub

## 6. Ray Distributed Processing

- [x] 6.1 Create RayPGNActor for distributed PGN file processing
- [x] 6.2 Create RayEncoderActor for parallel board-to-tensor conversion
- [x] 6.3 Implement distribute_pgn_files() to spread files across workers
- [x] 6.4 Add progress monitoring with file count and position count
- [x] 6.5 Implement graceful error handling and retry logic

## 7. Chess Position Dataset Class

- [x] 7.1 Create ChessPositionDataset class with config initialization
- [x] 7.2 Implement generate() method using PGNProcessor and Ray actors
- [x] 7.3 Implement push_batch() with train/test split
- [x] 7.4 Add schema validation before upload
- [x] 7.5 Create dataset card generation for HuggingFace Hub

## 8. Training Data Generator Class

- [x] 8.1 Create TrainingDataGenerator class for Keras integration
- [x] 8.2 Implement to_tf_dataset() with streaming from HuggingFace Hub
- [x] 8.3 Add split selection (train/test) support
- [x] 8.4 Implement shuffle with configurable buffer size
- [x] 8.5 Add transformation support (masking, augmentation)
- [x] 8.6 Implement epoch boundary handling with reshuffling

## 9. CLI Interface

- [x] 9.1 Create src/run/generate_hf_dataset.py entry point
- [x] 9.2 Add argparse arguments (repo, batches, batch-size, train-ratio, data-path)
- [x] 9.3 Add --dry-run flag for local testing without upload
- [x] 9.4 Add --resume flag to continue from last batch
- [x] 9.5 Add --workers and --memory flags for Ray configuration

## 10. Testing

- [x] 10.1 Write unit tests for PositionEncoder classes
- [x] 10.2 Write unit tests for PGNProcessor with sample PGN files
- [x] 10.3 Write unit tests for HuggingFaceClient (mocked)
- [x] 10.4 Write integration tests for ChessPositionDataset
- [x] 10.5 Write tests for TrainingDataGenerator streaming
