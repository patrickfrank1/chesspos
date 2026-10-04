## ADDED Requirements

### Requirement: ChessPositionDataset class
The system SHALL provide a ChessPositionDataset class that encapsulates dataset operations with clear separation of concerns.

#### Scenario: Initialize dataset with configuration
- **WHEN** a ChessPositionDataset is instantiated
- **THEN** it accepts configuration for repository name, batch size, encoding format, and splits

#### Scenario: Generate positions from source
- **WHEN** generate() is called on the dataset
- **THEN** it produces positions using the configured preprocessing pipeline

#### Scenario: Push batch to HuggingFace Hub
- **WHEN** push_batch() is called
- **THEN** the dataset uploads the current batch to HuggingFace Hub with proper metadata

### Requirement: PGNProcessor class
The system SHALL provide a PGNProcessor class that handles all PGN file operations and game extraction.

#### Scenario: Process PGN directory
- **WHEN** process_directory() is called with a path
- **THEN** the processor yields chess.Board objects from all PGN files in the directory

#### Scenario: Extract game with metadata
- **WHEN** extract_game() is called
- **THEN** the processor returns a GameRecord containing positions and metadata

#### Scenario: Apply sampling filters
- **WHEN** sampling filters are configured (ELO threshold, ply range)
- **THEN** the processor applies filters during extraction

### Requirement: PositionEncoder class
The system SHALL provide a PositionEncoder class that converts chess.Board objects to tensor representations.

#### Scenario: Encode single position
- **WHEN** encode() is called with a chess.Board
- **THEN** the encoder returns the encoded tensor in the configured format

#### Scenario: Encode batch of positions
- **WHEN** encode_batch() is called with multiple boards
- **THEN** the encoder returns a batched tensor array

#### Scenario: Support multiple encoding schemes
- **WHEN** the encoder is configured for a specific scheme
- **THEN** it applies the correct encoding (token_sequence, tensor, or bitboard)

### Requirement: HuggingFaceClient class
The system SHALL provide a HuggingFaceClient class that handles all HuggingFace Hub API interactions.

#### Scenario: Authenticate with HuggingFace
- **WHEN** the client is initialized
- **THEN** it verifies authentication status and raises clear error if not authenticated

#### Scenario: Create or load repository
- **WHEN** get_repository() is called
- **THEN** the client creates a new repo or clones an existing one

#### Scenario: Push dataset batch
- **WHEN** push_batch() is called with data and path
- **THEN** the client uploads the data as Parquet to the specified path

### Requirement: TrainingDataGenerator class
The system SHALL provide a TrainingDataGenerator class that integrates with Keras training pipelines.

#### Scenario: Create TensorFlow dataset
- **WHEN** to_tf_dataset() is called
- **THEN** the generator returns a tf.data.Dataset that streams from HuggingFace Hub

#### Scenario: Apply training transformations
- **WHEN** transformations are configured (masking, augmentation)
- **THEN** the generator applies them during data loading

#### Scenario: Handle epoch boundaries
- **WHEN** an epoch completes
- **THEN** the generator shuffles and prepares data for the next epoch

### Requirement: Configuration dataclasses
The system SHALL use dataclasses for configuration objects to ensure type safety and clarity.

#### Scenario: Validate configuration on creation
- **WHEN** a configuration dataclass is instantiated
- **THEN** it validates field values and raises errors for invalid configurations

#### Scenario: Serialize configuration to JSON
- **WHEN** configuration needs to be logged or stored
- **THEN** the dataclass can be serialized to JSON for reproducibility

### Requirement: Factory pattern for encoders
The system SHALL use a factory pattern to create encoder instances based on configuration.

#### Scenario: Create encoder by name
- **WHEN** get_encoder("token_sequence") is called
- **THEN** the factory returns a TokenSequenceEncoder instance

#### Scenario: Register custom encoders
- **WHEN** a custom encoder class is registered
- **THEN** the factory can create instances by the registered name
