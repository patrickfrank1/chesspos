## ADDED Requirements

### Requirement: Ray-based PGN file processing
The system SHALL use Ray for distributed processing of PGN files to extract chess positions efficiently.

#### Scenario: Distribute PGN files across workers
- **WHEN** multiple PGN files are processed
- **THEN** the system distributes files across Ray actors for parallel processing

#### Scenario: Process large PGN files
- **WHEN** a PGN file contains thousands of games
- **THEN** the system processes games in parallel using Ray to maximize throughput

#### Scenario: Handle worker failures gracefully
- **WHEN** a Ray worker fails during processing
- **THEN** the system retries the failed task or reports the error without losing progress

### Requirement: Parallel board-to-tensor conversion
The system SHALL use Ray for parallel conversion of chess.Board objects to tensor representations.

#### Scenario: Convert boards in parallel
- **WHEN** a batch of chess.Board objects is ready for encoding
- **THEN** the system distributes conversion tasks across Ray workers

#### Scenario: Support multiple encoding formats
- **WHEN** a user specifies an encoding format (token_sequence or tensor)
- **THEN** the system applies the correct conversion function in parallel

#### Scenario: Batch tensor output
- **WHEN** tensors are generated
- **THEN** the system collects results into batches for efficient downstream processing

### Requirement: Game-level processing pipeline
The system SHALL extract entire games from PGN files and process all positions within each game.

#### Scenario: Extract full game with metadata
- **WHEN** a game is read from a PGN file
- **THEN** the system extracts all positions along with game metadata (ELO, result, opening)

#### Scenario: Preserve game context
- **WHEN** positions are extracted from a game
- **THEN** the system maintains ply number and move sequence information for each position

#### Scenario: Temporal window extraction
- **WHEN** a user requests temporal windows
- **THEN** the system extracts consecutive positions from games as windows for autoregressive training

### Requirement: Scalable resource management
The system SHALL allow configuration of Ray cluster resources for preprocessing workloads.

#### Scenario: Configure worker count
- **WHEN** a user specifies the number of workers
- **THEN** the system initializes Ray with the requested number of actors

#### Scenario: Configure memory limits
- **WHEN** a user specifies memory limits per worker
- **THEN** the system enforces memory constraints to prevent OOM errors

#### Scenario: Monitor processing progress
- **WHEN** preprocessing is running
- **THEN** the system reports progress including files processed, positions extracted, and time elapsed

### Requirement: Lance DB compatibility
The system SHALL design preprocessing output to be compatible with future Lance DB integration for large-scale search indexing.

#### Scenario: Output to Lance-compatible format
- **WHEN** positions are preprocessed
- **THEN** the system outputs data in a format suitable for Lance DB ingestion (Parquet with proper schema)

#### Scenario: Include position embeddings placeholder
- **WHEN** positions are stored
- **THEN** the system includes a placeholder for future embedding vectors to support vector search
