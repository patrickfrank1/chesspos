## ADDED Requirements

### Requirement: Dataset repository management
The system SHALL provide a class to manage HuggingFace Hub dataset repositories including creation, cloning, and versioning.

#### Scenario: Create new dataset repository
- **WHEN** a user creates a new dataset with a repository name that does not exist
- **THEN** the system creates a new repository on HuggingFace Hub with the specified name

#### Scenario: Clone existing dataset repository
- **WHEN** a user opens an existing dataset repository
- **THEN** the system clones the repository locally for batch operations

#### Scenario: Version dataset with git tags
- **WHEN** a user completes a dataset generation run
- **THEN** the system can create a git tag for the dataset version on HuggingFace Hub

### Requirement: Batch dataset generation and upload
The system SHALL generate chess position datasets in batches and push them to HuggingFace Hub with memory efficiency.

#### Scenario: Generate and push a batch
- **WHEN** a batch of positions is generated
- **THEN** the system creates a Parquet file and pushes it to HuggingFace Hub without loading all batches into memory

#### Scenario: Train/test split per batch
- **WHEN** a batch is generated with configurable train ratio
- **THEN** the system splits positions into train and test subsets and stores them in separate paths

#### Scenario: Resume from last batch
- **WHEN** a user resumes dataset generation
- **THEN** the system detects existing batches and continues from the next batch number

### Requirement: Dataset streaming for training
The system SHALL provide a data generator that streams positions from HuggingFace Hub datasets for model training.

#### Scenario: Stream training data from HF Hub
- **WHEN** a training pipeline requests data
- **THEN** the system streams positions from HuggingFace Hub without downloading the full dataset

#### Scenario: Load dataset with split selection
- **WHEN** a user specifies a split (train/test)
- **THEN** the system loads only positions from the requested split

#### Scenario: Shuffle streaming data
- **WHEN** a training pipeline requests shuffled data
- **THEN** the system provides a shuffled stream with configurable buffer size

### Requirement: Dataset schema management
The system SHALL define and enforce a consistent schema for chess position data stored on HuggingFace Hub.

#### Scenario: Validate dataset schema
- **WHEN** a batch is prepared for upload
- **THEN** the system validates that all samples conform to the expected schema (window, scalars)

#### Scenario: Document dataset features
- **WHEN** a dataset is created
- **THEN** the system includes a dataset card with feature descriptions and usage examples
