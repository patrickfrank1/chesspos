# Evaluation of ML Pipeline Frameworks and Storage Formats for Chess Position Foundation Model Training

## Context

This project aims to build a foundation model for chess positions end-to-end: extract positions from PGN files, store them efficiently, train a model, and use it for downstream tasks (semantic search, position evaluation, tactical/positional classification).

The project currently uses:
- **Ray[data]** for parallel PGN processing (already a dependency)
- **TensorFlow/Keras** for model training (autoencoders, transformers)
- **NumPy .npz files** for local training data storage
- **Parquet** (via HuggingFace Datasets) for cloud/distributed datasets
- **Milvus** for vector similarity search

This report evaluates candidate technologies for the pipeline orchestration layer and the training data storage layer.

---

## Part 1: ML Pipeline Frameworks

### Evaluation Criteria

| Criterion | Weight | Description |
|---|---|---|
| ML-native features | High | Built-in support for distributed training, hyperparameter tuning, model tracking |
| TensorFlow/Keras integration | **Critical** | The project is TensorFlow-based; the framework must work well with it |
| Already in use | High | Using an existing dependency reduces complexity |
| Scalability | Medium | Must scale from single-machine to multi-node |
| Ease of adoption | Medium | Low boilerplate, familiar Python patterns |
| Pipeline observability | Medium | Logging, lineage, monitoring, retries |
| Scheduling & automation | Low | Cron-like scheduling, event-driven triggers |

### Candidates

#### 1. Ray (ray-project/ray)

Already a project dependency (`ray[data] >=2.40.0`).

**Architecture**: Unified distributed computing framework. Libraries: Ray Data (data processing), Ray Train (distributed training), Ray Serve (model serving), Ray Tune (hyperparameter optimization).

**Strengths**:
- Already in the codebase and actively used for PGN processing
- `Ray Data` reads Parquet, JSON, CSV, images; supports `map_batches` for parallel transformations identical to current usage
- `TensorflowTrainer` provides native distributed Keras training with `MultiWorkerMirroredStrategy`
- End-to-end pipeline: data loading → preprocessing → training → serving in one framework
- Scales from laptop to cluster with zero code changes
- `iter_torch_batches` / `iter_tf_batches` for seamless training data streaming

**Weaknesses**:
- More of a compute framework than a pipeline orchestrator (no built-in scheduling, retries, or DAG visualization)
- Debugging distributed failures can be complex
- Resource management overhead for small workloads

**Verdict**: **Strongly recommended** as the core compute layer. Already integrated, covers the entire ML lifecycle, and provides the exact functionality needed for data processing and distributed training.

#### 2. Dagster (dagster-io/dagster)

**Architecture**: Asset-centric orchestrator. Define data assets as Python functions with declarative dependencies. Automatic lineage tracking and observability.

**Strengths**:
- Asset-centric model: every step (dataset, model, evaluation) is a versioned asset with lineage
- Excellent UI for pipeline visualization and experiment comparison
- Built-in scheduling, retries, and error handling
- Strong testing support (assets are testable in isolation)
- ML-specific examples show PyTorch training orchestration with metadata logging

**Weaknesses**:
- Not ML-native: no distributed training, hyperparameter tuning, or model serving
- Would sit on top of Ray/Keras, adding another layer of abstraction
- No native TensorFlow integration (can call any Python code, but no specialized support)
- Deployment requires Dagster daemon/web server infrastructure

**Verdict**: **Good choice for pipeline management** if observability, scheduling, and asset lineage are needed. However, it adds complexity on top of Ray without providing ML-specific compute capabilities. Better as a complementary orchestrator around Ray, not a replacement.

#### 3. Flyte (flyteorg/flyte)

**Architecture**: Kubernetes-native workflow orchestrator. Tasks and workflows defined in pure Python (`flytekit`). Type-safe data passing between tasks, caching, retries.

**Strengths**:
- Purpose-built for ML pipelines (Lyft/Uber lineage)
- Strongly typed task contracts (data schema validation between steps)
- Kubernetes-native with multi-cluster routing
- Task-level caching and memoization
- Extensible plugin system

**Weaknesses**:
- Requires Kubernetes infrastructure (significant operational overhead)
- Steep learning curve for setup and configuration
- Overkill for single-machine or small-cluster workflows
- No native TensorFlow distributed training integration
- Not already in the project

**Verdict**: **Overkill for this project**. Requires Kubernetes infrastructure and adds significant operational complexity. Best suited for large teams with existing K8s clusters running production ML at scale.

#### 4. Kedro (kedro-org/kedro)

**Architecture**: Data science pipeline framework. Node/pipeline/catalog model. Focus on software engineering best practices, reproducibility, and modularity.

**Strengths**:
- Excellent project structure and conventions (nodes → pipelines → catalog)
- Built-in data catalog with versioning (supports Parquet, CSV, SQL, cloud storage)
- Strong MLflow integration for experiment tracking
- Promotes clean separation of concerns (data engineering vs. data science pipelines)
- Great for team collaboration and reproducibility

**Weaknesses**:
- More of a project template/framework than an orchestrator
- No distributed compute (relies on external runners like Ray, Spark, or Dask)
- Adds structure but may conflict with existing project organization
- No native TensorFlow training integration

**Verdict**: **Not a good fit**. The project already has a well-organized structure. Kedro would impose a different organizational pattern and doesn't provide distributed compute capabilities.

#### 5. Prefect (prefecthq/prefect)

**Architecture**: Python-native workflow orchestrator. `@flow` and `@task` decorators. Dynamic, event-driven pipelines with scheduling, caching, and retries.

**Strengths**:
- Easiest to adopt: pure Python, no DSL, minimal boilerplate
- Excellent observability (real-time monitoring dashboard)
- Flexible deployment (local, Docker, K8s, cloud)
- Built-in retry, caching, concurrency control
- Event-based triggers and automations

**Weaknesses**:
- General-purpose orchestrator, not ML-specific
- No distributed training or data processing capabilities
- Would sit on top of Ray/Keras for actual compute
- Not already in the project

**Verdict**: **Good for pipeline scheduling and observability** if automated retraining, scheduled dataset updates, or event-driven workflows are needed. Simpler than Dagster, but also less ML-specific.

#### 6. PySpark (Apache Spark Python API)

**Architecture**: Distributed data processing on JVM-based Spark clusters. DataFrame API with SQL, MLlib, and streaming.

**Strengths**:
- Battle-tested at massive scale (petabyte-level data processing)
- Rich DataFrame API with SQL support
- Mature ecosystem and tooling

**Weaknesses**:
- Heavily JVM-dependent (Spark runs on JVM, Python is a wrapper)
- Poor deep learning integration (MLlib is for classical ML)
- Massive infrastructure overhead (requires Spark cluster)
- Not suitable for training TensorFlow models
- Significant impedance mismatch with NumPy/TensorFlow data formats

**Verdict**: **Not recommended**. The overhead of running a Spark cluster is unjustified for this project. Ray handles distributed data processing more natively for Python/ML workloads.

### Pipeline Framework Recommendation

**Primary: Ray** — Use as the core compute layer for data processing and distributed training. Already integrated and provides everything needed.

**Optional complement: Prefect or Dagster** — Add if the project needs scheduled retraining, event-driven pipelines, or a UI for pipeline lineage. Prefect is simpler; Dagster is more feature-rich for asset management.

---

## Part 2: Training Data Storage Formats

### Evaluation Criteria

| Criterion | Weight | Description |
|---|---|---|
| ML framework integration | **Critical** | Direct DataLoader/Dataset support for TensorFlow/Keras |
| Random access performance | **Critical** | Fast row-level random access for training shuffling |
| Read throughput | High | High-bandwidth sequential reads for epoch iteration |
| Write throughput | High | Efficient batch writes during dataset generation |
| Versioning | Medium | Time travel, dataset snapshots for reproducibility |
| Vector search | Medium | Built-in similarity search (currently uses Milvus) |
| Schema evolution | Medium | Adding/removing columns without full rewrites |
| Maturity & ecosystem | Medium | Community size, tool support, stability |
| Already in use | Low | Compatibility with existing Parquet-based HuggingFace pipeline |

### Candidates

#### 1. Parquet (Apache)

Already used via HuggingFace Datasets pipeline.

**Strengths**:
- Universal standard: supported by virtually every data tool
- Excellent compression (column-oriented encoding: dictionary, RLE, delta)
- Predicate pushdown and column pruning for query optimization
- HuggingFace Datasets natively reads/writes Parquet
- Ray Data reads Parquet efficiently
- Mature, stable, well-documented

**Weaknesses**:
- No built-in versioning or time travel
- Append-only with no update/delete support
- Random access requires full file reads (not optimized for single-row access)
- No vector search capabilities
- Schema changes require rewriting files

**Verdict**: **Keep for HuggingFace distribution**. The existing Parquet pipeline works well for sharing datasets. Continue using it for this purpose.

#### 2. Delta Lake (delta-io/delta-rs)

Layer on top of Parquet providing lakehouse features.

**Strengths**:
- ACID transactions with concurrent read/write isolation
- Time travel: query data at any historical version
- Schema enforcement and evolution
- Z-ordering for improved read performance
- Column-level statistics for file skipping
- Python-native via `delta-rs` (no Spark required)
- Compatible with existing Parquet files

**Weaknesses**:
- Adds a transaction log layer over Parquet (slight overhead)
- Version history grows over time (needs vacuuming)
- No direct TensorFlow/PyTorch DataLoader integration
- No vector search capabilities
- Not as widely supported as plain Parquet in ML tools

**Verdict**: **Good if versioning is critical**. Delta Lake adds ACID and time travel to the existing Parquet workflow. Useful if the project needs reproducible dataset snapshots or concurrent read/write from multiple processes. However, it doesn't solve the random access or ML integration problems.

#### 3. LanceDB / Lance Format (lancedb/lancedb)

Modern lakehouse format designed for multimodal AI workloads.

**Architecture**: Lance is a columnar storage format optimized for ML. LanceDB is a database built on Lance with vector search, FTS, and SQL filtering.

**Strengths**:
- **Blazing fast random access**: O(1) row-level access, critical for training data shuffling
- **Native PyTorch DataLoader integration**: `torch.utils.data.DataLoader(table, batch_size=1024, shuffle=True)` — LanceDB tables are directly iterable as PyTorch Datasets
- **Column projection**: `select_columns(["image", "embedding"])` loads only needed columns
- **Built-in vector indexes** (IVF_PQ, IVF_FLAT): could replace or complement Milvus for simpler deployments
- **Built-in full-text search** (FTS) indexes
- **Hybrid search**: combine vector similarity + keyword search + SQL filtering
- **Data versioning**: mutations create new versions, preserving history
- **Multimodal native**: stores tensors, embeddings, text, metadata in a single table
- **Efficient writes**: appends and column additions without full rewrites
- **Zero-copy with Apache Arrow**: data flows to NumPy/PyTorch without copies
- **Python-native**: `pip install lancedb`, no infrastructure dependencies

**Weaknesses**:
- Newer project (smaller community than Parquet)
- API still evolving (some instability risk)
- No built-in TensorFlow/Keras DataLoader (but works via `tf.data.Dataset.from_generator`)
- Less tooling and integrations than Parquet
- Not a drop-in replacement for the HuggingFace pipeline (HuggingFace Datasets uses Parquet)

**Verdict**: **Strongly recommended for local training data storage**. LanceDB directly addresses the key pain points: fast random access for shuffling, native ML framework integration, and built-in vector search. It could replace both `.npz` files and Milvus for simpler deployments.

#### 4. NumPy .npz Files

Currently used for local training data.

**Strengths**:
- Zero dependencies beyond NumPy
- Fast reads (memory-mapped mode supported)
- Simple format
- Already working

**Weaknesses**:
- No random access without loading the entire file
- No schema/metadata
- No querying or filtering
- No versioning
- Duplication across train/test splits
- Inflexible: changing encoding format requires regenerating all files

**Verdict**: **Phased out**. While functional, `.npz` files are a bottleneck for the foundation model vision. They lack random access, versioning, and ML framework integration.

### Storage Format Recommendation

**Primary: LanceDB (Lance format)** for local training data storage and vector search.

**Secondary: Parquet** for HuggingFace dataset distribution (keep existing pipeline).

**Rationale**:
1. **Training data**: LanceDB provides fast random access (critical for shuffling), direct PyTorch DataLoader integration, and built-in vector search — all of which are core requirements.
2. **Distribution**: Parquet remains the standard for sharing datasets via HuggingFace Hub. The existing pipeline for this works well.
3. **Vector search**: LanceDB's built-in vector indexes could simplify the stack by replacing Milvus for smaller to medium-scale deployments, while Milvus could still be used for large-scale production.

---

## Part 3: Integrated Recommendation

### Proposed Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    Pipeline Orchestration                        │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │  Ray (Compute Layer)                                      │   │
│  │  ┌─────────┐   ┌──────────┐   ┌──────────────────────┐   │   │
│  │  │Ray Data │ → │Ray Train │ → │   Model Artifact     │   │   │
│  │  │ (ETL)   │   │(TF Keras)│   │   (SavedModel/.h5)   │   │   │
│  │  └─────────┘   └──────────┘   └──────────────────────┘   │   │
│  └──────────────────────────────────────────────────────────┘   │
│                                                                  │
│  Optional: Prefect/Dagster for scheduling & lineage              │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│                    Storage Layer                                 │
│  ┌───────────────────────┐   ┌──────────────────────────────┐  │
│  │  LanceDB (Local)      │   │  Parquet (HuggingFace Hub)    │  │
│  │  ┌─────────────────┐  │   │  ┌────────────────────────┐  │  │
│  │  │ Training data    │  │   │  │ Distributed datasets    │  │  │
│  │  │ - Token sequences│  │   │  │ - train/                │  │  │
│  │  │ - Tensor boards  │  │   │  │ - test/                 │  │  │
│  │  │ - Embeddings     │  │   │  │ - version tags          │  │  │
│  │  │ - Metadata       │  │   │  └────────────────────────┘  │  │
│  │  └─────────────────┘  │   │                                │  │
│  │  ┌─────────────────┐  │   │  Milvus (Production Vector DB)│  │
│  │  │ Vector indexes   │  │   │  ┌────────────────────────┐  │  │
│  │  │ (optionally      │  │   │  │ Large-scale similarity  │  │  │
│  │  │  replacing       │  │   │  │ search                  │  │  │
│  │  │  Milvus)         │  │   │  └────────────────────────┘  │  │
│  │  └─────────────────┘  │   │                                │  │
│  └───────────────────────┘   └──────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
```

### Migration Path

| Phase | Action | Impact |
|---|---|---|
| **1. Immediate** | Keep current stack (Ray + .npz + Parquet + Milvus) | No changes needed |
| **2. Short-term** | Evaluate LanceDB for local training data: replace `.npz` files with Lance tables | Improved shuffle performance, simpler storage, optional Milvus replacement for dev |
| **3. Medium-term** | Add Prefect (minimal) for scheduled retraining and pipeline observability | If automated retraining or event-driven workflows are needed |
| **4. Long-term** | Consider Delta Lake for HuggingFace datasets if versioning/ACID becomes critical | Only if concurrent writes or time travel are needed on distributed datasets |

### Key Decision Factors

1. **Ray is already in use** — No new compute framework needed. The project has already made the right choice.

2. **LanceDB fills a real gap** — The current `.npz` storage is the weakest link. LanceDB directly addresses random access, ML framework integration, and vector search.

3. **Don't over-engineer** — The project doesn't need Kubernetes, Spark clusters, or multi-component orchestration yet. Ray handles the compute; LanceDB handles the storage; Milvus handles vector search at scale.

4. **Keep Parquet for distribution** — The HuggingFace ecosystem standardizes on Parquet. The existing pipeline works and should be preserved.

### Technology Comparison Summary

| Technology | ML-native | TF/Keras Support | In Project | Scalability | Ease of Use | Recommendation |
|---|---|---|---|---|---|---|
| **Ray** | Yes | Yes (TensorflowTrainer) | Yes | Excellent | Medium | **Use as primary** |
| Dagster | Partial | Via Python | No | Good | Medium | Optional complement |
| Flyte | Yes | No native | No | Excellent | Hard | Overkill |
| Kedro | Yes | No native | No | Limited | Easy | Not a good fit |
| Prefect | No | No native | No | Good | Easy | Optional complement |
| PySpark | No | No | No | Excellent | Hard | Not recommended |

| Technology | ML DataLoader | Random Access | Versioning | Vector Search | Maturity | Recommendation |
|---|---|---|---|---|---|---|
| **LanceDB** | Yes (PyTorch) | Excellent | Yes | Yes (built-in) | Newer | **Use for local training** |
| Parquet | Via HF Datasets | Poor | No | No | Excellent | Keep for distribution |
| Delta Lake | Via HF Datasets | Poor | Yes | No | Good | If versioning needed |
| .npz files | Manual | None | No | No | Excellent | **Phase out** |
