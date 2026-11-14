# Text embedding survey and simplified chess embedding plan (2026-11-10)

## Status

Analysis and plan. No new training run has been started from this plan.

Current code state:

- Branch: `feature/practical-alternative-visible-content`
- Commit: `e0941d8` — visible-content identity loss, tests, and SVG
- The practical alternative is implemented but has not been trained.
- The Perceiver experiments are stopped and the GPU pod is decommissioned.

## Executive summary

The leading MTEB text embedding models are **architecturally simpler** than the
current chess embedding pipeline. They are usually single-tower bi-encoders with
one pooled vector, L2 normalization, and a contrastive objective. The complexity
lives in data, positive/negative construction, hard-negative mining, large
batches, and multi-stage training.

The chesspos experiments have already tested many of the individual ingredients:
MLM, InfoNCE, VICReg, invariance, SupCon, content/state isolation, attention
pooling, a Perceiver bottleneck, retrieval evaluation, and clustering analysis.
The main missing pieces are:

1. a **simple one-board, single-tower embedding baseline**;
2. **validated hard negatives used in the loss**;
3. **large effective batches**;
4. a clean **pretrain → contrastive fine-tune** separation;
5. optional **task prefixes, MRL, and distillation**.

Material smearing is also expected in a lossy semantic embedding. It matters for
material-aware tasks, but it does **not** imply that the embedding must be
reconstructable. Retrieval, clustering, classification, and reranking can store
the original FEN/tokens as payload and use metadata or heads for material.

The recommended next step is not another architecture. It is a simpler training
contract:

```text
one board -> one encoder -> one pooled vector -> L2 norm -> contrastive loss
```

Keep the tokenizer, data ETL, retrieval benchmark, hard-negative candidate
schema, evaluation probes, and collapse guards. Drop the dual content/state
trunks, Perceiver bottleneck, joint multi-objective training, and packed-window
contrastive views for the first baseline.

## Part 1 — What MTEB-leading text embedding models do

### MTEB task types

MTEB evaluates embeddings across:

- Retrieval
- Reranking
- Clustering
- Classification
- STS
- PairClassification
- Summarization
- InstructionRetrieval

A good embedding model is therefore not only a retrieval model. It must produce
a broadly useful cosine-similarity space.

Sources: [MTEB](https://github.com/embeddings-benchmark/mteb)

### Model comparison

| Model | Backbone | Pooling | Dimension | Context | Training pattern |
|---|---|---|---|---|---|
| Qwen3-Embedding-0.6B | Decoder-only Qwen3 | Last token | Up to 1024, MRL | 32k | Instruction-aware contrastive with hard negatives |
| Harrier-OSS-v1-0.6B | Decoder-only | Last token | 1024 | 32k | Contrastive + distillation from larger embedding models |
| Llama Nemotron Embed 1B v2 | Llama 3.2 encoder | Mean / average | Up to 2048, MRL | 8k | Bi-encoder contrastive + hard-negative mining |
| Gemini Embedding 2 | Proprietary | Proprietary | 128–3072, MRL | 8k | Task prefixes; training details proprietary |

Sources:

- [Qwen3-Embedding-0.6B](https://huggingface.co/Qwen/Qwen3-Embedding-0.6B)
- [Harrier OSS](https://huggingface.co/microsoft/harrier-oss-v1-0.6b)
- [Llama Nemotron Embed](https://huggingface.co/nvidia/llama-nemotron-embed-1b-v2)
- [Gemini Embeddings](https://ai.google.dev/gemini-api/docs/embeddings)

### Shared recipe

1. **Start from a strong pretrained backbone.**
   - Qwen3 and Harrier start from decoder-only language models.
   - Llama Nemotron fine-tunes Llama 3.2.
   - Embedding training is usually a second stage, not the original pretraining.

2. **Use a bi-encoder.**
   - Query and document are encoded independently.
   - There is no task-specific dual trunk, reconstruction bottleneck, or
     per-square decoder in the main embedding model.

3. **Pool to one vector.**
   - Decoder-only models commonly use last-token pooling.
   - Encoder models commonly use CLS or mean pooling.
   - The pooled vector is projected and L2-normalized.

4. **Train with contrastive InfoNCE.**
   - In-batch negatives.
   - Explicit hard negatives.
   - Large effective batch.
   - Temperature around `0.05`.
   - Qwen3 uses an improved InfoNCE with hard negatives and a false-negative
     mask: if a negative is more similar than the positive plus a margin, it is
     masked from the denominator.

5. **Use query/document or task prefixes.**
   - Qwen3: instruction on the query side; documents raw.
   - Harrier: query prompts such as `web_search_query`.
   - Llama Nemotron: `query:` vs `passage:`.
   - Gemini: `task: search result | query: ...` and document templates.
   - Instructions typically improve retrieval by roughly 1–5%.

6. **Mine hard negatives and iterate.**
   - Llama Nemotron explicitly uses hard-negative mining.
   - Qwen3 uses hard negatives and false-negative masking.
   - This is usually more valuable than adding architectural complexity.

7. **Use large effective batches.**
   - Sentence Transformers recommends batch size as a quality knob for
     `MultipleNegativesRankingLoss`.
   - `CachedMultipleNegativesRankingLoss` / gradient caching allows effective
     batches of 512–1024+ without fitting everything in VRAM.

8. **Distill from a stronger model or reranker.**
   - Harrier distills from larger embedding models.
   - Llama Nemotron and NV-Retriever use hard-negative mining and reranking.
   - A cross-encoder can provide both a reranker and distillation targets.

9. **Use MRL for flexible dimensions.**
   - Qwen3, Llama Nemotron, and Gemini support truncatable embeddings.
   - This is useful for vector databases and storage trade-offs.

10. **Evaluate broadly.**
    - Retrieval is not enough; MTEB includes clustering, classification, STS,
      pair classification, reranking, and instruction retrieval.

Sources:

- [Sentence Transformers training](https://github.com/huggingface/sentence-transformers)
- Qwen3 technical report: <https://arxiv.org/abs/2506.05176>

## Part 2 — What chesspos has already done

### Early experiments

From [previous_experiments.md](previous_experiments.md):

- Dense, CNN, and shallow/deep autoencoders on bitboard and tensor encodings.
- Mode collapse: most positions reconstructed as an empty board.
- Regularizers reduced collapse but did not make reconstruction usable.
- Skip connections improved dense autoencoders significantly.
- Real-game data did not fix the model-capacity problem.
- The conclusion was to move to transformers.

### Transformer baseline

`run_6m_isolated`:

- 8 layers, `d_model=256`, `d_ff=1024`, 16 heads, 6.4M parameters.
- Batch 128.
- MLM + content-isolated InfoNCE + invariance + VICReg.
- 50% rectangle / 50% skip-gram masking.
- Stopped at 72k because semantic metrics plateaued.
- Material was present in early layers but not in the final pooled vector.
- Visible content positions were not supervised to preserve their own identity.
- Recommended fix: visible-content identity loss.

Source: [analysis_run_6m_isolated_plateau_2026-11-09.md](analysis_run_6m_isolated_plateau_2026-11-09.md)

### Objective sweep

From [experiment_objective_sweep_2026-10-25.md](experiment_objective_sweep_2026-10-25.md):

- VICReg variance was the strongest anti-collapse lever.
- InfoNCE alone did not fix token collapse.
- VICReg + NCE hurt strategic MAP at tiny scale.
- Invariance was near-neutral at the tested weights.
- Masking choice was neutral.
- The all-objectives defaults were a compromise, not a clear winner.

This is consistent with the text embedding lesson: **collapse is better solved by
the right data/objective contract than by stacking auxiliary losses**.

### Perceiver experiments

From [perceiver_6m_lr2e4_2026-11-10.md](perceiver_6m_lr2e4_2026-11-10.md):

- Segment-aware Perceiver IO with content/state latent banks.
- Initial content collapse was fixed with small segment-embedding init and
  decoded-slot attention pooling.
- Extending the run from 72k to 200k changed the cosine schedule and caused a
  train-loss explosion at ~33.4k.
- The resume run used a lower peak LR, eval every 2k, and an NCE guard.
- Train loss stayed stable, but validation drifted after 34k and material R²
  stayed negative (`-0.50` to `-0.67`).
- Best validation loss was `0.8510` at step 34k.
- The Perceiver is not proven broken, but its content path is not learning
  material.

Source: [analysis_perceiver_content_collapse_2026-11-09.md](analysis_perceiver_content_collapse_2026-11-09.md)

### Practical alternative

Implemented in commit `e0941d8`:

- Keeps the existing transformer and content-isolated trunk.
- Adds visible-content identity loss:
  `CE(head(h_i), token_i)` at visible content positions.
- Reuses the shared vocabulary head.
- Keeps mean-pooled retrieval; no unpooling.
- Opt-in via `--visible-identity-weight`.

This is a useful diagnostic and a direct response to the `run_6m_isolated`
finding, but it keeps the dual-trunk and multi-objective complexity. It has not
been trained yet.

### Evaluation already in place

- Retrieval benchmark:
  - near-duplicate queries;
  - strategic/theme queries;
  - per-theme MAP/recall;
  - hard-negative candidate export.
- Material R² and side-to-move probes.
- PC1/PC2 piece correlation.
- UMAP/HDBSCAN clustering analysis.
- Token and retrieval effective rank.
- NCE collapse guard.

Not implemented:

- Reranking task or reranker.
- Full MTEB-style classification/STS task suite.
- Instruction prefixes.
- MRL.
- Large-batch gradient caching.
- Distillation.

Source: [umap_embedding_structure.md](umap_embedding_structure.md)

### Mapping to the text embedding recipe

| Recommendation | Tried? | Result |
|---|---|---|
| Single bi-encoder | Partially | Transformer and Perceiver used dual content/state paths; no plain single-tower baseline. |
| CLS / mean / last-token pooling | Partially | Mean, attention, slot, and per-token paths tried; no CLS-token embedding. |
| L2 normalization | Yes | Retrieval uses normalized content embeddings. |
| InfoNCE | Yes | Tried alone and with VICReg; mixed results; NCE alone did not fix collapse. |
| In-batch negatives | Yes, small | Batch 128–256; no 512+ effective batch. |
| Explicit hard negatives | **No** | Candidates saved, never used in the loss. |
| False-negative filtering | **No** | No validation layer. |
| Positive mining | Yes | Temporal/SupCon positives, window/alone pairs, state-invariance positives. |
| Query/document prefixes | **No** | No instructions or task prefixes. |
| Encoder/decoder backbones | Partially | Encoder transformer and Perceiver encoder-decoder; no decoder-only LLM. |
| Multi-stage pretrain → contrastive | **No** | MLM is joint with contrastive/auxiliary losses. |
| Large batches / GradCache | **No** | Not implemented. |
| Distillation / reranker | **No** | Not implemented. |
| MRL | **No** | Not implemented. |
| Retrieval eval | Yes | Near and strategic retrieval benchmark. |
| Clustering eval | Yes | UMAP/HDBSCAN; theme purity weak. |
| Reranking eval | **No** | Not implemented. |
| Classification/STS eval | Partially | Material/side probes only. |
| One-board-per-example data | **No** | Training uses packed windows. |

## Part 3 — Material smearing and reconstructibility

The earlier experiments found that material is not linearly recoverable from the
pooled content embedding
([run_6m_isolated plateau](analysis_run_6m_isolated_plateau_2026-11-09.md)).
That is a real limitation for material-aware tasks, but it is not evidence that
the embedding must reconstruct the board.

Text embedding models behave the same way. A sentence embedding is a lossy
semantic summary; it does not reconstruct the original sentence. RAG systems
store the original documents alongside the vectors. Chess should do the same:
store the FEN/tokens as the payload and use the vector as a semantic index.

### Three different requirements

| Requirement | What it means | Typical use |
|---|---|---|
| Similarity preservation | Nearby vectors mean similar positions | Search, retrieval, clustering, classification |
| Attribute preservation | A specific property (material, piece count, side-to-move) is recoverable | Material-aware search, probes, filters |
| Full reconstruction | The exact board can be decoded from the vector | Generation, editing, latent simulation, compression |

Material smearing is a failure of **attribute preservation**, not necessarily of
similarity preservation. Full reconstruction is a much stronger requirement.

### Why material smearing is normal

- The contrastive objective only preserves information that helps separate
  positives from negatives.
- A detail such as material is discarded or entangled if it does not help that
  separation.
- Vector capacity is usually not the bottleneck; the **training objective** is.
- Forcing reconstruction can make the embedding less useful for similarity,
  because it spends capacity on details irrelevant to retrieval.

If material is important for search, the right fix is to supervise it or filter
on it, not to require full board reconstruction.

### When reconstruction is actually needed

| Use case | Need reconstruction? | Why / alternative |
|---|---|---|
| Board-to-board similarity search | **No** | Store FEN/tokens as payload; return the stored board. |
| Text-to-board retrieval | **No** | Retrieval returns stored boards. |
| Clustering positions | **No** | Clustering works on the lossy vector. |
| Classification of positions | **No** | Use the vector or a task-specific head. |
| Reranking candidates | **No** | Rerank with a cross-encoder using the full board. |
| Deduplication | **No** | Use canonical FEN / Zobrist hash. |
| Material-aware search | **No** | Compute material from the stored FEN or add a material filter/head. |
| Vector database storage | **No** | Store vector + FEN + metadata, never the vector alone. |
| Lossy compression / embedding-only storage | **Yes** | Not recommended; store the board instead. |
| Move generation / legality checks | **Yes** | Exact board state is required; use the FEN. |
| Engine evaluation from embedding only | **Yes** | Exact position is required; store the board. |
| Generative modeling / board editing | **Yes** | Sampling or interpolating embeddings needs a decoder. |
| Latent-space planning / RL | **Depends** | A latent dynamics model may suffice, but it must preserve transition-relevant state. |
| Board completion from a partial vector | **Yes** | Needs a decoder or a separate generative model. |
| Debugging / interpretability | **Yes** | Reconstruction is useful for diagnostics, not production. |

### Recommendation for chesspos

Treat the embedding as a **semantic index**, not as the source of truth.

- Store `FEN`/tokens and metadata alongside the vector:
  - material
  - piece count
  - phase
  - side-to-move
  - themes
- For material-aware queries, use hybrid retrieval:
  - vector recall;
  - metadata/material filter;
  - optional reranker.
- If material is a core semantic for search, add a material auxiliary head or
  material-balanced positives/negatives.
- Do **not** add a reconstruction decoder to the retrieval model unless the
  product actually needs board generation or editing.
- If generation/editing is needed, train a separate decoder or generative model.

This means the negative material R² is not by itself a reason to reject the
embedding. It defines which tasks the embedding supports, and which tasks need
extra supervision or metadata.

## Part 4 — Recommendations

### Keep

- `TokenStreamEncoder` and the token-stream format.
- Existing ETL and parquet data.
- Retrieval benchmark and existing probes.
- Hard-negative candidate schema.
- NCE collapse guard and periodic eval.
- Existing MLM checkpoint as an initializer.
- UMAP/clustering analysis as an evaluation tool.

### Drop for the first baseline

- Dual content/state trunks.
- Perceiver bottleneck.
- Joint MLM + InfoNCE + invariance + VICReg + auxiliary heads.
- Packed multi-position windows.
- Alone views and masking at contrastive time.
- Auxiliary heads in the embedding model.

### New baseline

```text
one board
   │
   ▼
single transformer encoder
   │
   ▼
CLS / mean pooling
   │
   ▼
linear projection
   │
   ▼
L2 normalize
   │
   ▼
InfoNCE with in-batch + validated hard negatives
```

Recommended first configuration:

- 6–12 layers, `d_model=384–512`, 6–8 heads.
- Max length ~80 tokens: `[CLS] + 64 squares` (+ optional state prefix).
- CLS pooling; mean pooling as an ablation.
- Projection to 256–512 dimensions.
- L2 normalization.
- InfoNCE temperature `0.05`.
- Batch 512–1024 effective negatives; gradient caching if needed.
- Initialize from the existing MLM checkpoint or run a short MLM pretraining.
- No VICReg/invariance initially; add only if collapse appears.
- No packed windows for contrastive training.

### State handling

Start with **content-only** embeddings:

- omit turn/castle/en-passant;
- this removes the leakage problem entirely;
- retrieval is about board content.

If a task needs state:

- prepend a task prefix such as `turn: white | castle: KQ | ep: -`;
- or train a separate state-aware variant;
- do not reintroduce a second trunk.

### Training stages

1. **Baseline contrastive**
   - one board per example;
   - InfoNCE with random in-batch negatives;
   - evaluate retrieval + clustering + material/side probes.

2. **Validated hard negatives**
   - mine candidates;
   - filter false negatives;
   - add 1–4 hard negatives per anchor;
   - use false-negative masking in the loss.

3. **Task prefixes**
   - `task: similar position | board: ...`
   - `task: same tactical theme | board: ...`
   - `task: material query | board: ...`

4. **MRL**
   - train 512/256/128/64-d prefixes.

5. **Distillation**
   - train a cross-encoder/reranker;
   - distill its scores into the bi-encoder.

## Part 5 — Hard-negative validation

The current hard-negative export is a good start, but it only contains candidates.
A hard negative is not automatically a true negative.

### Define the target first

The definition depends on the task:

- Exact/near-duplicate retrieval:
  - transpositions, same game within 1–2 plies, same puzzle ID are positives.
- Tactical-theme retrieval:
  - same theme is a positive even if the board differs.
- Generic position similarity:
  - one-piece-move can be a hard true negative.

### Efficient triage pipeline

```text
candidate (anchor, negative)
        │
        ├─ same canonical position?              -> false negative
        ├─ same game and |Δply| <= r?            -> positive/false negative
        ├─ same puzzle ID or solution?           -> positive/false negative
        ├─ same theme?                           -> task-dependent
        ├─ cross-encoder score > τ?              -> false negative/uncertain
        ├─ engine/material delta < ε?            -> uncertain
        └─ otherwise                             -> true negative
```

#### 1. Deterministic chess equivalence

Use `python-chess` transposition keys / canonical FEN:

- `placement_hash`: content only, ignoring turn/castle/EP.
- `state_hash`: full position including state.

Rules:

- same `placement_hash` → false negative for content retrieval;
- same `state_hash` → exact duplicate;
- same game and small ply delta → near-duplicate positive;
- same puzzle ID/solution → positive;
- same theme → task-dependent.

This is vectorizable with Polars/PyArrow and scales to millions of pairs.

#### 2. Metadata filters

Use the existing candidate fields:

- `game_id`, `anchor_ply`, `negative_ply`
- `piece_distance`
- `anchor_material`, `negative_material`
- `anchor_piece_count`, `negative_piece_count`
- `anchor_turn`, `negative_turn`
- `anchor_opening`, `negative_opening`
- `similarity`

Examples:

- same game + small ply delta → drop from negatives;
- same material/piece count but different structure → good hard true negative;
- one-piece move → hard true negative for exact retrieval, uncertain for generic
  similarity;
- opening positions → ambiguous; exclude or use only as a separate hard set.

#### 3. Cross-encoder adjudication

Train a small cross-encoder:

```text
[anchor board tokens] + [candidate board tokens]
        │
        ▼
same/different or similarity score
```

- label a few thousand pairs using rules plus manual/engine checks;
- score ambiguous candidates;
- `score > τ` → false negative or uncertain;
- `score < τ` → true negative.

This is more reliable than using the current embedding model, which may share
the same blind spots. It also gives a reranker and distillation target.

#### 4. False-negative masking in the loss

This is the cheapest mitigation and is used by Qwen3-style recipes:

```text
if similarity(anchor, negative) > similarity(anchor, positive) + margin:
    mask the negative from the InfoNCE denominator
```

Start with a small cosine margin, e.g. `0.05–0.10`.

#### 5. Calibration and active learning

- Build a calibration set of 2k–5k pairs:
  - duplicates;
  - near-duplicates;
  - one-piece moves;
  - same theme;
  - same material but different structure.
- Manually or engine-check the labels.
- Measure precision/recall of each filter.
- Then loop: mine → filter → train → remine → review only uncertain pairs.

Do not label every candidate. Label the uncertain bucket.

### Suggested storage schema

Extend the hard-negative parquet with:

- `anchor_hash`, `negative_hash`
- `same_placement`, `same_state`
- `same_game`, `ply_delta`
- `same_puzzle`, `theme_overlap`
- `piece_distance`, `material_delta`
- `cross_encoder_score`
- `label`: `positive`, `true_negative`, `false_negative`, `uncertain`
- `filter_reason`

### Implementation plan

1. Add a `HardNegativeFilter` that reads the existing exported parquet.
2. Add canonical placement/state hashes to each anchor/negative.
3. Add metadata filters and a `label` column.
4. Add a small cross-encoder and a calibration script.
5. Write validated pairs to `data/processed/pairs/{train,val}`.
6. Train the baseline with `CachedMultipleNegativesRankingLoss`-style batching
   and false-negative masking.

## Part 6 — Next steps

### Phase 1 — Data contract

- Define the downstream contract:
  - similarity search, clustering, classification, or reranking (no reconstruction);
  - material-aware search (metadata filter or material head);
  - generation/editing (separate decoder, not the retrieval model).
- Build one-board examples from the existing parquet shards.
- Store FEN/tokens plus metadata as the vector-DB payload:
  - material;
  - piece count;
  - phase;
  - side-to-move;
  - themes.
- Define positive pairs:
  - transpositions;
  - same game within 1–2 plies;
  - same puzzle/theme.
- Define negative pools:
  - random boards;
  - one-piece-move boards;
  - same-material different-structure boards.
- Add canonical hashes and metadata filters.

### Phase 2 — Minimal baseline

- New branch: `feature/chess-embedding-baseline`.
- Single-tower transformer encoder.
- CLS pooling, projection, L2 norm.
- InfoNCE with in-batch negatives.
- Initialize from the existing MLM checkpoint.
- Eval every 1k steps on the existing retrieval benchmark plus material and
  side-to-move probes.

### Phase 3 — Hard negatives

- Add validated hard negatives.
- Use false-negative masking.
- Increase effective batch with gradient caching.
- Compare against Phase 2.

### Phase 4 — Task prefixes and MRL

- Add task prefixes if multiple retrieval modes are needed.
- Add Matryoshka dimensions.
- Measure per-task MTEB-style metrics.

### Phase 5 — Distillation and reranking

- Train a cross-encoder/reranker on validated pairs.
- Distill into the bi-encoder.
- Add reranking evaluation to the chess benchmark.

### Success criteria

- Beats the best current checkpoint on retrieval without collapse.
- Improves material R² while keeping side-to-move leakage controlled.
- Remains stable at 20k–50k steps with a clean LR schedule.
- Requires fewer model paths and fewer loss terms.
- Hard-negative precision is measured, not assumed.

## Risks

- Contrastive-only training may collapse without VICReg; monitor NCE, cosine
  similarity, and effective rank.
- Content-only embeddings may lose state; add a prefix or separate model only if
  a task needs it.
- False negatives can dominate hard-negative training; use filtering and
  false-negative masking.
- Data quality and pair definition matter more than architecture.
- Cross-encoder labeling adds cost; start with deterministic filters and a small
  calibration set.
- Changing `--steps` changes the cosine schedule; treat schedule changes as new
  runs.

## References

- MTEB: <https://github.com/embeddings-benchmark/mteb>
- Qwen3-Embedding: <https://huggingface.co/Qwen/Qwen3-Embedding-0.6B>
- Qwen3 technical report: <https://arxiv.org/abs/2506.05176>
- Harrier OSS: <https://huggingface.co/microsoft/harrier-oss-v1-0.6b>
- Llama Nemotron Embed: <https://huggingface.co/nvidia/llama-nemotron-embed-1b-v2>
- Gemini Embeddings: <https://ai.google.dev/gemini-api/docs/embeddings>
- Sentence Transformers: <https://github.com/huggingface/sentence-transformers>

## Related chesspos reports

- [Previous experiments](previous_experiments.md)
- [Objective sweep](experiment_objective_sweep_2026-10-25.md)
- [run_6m_isolated plateau](analysis_run_6m_isolated_plateau_2026-11-09.md)
- [Perceiver content collapse](analysis_perceiver_content_collapse_2026-11-09.md)
- [Perceiver 6M LR schedule and validation drift](perceiver_6m_lr2e4_2026-11-10.md)
- [UMAP embedding structure](umap_embedding_structure.md)
