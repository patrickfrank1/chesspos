# Plan: Observability for the Position Similarity Objective

- **Status:** Proposed
- **Date:** 2026-11-07
- **Scope:** Standalone evaluation tooling for a trained encoder: clustering,
  similarity metrics, saliency/explanation, and a self-contained HTML report.
- **Out of scope (v1):** Training-loop integration, motif-classifier saliency,
  `pytorch-concepts` concept-bottleneck models, FloDR backend.

## 1. Goal

Make the position-similarity objective observable on two datasets:

1. **Random test batch** — unlabeled positions from the test split. Used to
   judge the global embedding geometry and its agreement with structural
   surrogates (material, pawn structure, open files, phase, …).
2. **Tactical-motif dataset** — Lichess puzzle positions labelled by tactical
   themes. Used to judge whether positions sharing a motif are actually close
   in embedding space, beyond trivial structural similarity.

The output should let a human answer:

- Do tactical motifs form clusters?
- Which themes separate, and which collapse together?
- What do nearest neighbours look like?
- Which squares/pieces drive a given similarity?
- Are there outliers or degenerate regions?
- Does the embedding capture material, pawn structure, open files, and phase?

## 2. Research findings (current state)

### Existing evaluation

- `src/evaluation/eval_suite.py` — MLM breakdown, neighbour statistics,
  counterfactual similarity, material/side-to-move probes, PCA phase plot, and
  `neighbor_examples.html` (built with `chess.svg.board`).
- `src/evaluation/retrieval.py` — frozen retrieval benchmark for near-duplicate
  and theme-cluster queries; ranking metrics, alignment/uniformity.
- `src/run/evaluate_retrieval.py` — standalone retrieval-benchmark evaluation.
- `src/run/eval_model.py` — standalone checkpoint evaluation. No clustering,
  no saliency, no interactive report.

### Datasets

- **Test split:** the HF `test` split; local shards exist under
  `data/processed/hub/test/` with columns `packed`, `n_positions`, `game_id`,
  `white_elo`, `black_elo`, `result`. Current loaders discard game/ply
  provenance; the new evaluation must preserve it.
- **Tactical motifs:** already on `main`. The `features/puzzle-classification`
  branch is fully merged. `src/run/generate_puzzle_dataset.py` builds
  `data/processed/puzzles` from `data/raw/lichess_db_puzzle.csv` (present,
  1.1 GB). Each row has `puzzle_id`, `previous_position`, `packed` (one
  segment per position), `rating`, and one `int8` column per Lichess theme.
- **Label caveat:** themes are puzzle-level multi-labels, not per-position
  labels. Positions from one puzzle are near-duplicates, so naive clustering
  over all positions clusters by puzzle and inflates theme purity. The v1
  default therefore samples **one position per puzzle**.

### Tooling compatibility

| Tool | Status |
|---|---|
| `umap-learn` | installed (0.5.12), existing dev dependency |
| `scikit-learn` | installed (1.9.0) |
| `hdbscan` | not installed; PyPI 0.8.44 supports Python 3.11 |
| `shap` | not installed; latest needs Python ≥3.12 + numpy ≥2 → pin `shap>=0.46,<0.49` |
| `plotly` | not installed; 7.x supports Python 3.11 |
| `flodr` | not installed; 0.2.2 supports Python 3.11, but deferred |
| `pytorch-concepts` | 1.0.0a7 alpha, heavy deps; not a post-hoc attribution tool; skipped |

## 3. Decisions (locked)

- **Puzzle sampling:** one position per puzzle, **first segment**
  (`packed` position after the puzzle's first move; `previous_position` is not
  used).
- **Explanation scope:** pairwise/centroid similarity + structural concepts.
  No motif-classifier logits and no PyC in v1.
- **Output:** single self-contained HTML report.
- **Integration:** standalone CLI first; no training-loop hook in v1.
- **FloDR:** deferred until after the first report; the reducer interface stays
  open for it.

## 4. Architecture

```
checkpoint + test/puzzle parquet
    -> samples.py     PositionSample, loaders, surrogates, deterministic sampling
    -> loading.py     checkpoint auto-detect (foundation vs classifier)
    -> embeddings.py  batched, normalized embeddings + cache
    -> reduce.py      Reducer protocol: PCA, UMAP (FloDR later)
    -> cluster.py     HDBSCAN + label/surrogate metrics + stability
    -> explain.py     occlusion, SHAP/IG, structural concept probes / TCAV
    -> render.py      board rendering + Plotly scatter + self-contained HTML
src/run/evaluate_similarity.py   CLI orchestrator
```

New package:

```
src/evaluation/observability/
  __init__.py
  samples.py
  loading.py
  embeddings.py
  reduce.py
  cluster.py
  explain.py
  render.py
```

## 5. Phase 0 — Dependency spike

Add an optional dependency group so core/training installs stay lean:

```toml
[project.optional-dependencies]
observability = [
  "scikit-learn>=1.6",
  "umap-learn>=0.5.11",
  "hdbscan>=0.8.44",
  "shap>=0.46,<0.49",   # py3.11-compatible
  "plotly>=5.24",
]
```

Do **not** add `flodr` or `pytorch-concepts`.

Verify:

- `uv lock` succeeds and composes with `--extra cpu` and `--extra gpu`;
- `hdbscan`/`shap`/`plotly` resolve against numpy 2.3 and torch;
- `shap.GradientExplainer` runs on a tiny encoder with the installed torch;
- UMAP's first-run numba compilation is acceptable;
- no pre-commit/ruff/CI conflicts.

No production code in this phase.

## 6. Phase 1 — Position samples and loaders

### 6.1 Data model

```python
@dataclass(frozen=True)
class PositionSample:
    sample_id: int
    tokens: tuple[int, ...]      # 67-token segment
    fen: str
    source: str                  # "test" | "puzzle" | "custom"
    game_id: str | None
    puzzle_id: str | None
    ply: int | None
    themes: tuple[str, ...]
    rating: int | None
    surrogates: dict[str, float]
```

### 6.2 Test-split loader

- Read parquet rows directly; preserve `game_id` and per-position ply.
- Flatten each game's `packed` stream with the existing segment splitting
  utility; decode each segment to a FEN with `TokenStreamEncoder`.
- Deterministic reservoir sampling: `--n-test`, `--per-game-cap`, `--seed`.
- Optionally stratify by phase or material bins for a balanced random batch.

### 6.3 Puzzle loader

- Read `puzzle_id`, `packed`, theme columns, and `rating` from parquet shards.
- `--puzzle-position first` (locked default) → `segments[0]`; skip empty
  sequences.
- Attach the puzzle's full multi-hot theme set to the sampled position.
- Filter `--themes`, cap per theme, and optionally balance themes.
- Add `--custom-parquet` with column mapping for a possible unmerged branch
  with a different schema.

### 6.4 Structural surrogates

Derive from `chess.Board`:

- material balance, piece count, phase, side to move, castling, en passant;
- pawn-structure hash, isolated/doubled/passed pawns, pawn islands;
- open/semi-open files, rook on open file;
- centre occupancy, bishop pair, king safety, mobility.

These become the "surrogate labels" for unlabeled clustering checks.

### 6.5 Tests (write first, Gherkin Given/When/Then)

`test/unit/test_observability_samples.py`:

- test split load preserves `game_id`/ply/FEN and sampling is deterministic;
- first-position-per-puzzle produces exactly one sample per puzzle;
- theme filtering and multi-label attachment work;
- hand-checked surrogates on known boards (start position material 0, pawn
  islands, open files).

## 7. Phase 2 — Checkpoint loading, embeddings, CLI skeleton

### 7.1 Checkpoint loading

`loading.load_encoder(path, device)` auto-detects:

- foundation checkpoint: `model_config` + `model_state`;
- classifier checkpoint: `encoder_config` + `encoder_state`.

Return an eval-mode encoder with gradients disabled by default. Expose a
grad-enabled path for explanations.

### 7.2 Embeddings

- Reuse `eval_suite.embed_segments` / `retrieval.embed_positions`; both respect
  `content_pooling`.
- L2-normalize; keep float32 on CPU.
- CUDA bf16 autocast during extraction; restore train/eval state afterwards.
- Cache `embeddings.npy` + `samples.parquet`; recompute only on `--force`.

### 7.3 CLI skeleton

`src/run/evaluate_similarity.py`:

```bash
uv run python -m src.run.evaluate_similarity \
  --checkpoint data/models/run_d512l8_20261021_streaming/best_step40000.pt \
  --test-dir data/processed/hub/test \
  --puzzles data/processed/puzzles \
  --themes fork,pin,skewer,sacrifice,discoveredAttack,backRankMate \
  --n-test 5000 --n-puzzles 5000 --puzzle-position first \
  --reducer umap --clusterer hdbscan \
  --explain occlusion,shap,concept --explain-examples 20 \
  --device cuda --bf16 \
  --out-dir data/eval/similarity/<run>
```

Artifacts:

```
out/
  config.json
  metrics.json
  samples.parquet
  embeddings.npy
  coords.parquet
  clusters.parquet
  explanations/*.json
  report.html
```

### 7.4 Tests

- checkpoint auto-detection on synthetic foundation/classifier checkpoints;
- embedding shape, normalization, and content/mean pooling paths;
- CLI creates all artifacts on tiny local data, with no network access.

## 8. Phase 3 — Clustering and similarity metrics

### 8.1 Reducers

`reduce.py` defines a `Reducer` protocol and implementations:

- `pca` — deterministic baseline;
- `umap` — `UMAP(n_neighbors=30, min_dist=0.0, n_components=10,
  metric="cosine", random_state=...)` for clustering, plus a separate 2-D fit
  for display.
- Cluster on raw/10-D embeddings, **not** on 2-D UMAP; 2-D is for display only.
- Leave `flodr` as a documented future adapter.

### 8.2 Clusterer

- `hdbscan.HDBSCAN(prediction_data=True)`.
- Adaptive `min_cluster_size` from N, plus a small sensitivity sweep
  (`min_cluster_size ∈ {10, 25, 50, 100}`).
- Report labels, `probabilities_`, `outlier_scores_`, persistence, noise
  fraction, and cluster-size distribution.
- `approximate_predict` for new query positions.

### 8.3 Metrics

**Unlabeled (test batch):**

- silhouette (cosine), Calinski-Harabasz, Davies-Bouldin;
- HDBSCAN noise fraction and outlier rate;
- stability: AMI across seeds;
- UMAP trustworthiness/continuity.

**Motif labels:**

- NMI, ARI, homogeneity, completeness, V-measure;
- cluster purity and per-theme enrichment/lift;
- group-aware variant keyed by `puzzle_id`;
- theme coverage per cluster and cluster concentration per theme.

**Surrogates:**

- NMI/ARI of surrogate bins vs clusters;
- kNN agreement on surrogate values;
- Spearman correlation between pairwise embedding distance and surrogate
  distance on sampled pairs;
- ridge/logistic probes for material, pawn structure, open files, phase.

**Similarity objective (core):**

- cross-puzzle same-theme vs different-theme similarity AUC, with negatives
  matched on material/pawn/open-file bins;
- grouped-CV theme probe AUROC;
- kNN theme precision@k / mAP;
- partial effect of cosine similarity on same-theme after controlling for
  surrogate differences;
- test-vs-puzzle domain classifier AUC (shared manifold check).

### 8.4 Tests

`test/unit/test_observability_cluster.py`:

- separated Gaussian blobs → high ARI and low noise;
- HDBSCAN `-1` noise excluded from label metrics and reported separately;
- NMI/ARI match sklearn on a hand example;
- exact surrogate kNN agreement on toy data;
- AMI=1 for identical clusterings and < 1 for shuffled labels.

## 9. Phase 4 — Explanations

### 9.1 Occlusion (primary)

For a query–neighbour pair or theme centroid:

- replace each of the 64 squares with `EMPTY` (optionally each state token);
- batch all 67 perturbations in one forward pass per example;
- measure Δcosine against the original score;
- aggregate to per-square and per-piece scores.

This is model-agnostic, cheap, and maps directly to highlighted pieces.

### 9.2 SHAP / Integrated Gradients (secondary)

- Add an optional input-embedding path to `MaskedStreamTransformer`:
  `onehot @ token_embedding.weight` → trunk, with structure tokens still
  derived from `argmax`. Raw token-ID gradients are not meaningful, so one-hot
  is required.
- Use `shap.GradientExplainer` on the scalar similarity wrapper;
  `KernelExplainer` only as a slow exact fallback for a few examples.
- Aggregate piece-identity dims per square.
- Run explanations in fp32; explanation batches are small.

### 9.3 Structural concepts / TCAV

- Concept labels derived from the board (not from the themes being evaluated):
  material advantage, isolated/doubled/passed pawns, open file, king safety,
  piece activity, centre control.
- Train logistic probes on frozen embeddings with grouped CV.
- Compute concept activation vectors and TCAV scores for the similarity
  target; report sign-test significance and per-concept direction.
- PyC is intentionally excluded from v1 (alpha, heavy, designed for building
  concept models rather than post-hoc attribution).

### 9.4 Faithfulness

- Deletion/insertion curves: removing top-k attributed squares must lower
  similarity more than removing random-k squares.

### 9.5 Tests

`test/unit/test_observability_explain.py`:

- occlusion returns 64 square attributions and grouped deltas match;
- one-hot input path matches the normal token forward;
- SHAP shapes and finiteness on a tiny model;
- top-k deletion is more faithful than random-k on a toy model;
- TCAV ∈ [0, 1] for a synthetic concept aligned with the target;
- optional backends are guarded with `pytest.importorskip`.

## 10. Phase 5 — Self-contained HTML report

`render.py` produces one `report.html` with inline CSS/JS and no external
requests.

### Sections

1. **Overview** — checkpoint config, dataset sizes, metric tables, warnings.
2. **Embedding map** — Plotly scatter coloured by cluster/theme/surrogate,
   hover shows FEN/IDs, click updates a board panel.
3. **Theme enrichment** — per-theme top clusters, precision@k, lift, example
   boards.
4. **Outliers/noise** — HDBSCAN noise points and high outlier scores.
5. **Query gallery** — query board, top-k neighbour boards, cosine bars,
   themes/rating.
6. **Saliency** — blue (negative) → red (positive) square overlay, highlighted
   pieces, top-piece list, occlusion/SHAP toggle.

### Rendering approach

- Reuse the pure HTML/CSS/JS Unicode-piece renderer pattern already used by
  `data/processed/puzzle_preview.html` so any FEN can be drawn client-side with
  highlight overlays.
- Use `chess.svg.board(fill=...)` for static snapshots/fallbacks.
- Cap interactive points; compute explanations only for `--explain-examples`.

### Tests

`test/unit/test_observability_report.py`:

- HTML contains FEN, boards, and highlights;
- no external URLs/CDNs;
- deterministic output for a fixed seed;
- labels/FENs escaped correctly;
- empty/noise-only clusters render a clear empty state.

## 11. Phase 6 — Docs and follow-ups

- Add an observability runbook to `docs/` and the `vastai-training` skill.
- Document the theme-label caveat: labels are puzzle-level, not per-position;
  first-position sampling reduces but does not remove noise.
- Follow-ups: FloDR reducer, motif-classifier saliency, PyC concept-bottleneck,
  and an optional training-loop hook.

## 12. BDD test inventory

All tests follow the repo's Given/When/Then plain-pytest style and are written
before implementation.

- `test/unit/test_observability_samples.py`
- `test/unit/test_observability_cluster.py`
- `test/unit/test_observability_explain.py`
- `test/unit/test_observability_report.py`
- `test/integration/test_evaluate_similarity_cli.py`

Per `AGENTS.md`, failing tests are written first and reviewed by the user
before implementation begins.

## 13. Acceptance criteria

- `uv run python -m src.run.evaluate_similarity ...` runs on CPU with a tiny
  checkpoint in CI and on a real checkpoint/test/puzzle set.
- The report answers: do motifs cluster? which themes separate? what do
  nearest neighbours look like? which squares/pieces drive a similarity? are
  there outliers? does the embedding capture structural surrogates?
- Deterministic with `--seed`, no network, no secrets.
- Optional observability dependencies do not affect core installation or the
  training path.

## 14. Risks and mitigations

| Risk | Mitigation |
|---|---|
| Same-puzzle near-duplicates dominate motif clusters | One position per puzzle; group-aware metrics |
| Theme labels are sequence-level and noisy | Report caveat beside every theme metric |
| UMAP 2-D distances are misleading | Cluster on raw/10-D; 2-D only for display |
| HDBSCAN marks too much as noise on cosine high-dim | Compare raw/10-D, tune `min_cluster_size`, report noise fraction |
| SHAP on categorical token IDs is meaningless | One-hot input path through the embedding matrix; fp32 explanation |
| `content_pooling` ignores state tokens by design | Expect near-zero state attribution; do not treat as a bug |
| New deps break Python 3.11 / numpy pin | Phase 0 spike; pin `shap<0.49`; optional extra only |
| HTML size with many boards | Client-side FEN renderer; cap interactive points |
