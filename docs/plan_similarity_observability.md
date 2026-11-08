# Plan: Focused Position Similarity Ranking

- **Status:** Implemented (focused v2)
- **Date:** 2026-11-07
- **Scope:** Standalone ranking-based observability for a trained encoder.
- **Out of scope:** Clustering, embedding maps, occlusion/SHAP/TCAV,
  checkpoint comparison, and vector-store work.

## 1. Goal

Answer one question with interpretable metrics:

> Does the encoder rank semantically related positions above unrelated ones?

Use a fixed **5-way ranking task**: one positive and four negatives per query.
Report top-1, MRR, mean rank, and chance baselines.

## 2. Three ranking tasks

### 2.1 Test: in-game vs out-of-game

- Query: a test-split position.
- Positive: a random same-game position within `--max-ply-distance` plies
  (default 5); test positions are sampled as contiguous runs per game.
- Negatives: positions from other games.
- Variants: random negatives and surrogate-matched negatives.

This measures the temporal objective the foundation model was trained on.

### 2.2 Motifs

- Query: a puzzle position.
- Positive: a position with the same Lichess theme from a different puzzle.
- Negatives: different-theme positions.
- Variants: random and surrogate-matched negatives.

Theme labels are sequence-level, so the report samples one position per puzzle
(`--puzzle-position first|last|random`).

### 2.3 Surrogates

- Query: a test position (or all positions if no test set).
- Positive: a position in the same structural class from a different
  game/puzzle.
- Negatives: positions in a different class.
- Variants:
  - random negatives;
  - **leave-one-surrogate-out** matched negatives. Matching uses all
    surrogates except the target one, so the target signal is not nullified.

Semantic classes cover material, phase, open files, isolated/passed/doubled
pawns, pawn islands, king safety, mobility, centre control, bishop pair, check,
king attack pressure, knights on a/h files, inactive bishops, and smothered
positions.

## 3. Report

A single self-contained `report.html` with three tabs:

1. Test
2. Motifs
3. Surrogates

Each tab contains:

- a compact metrics table (random/matched, top-1, MRR, mean rank, chance);
- `<details>` dropdowns per theme/class;
- a definition and class threshold for each condition;
- three example buckets per condition: **3 best**, **3 random**, **3 worst**;
- ranking examples showing the query board and all five candidate boards;
- the positive candidate outlined in green.

No Plotly, no clustering, no embedding map, no saliency sections. Raw metrics
and examples remain available in `metrics.json` and
`ranking_examples.json`.

## 4. Implementation map

| Path | Responsibility |
|---|---|
| `src/evaluation/observability/samples.py` | position loading, provenance, surrogates |
| `src/evaluation/observability/surrogates.py` | semantic surrogate classes and leave-one-out matching matrix |
| `src/evaluation/observability/ranking.py` | ranking cases, metrics, rank-5 examples, matched negatives |
| `src/evaluation/observability/embeddings.py` | batched, normalized embeddings and artifacts |
| `src/evaluation/observability/loading.py` | checkpoint loading and metadata |
| `src/evaluation/observability/render.py` | focused tabbed HTML report |
| `src/run/evaluate_similarity.py` | CLI and orchestration |

## 5. Tests

- `test_observability_samples.py` — loaders, provenance, surrogates.
- `test_observability_surrogates.py` — semantic classes and leave-one-out matrix.
- `test_observability_ranking.py` — ranking metrics, random/matched cases,
  theme/temporal/surrogate examples.
- `test_observability_report.py` — tabs, tables, dropdowns, candidate boards,
  escaping, empty states.
- `test_evaluate_similarity_cli.py` — end-to-end artifacts and report.

## 6. Acceptance criteria

- `uv run python -m src.run.evaluate_similarity ...` produces `report.html`,
  `metrics.json`, `samples.parquet`, `embeddings.npy`, and
  `ranking_examples.json`.
- The report shows Test, Motifs, and Surrogates tabs with 5-way ranking metrics
  and examples.
- Matched-negative metrics are reported alongside random-negative metrics.
- Surrogate matching excludes the target surrogate from the matching vector.
- No clustering, occlusion, SHAP, TCAV, Plotly, UMAP, or HDBSCAN dependencies.
