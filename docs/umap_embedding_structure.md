# UMAP Analysis of Chess Position Embeddings

- **Status:** Completed exploratory analysis
- **Date:** 2026-11-08
- **Checkpoint:** `data/models/run_d512l8_p1p3_v1_20261026/last.pt`
  (step 102000, val_loss 0.4355, 512-dimensional content-pooled embeddings)
- **Script:** [`scripts/analysis/umap_embedding_structure.py`](../scripts/analysis/umap_embedding_structure.py)
- **Artifacts:** `data/eval/similarity/umap_experiments_20261108/`
  (git-ignored)

## 1. Summary

This analysis tested five hypotheses about the geometry of the learned
position space:

1. **Phase/manifold structure:** strongly supported.
2. **Tactical-theme islands:** weak overall; only a few locally enriched themes.
3. **Smooth temporal trajectories:** strongly supported.
4. **Dominant structural factors:** phase and pawn/king structure dominate;
   material balance is surprisingly weak in local neighbourhoods.
5. **Cluster-stratified failure regions:** no gross phase-purity failures, but
   the space has a large mixed transition cluster and weak global theme
   separation.

The strongest result is that the embedding is organised primarily as a
**game-progress/phase manifold**, with an almost monotonic ply axis
(Spearman `-0.979` between UMAP dimension 1 and ply). Position-type structure
is clear in openings; middlegame and endgame regions are more mixed.

Tactical themes do not form clean global islands. There is modest local
enrichment for visually distinctive themes such as `promotion` and
`backRankMate`, especially at the **last** puzzle position, but the effect is
far from a clean thematic partition.

All quantitative metrics below were computed on the original 512-dimensional
embeddings. UMAP was used only for visualization and for defining clusters to
stratify diagnostics.

## 2. Data and Method

### Data

| Sample | Positions | Notes |
|---|---:|---|
| Global test | 10,000 | One random position per game, 10,000 games |
| Trajectory test | 9,964 | 500 games, up to 20 contiguous positions per game |
| Puzzles, first position | 1,368 | Puzzles carrying at least one of eight themes |
| Puzzles, last position | 1,368 | Same puzzles, last packed position |

Test positions come from the current 67-token HF test snapshot
(`68131e5bd685023a95e0b8ea1ebe089a721deb99`). The puzzle shard is the local
Lichess-derived shard at `/tmp/obs_report_data/puzzles`. The eight themes are
`fork`, `pin`, `skewer`, `sacrifice`, `discoveredAttack`, `doubleCheck`,
`backRankMate`, and `promotion`.

The **global** test sample is deliberately one position per game to avoid the
temporal-continuity confound. The **trajectory** sample intentionally contains
many contiguous positions per game so that temporal structure can be measured.

### Embeddings and UMAP

- Embeddings are L2-normalized 512-dimensional vectors from the final
  checkpoint.
- UMAP: `metric="cosine"`, `n_neighbors=30`, `min_dist=0.05`,
  `n_components=2`, `random_state=17`, `n_jobs=-1`.
- k-nearest-neighbour diagnostics use `k=15`.
- HDBSCAN: `min_cluster_size=50`, `min_samples=10` for test and puzzle maps;
  `min_cluster_size=100`, `min_samples=10` for the combined test+puzzle map.
- Phase labels: opening `>=24` pieces, middlegame `14-23`, endgame `<14`.
- Surrogate labels follow `src/evaluation/observability/surrogates.py`.

## 3. Experiment 1: Phase and Manifold Structure

### Phase distribution

| Phase | Positions | Mean ply | Median ply |
|---|---:|---:|---:|
| Opening | 5,925 | 20.1 | 18 |
| Middlegame | 2,876 | 56.2 | 54 |
| Endgame | 1,199 | 99.6 | 94 |

### Phase local-neighbour metrics

| Metric | Value |
|---|---:|
| Phase kNN purity | 0.886 |
| Random baseline (query prevalence) | 0.448 |
| Lift | +0.438 |
| Opening purity | 0.936 |
| Middlegame purity | 0.794 |
| Endgame purity | 0.861 |
| Phase silhouette, original | 0.092 |
| Phase silhouette, UMAP 2D | 0.299 |
| UMAP dim 1 vs ply, Spearman | -0.979 |

HDBSCAN found **12 clusters**, with weighted phase purity `0.883`, minimum
cluster phase purity `0.840`, and noise fraction `0.0415`. The UMAP map is
therefore not a collection of arbitrary islands: phase is one of the dominant
organising factors.

The trajectory sample shows how strong game continuity is locally:
`60.8%` of each position's 15 nearest neighbours come from the same game,
against a random baseline of `0.19%` (lift `+0.606`). This is expected from the
temporal training objective, but it also means that a trajectory map is
dominated by within-game progression.

**Interpretation:** the phase/manifold hypothesis is strongly supported. The
embedding organises positions along a continuous game-progress axis, with
openings especially well separated. Game continuity is a major component of
local neighbourhoods.

## 4. Experiment 2: Tactical-Theme Islands

Theme values are sequence-level puzzle labels. The table reports local kNN
purity for puzzles carrying the theme, compared with the theme prevalence
(random baseline).

| Theme | n | Prevalence | First purity | First lift | Last purity | Last lift |
|---|---:|---:|---:|---:|---:|---:|
| backRankMate | 135 | 0.099 | 0.296 | +0.197 | 0.516 | +0.417 |
| discoveredAttack | 191 | 0.140 | 0.191 | +0.051 | 0.179 | +0.040 |
| doubleCheck | 24 | 0.018 | 0.019 | +0.002 | 0.022 | +0.005 |
| fork | 519 | 0.379 | 0.388 | +0.008 | 0.454 | +0.074 |
| pin | 221 | 0.162 | 0.179 | +0.017 | 0.180 | +0.018 |
| promotion | 96 | 0.070 | 0.328 | +0.258 | 0.406 | +0.336 |
| sacrifice | 293 | 0.214 | 0.248 | +0.034 | 0.211 | -0.003 |
| skewer | 81 | 0.059 | 0.084 | +0.025 | 0.128 | +0.068 |

| Summary metric | First | Last |
|---|---:|---:|
| Primary-theme silhouette, original | -0.015 | -0.014 |
| Primary-theme silhouette, UMAP | -0.248 | -0.239 |
| HDBSCAN theme AMI | 0.012 | 0.070 |
| HDBSCAN clusters | 2 | 6 |
| HDBSCAN noise fraction | 0.039 | 0.333 |

The last position is consistently more informative for `backRankMate`,
`promotion`, `fork`, and `skewer`, but almost all theme silhouettes remain near
zero. There is no clean global theme partition. The weak global result is not
caused by a single failed cluster: even the largest puzzle-last cluster is only
`39%` fork-dominant, and the most theme-pure small clusters (`backRankMate`
purity `0.613`, fork purity `0.607`) are small.

**Interpretation:** theme islands are weakly supported at best. The encoder
captures some local tactic-related signal, especially for distinctive motifs and
later puzzle positions, but tactical themes are not a primary axis of the global
embedding geometry.

## 5. Experiment 3: Temporal Trajectories

The trajectory sample contains `9,464` consecutive ply pairs.

| Metric | Value |
|---|---:|
| Adjacent-ply median cosine distance | 0.107 |
| Random same-game median cosine distance | 0.367 |
| Global random median cosine distance | 0.590 |
| Adjacent / random global ratio | 0.182 |
| Adjacent positive in top-15 neighbours | 0.967 |
| Game-aware kNN baseline | 0.480 |
| Game-aware lift | +0.487 |
| UMAP adjacent-step median | 0.036 |
| UMAP random-step median | 6.497 |
| UMAP step ratio | 0.0056 |

Adjacent positions are roughly `3.4x` closer than random same-game positions
and `5.5x` closer than random positions overall. In UMAP, same-game trajectories
form visibly smooth curves. The high raw hit rate is partly expected because of
the sampling design, but the game-aware baseline (`0.480`) still leaves a large
lift.

**Interpretation:** temporal continuity is strongly encoded. This is the
best-supported geometric property after phase and is consistent with the strong
temporal ranking results in the existing observability report.

## 6. Experiment 4: Dominant Surrogate Factors

The global test sample avoids same-game neighbour inflation. The ranking below
uses kNN lift over the query-class prevalence.

| Factor | Label | Purity | Baseline | Lift | Original silhouette | Trajectory lift |
|---|---|---:|---:|---:|---:|---:|
| phase | Game phase | 0.886 | 0.448 | +0.438 | 0.090 | +0.457 |
| bishop_pair | Bishop pair | 0.806 | 0.511 | +0.295 | 0.074 | +0.397 |
| open_files | Open files | 0.776 | 0.502 | +0.273 | 0.065 | +0.330 |
| king_attack | King attack | 0.581 | 0.357 | +0.224 | 0.031 | +0.288 |
| king_safety | King safety | 0.719 | 0.500 | +0.219 | 0.015 | +0.331 |
| isolated_pawns | Isolated pawns | 0.716 | 0.502 | +0.214 | 0.056 | +0.342 |
| pawn_islands | Pawn islands | 0.562 | 0.368 | +0.195 | 0.011 | +0.350 |
| mobility | Mobility | 0.553 | 0.373 | +0.180 | 0.004 | +0.178 |
| passed_pawns | Passed pawns | 0.778 | 0.600 | +0.178 | 0.065 | +0.299 |
| center_control | Centre control | 0.680 | 0.534 | +0.146 | -0.017 | +0.187 |

Phase is the dominant factor, followed by bishop-pair status, open files, king
attack pressure, king safety, and pawn structure. `material` is much weaker in
local neighbourhoods (lift `+0.050`, purity `0.762`, baseline `0.712`) than its
prominence in chess evaluation would suggest.

**Interpretation:** the embedding encodes a mixture of phase and structural
board properties. Material balance is not the main driver of local similarity.
The trajectory-sample lifts are larger because same-game positions share many
surrogates; the global sample is the cleaner comparison.

## 7. Experiment 5: Clusters and Failure Regions

### Global test clusters

HDBSCAN found **12 clusters**, noise fraction `0.0415`, weighted phase purity
`0.883`, minimum phase purity `0.840`. No cluster met the failure criterion
(`n >= 100` and phase purity more than `0.10` below the global `0.886`).

| Cluster | n | Top phase | Top-phase fraction | Phase purity | Mean ply |
|---:|---:|---|---:|---:|---:|
| 10 | 7,015 | opening | 0.430 | 0.840 | 52.7 |
| 4 | 1,198 | opening | 1.000 | 1.000 | 7.7 |
| 5 | 281 | opening | 1.000 | 1.000 | 13.2 |
| 8 | 273 | opening | 1.000 | 1.000 | 7.4 |
| 6 | 202 | opening | 1.000 | 1.000 | 17.2 |
| 9 | 108 | opening | 1.000 | 1.000 | 12.6 |

The largest cluster contains `70%` of all positions and mixes phases; it is a
broad transition/mixed region rather than a clean failure. The pure clusters are
mostly early-opening position types.

### Puzzle clusters

For puzzle-first, HDBSCAN produced 2 clusters with global theme purity `0.285`
and no failure clusters. The largest cluster (`n=1,195`) is fork-dominant but
only `35%` fork-pure; the smaller cluster (`n=119`) has theme purity `0.507`.

For puzzle-last, HDBSCAN produced 6 clusters with global theme purity `0.329`
and no failure clusters. Some small clusters are moderately theme-pure:

| Cluster | n | Top theme | Top-theme fraction | Theme purity | Mean rating |
|---:|---:|---|---:|---:|---:|
| 4 | 361 | fork | 0.393 | 0.383 | 1465 |
| 5 | 237 | fork | 0.388 | 0.414 | 1517 |
| 0 | 74 | fork | 0.554 | 0.607 | 1395 |
| 1 | 52 | backRankMate | 0.558 | 0.613 | 934 |

However, the noise fraction is `33.3%` and the global theme AMI is only
`0.070`. The theme weakness is global, not localised to one failed cluster.

### Combined test+puzzle map

The combined map does not separate test positions from puzzles:

| Metric | Value |
|---|---:|
| Source kNN purity | 0.810 |
| Source random baseline | 0.788 |
| Source lift | +0.022 |
| UMAP source silhouette | -0.086 |
| HDBSCAN clusters | 3 |
| Noise fraction | 0.004 |

One cluster contains `11,114` of `11,368` positions. The embedding space does
not place puzzle and game positions in separate islands.

**Interpretation:** there are no gross phase- or theme-failure regions under the
chosen thresholds. The main structural weakness is an undifferentiated mixed
region covering most positions, plus the absence of clean tactical-theme
clusters. For future stratified evaluation, the large mixed cluster is the most
important “hard” region to inspect.

## 8. Discussion

The experiments support a coherent picture:

- The embedding is primarily a **phase/game-progress manifold**.
- **Temporal continuity** is very strong locally.
- The model encodes **structural board factors** such as pawn configuration,
  king safety, and bishop-pair status.
- Tactical themes are only weakly and locally encoded; they do not define global
  islands.
- There are no obvious catastrophic clusters, but the space has a large mixed
  transition region and weak theme separation.

These results are consistent with the existing ranking observability: temporal
retrieval is comparatively easy, while motif ranking is close to chance for
several themes.

## 9. Limitations

- Single checkpoint, single UMAP seed, and a subsample of positions.
- UMAP preserves local neighbourhoods, not global distances or density; all
  quantitative conclusions therefore use the original embeddings.
- HDBSCAN cluster boundaries and the `0.10` failure threshold are analysis
  choices, not ground truth.
- Puzzle themes are sequence-level labels; using the first or last position is
  an approximation of when the motif is visible.
- Only eight themes could be analysed because the local puzzle shard yielded
  `1,368` puzzles carrying at least one of them.
- The trajectory sample intentionally over-represents same-game neighbours.
- No significance testing, cross-checkpoint comparison, or human motif
  adjudication was performed.

## 10. Reproduce

`umap-learn` is optional and is not a project dependency. Run it through `uv`:

```bash
uv run --with umap-learn python scripts/analysis/umap_embedding_structure.py \
  --checkpoint data/models/run_d512l8_p1p3_v1_20261026/last.pt \
  --test-dir data/hf-cache/datasets--patrickfrank1--chess-positions/snapshots/68131e5bd685023a95e0b8ea1ebe089a721deb99/test \
  --puzzles /tmp/obs_report_data/puzzles \
  --out-dir data/eval/similarity/umap_experiments_20261108 \
  --n-global 10000 \
  --n-trajectory 10000 \
  --trajectory-per-game-cap 20 \
  --n-puzzles 4000 \
  --puzzle-positions first,last
```

Add `--force` to recompute cached embeddings and UMAP coordinates. Without
`--force`, the script reuses `*_embeddings.npy` and `umap_*.npy` files in the
output directory; use a different output directory when changing the data
sample.

Key outputs:

```text
results.json                 # all metrics and cluster tables
global_*_clusters.csv        # cluster-stratified global test metrics
puzzle_first_clusters.csv
puzzle_last_clusters.csv
combined_global_clusters.csv
exp1_global_manifold.png
exp2_theme_first_last.png
exp3_temporal_trajectories.png
exp4_dominant_factors_global.png
exp5_clusters.png
```
