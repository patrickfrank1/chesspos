"""Explore chess-position embedding geometry with UMAP and HDBSCAN.

The script runs five descriptive experiments:

1. phase/manifold structure,
2. tactical-theme islands,
3. temporal trajectories,
4. dominant surrogate factors,
5. cluster-stratified failure regions.

All quantitative metrics are computed on the original embeddings. UMAP is used
only for visualization and cluster stratification.

Run with the optional dependency:

    uv run --with umap-learn python scripts/analysis/umap_embedding_structure.py \
        --checkpoint data/models/run_d512l8_p1p3_v1_20261026/last.pt \
        --test-dir data/hf-cache/.../test \
        --puzzles /tmp/obs_report_data/puzzles \
        --out-dir data/eval/similarity/umap_experiments
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.cluster import HDBSCAN
from sklearn.metrics import adjusted_mutual_info_score, silhouette_score
from sklearn.neighbors import NearestNeighbors

from src.evaluation.observability.embeddings import embed_samples
from src.evaluation.observability.loading import load_checkpoint
from src.evaluation.observability.samples import (
    PositionSample,
    load_puzzle_samples,
    load_test_samples,
)
from src.evaluation.observability.surrogates import (
    SURROGATE_TASKS,
    classify_surrogates,
)

DEFAULT_THEMES = (
    "fork",
    "pin",
    "skewer",
    "sacrifice",
    "discoveredAttack",
    "doubleCheck",
    "backRankMate",
    "promotion",
)


@dataclass(frozen=True)
class AnalysisConfig:
    checkpoint: str
    test_dir: str
    puzzles: str
    out_dir: Path
    themes: tuple[str, ...]
    n_global: int = 10_000
    n_trajectory: int = 10_000
    trajectory_per_game_cap: int = 20
    n_puzzles: int = 4_000
    puzzle_positions: tuple[str, ...] = ("first", "last")
    k: int = 15
    umap_neighbors: int = 30
    umap_min_dist: float = 0.05
    umap_n_jobs: int = -1
    device: str = "cpu"
    batch_size: int = 128
    seed: int = 17
    force: bool = False


def parse_args(argv: Sequence[str] | None = None) -> AnalysisConfig:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--test-dir", required=True)
    parser.add_argument("--puzzles", required=True)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--themes", default=",".join(DEFAULT_THEMES))
    parser.add_argument("--n-global", type=int, default=10_000)
    parser.add_argument("--n-trajectory", type=int, default=10_000)
    parser.add_argument("--trajectory-per-game-cap", type=int, default=20)
    parser.add_argument("--n-puzzles", type=int, default=4_000)
    parser.add_argument(
        "--puzzle-positions",
        default="first,last",
        help="comma-separated subset of first,last",
    )
    parser.add_argument("--k", type=int, default=15)
    parser.add_argument("--umap-neighbors", type=int, default=30)
    parser.add_argument("--umap-min-dist", type=float, default=0.05)
    parser.add_argument("--umap-n-jobs", type=int, default=-1)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)
    positions = tuple(
        value.strip() for value in args.puzzle_positions.split(",") if value.strip()
    )
    unknown = set(positions) - {"first", "last"}
    if unknown:
        raise ValueError(f"Unknown puzzle positions: {sorted(unknown)}")
    return AnalysisConfig(
        checkpoint=args.checkpoint,
        test_dir=args.test_dir,
        puzzles=args.puzzles,
        out_dir=args.out_dir,
        themes=tuple(
            value.strip() for value in args.themes.split(",") if value.strip()
        ),
        n_global=args.n_global,
        n_trajectory=args.n_trajectory,
        trajectory_per_game_cap=args.trajectory_per_game_cap,
        n_puzzles=args.n_puzzles,
        puzzle_positions=positions,
        k=args.k,
        umap_neighbors=args.umap_neighbors,
        umap_min_dist=args.umap_min_dist,
        umap_n_jobs=args.umap_n_jobs,
        device=args.device,
        batch_size=args.batch_size,
        seed=args.seed,
        force=args.force,
    )


def jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return jsonable(value.tolist())
    if isinstance(value, np.generic):
        return jsonable(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def sample_frame(samples: Sequence[PositionSample]) -> pd.DataFrame:
    rows = []
    for sample in samples:
        classes = classify_surrogates(sample.surrogates)
        row = {
            "sample_id": sample.sample_id,
            "source": sample.source,
            "fen": sample.fen,
            "game_id": sample.game_id,
            "puzzle_id": sample.puzzle_id,
            "ply": sample.ply,
            "rating": sample.rating,
            "themes": "|".join(sample.themes),
        }
        for key, value in sample.surrogates.items():
            row[f"sur_{key}"] = value
        for task in SURROGATE_TASKS:
            row[f"class_{task.name}"] = classes[task.name]
        rows.append(row)
    return pd.DataFrame(rows)


def load_frame(path: Path, themes: Sequence[str]) -> pd.DataFrame:
    frame = pd.read_parquet(path)
    frame["themes"] = frame["themes"].fillna("")
    frame["theme_list"] = frame["themes"].map(
        lambda value: tuple(t for t in str(value).split("|") if t)
    )
    frame["primary_theme"] = frame["theme_list"].map(
        lambda values: next(
            (theme for theme in themes if theme in values),
            values[0] if values else "other",
        )
    )
    return frame


def save_sample_artifacts(
    config: AnalysisConfig, name: str, samples: Sequence[PositionSample]
) -> tuple[pd.DataFrame, np.ndarray]:
    samples_path = config.out_dir / f"{name}_samples.parquet"
    embeddings_path = config.out_dir / f"{name}_embeddings.npy"
    if not config.force and samples_path.exists() and embeddings_path.exists():
        return load_frame(samples_path, config.themes), np.load(embeddings_path)
    frame = sample_frame(samples)
    model, _ = load_checkpoint(config.checkpoint, config.device)
    embeddings = embed_samples(
        model,
        list(samples),
        torch.device(config.device),
        batch_size=config.batch_size,
        bf16=False,
    )
    frame.to_parquet(samples_path, index=False)
    np.save(embeddings_path, embeddings)
    return load_frame(samples_path, config.themes), embeddings


def load_test_dataset(
    config: AnalysisConfig,
    name: str,
    n: int,
    *,
    per_game_cap: int,
) -> tuple[pd.DataFrame, np.ndarray]:
    samples_path = config.out_dir / f"{name}_samples.parquet"
    embeddings_path = config.out_dir / f"{name}_embeddings.npy"
    if not config.force and samples_path.exists() and embeddings_path.exists():
        return load_frame(samples_path, config.themes), np.load(embeddings_path)
    samples = load_test_samples(
        config.test_dir,
        n,
        per_game_cap=per_game_cap,
        seed=config.seed,
    )
    return save_sample_artifacts(config, name, samples)


def load_puzzle_dataset(
    config: AnalysisConfig,
    position: str,
) -> tuple[pd.DataFrame, np.ndarray]:
    name = f"puzzle_{position}"
    samples_path = config.out_dir / f"{name}_samples.parquet"
    embeddings_path = config.out_dir / f"{name}_embeddings.npy"
    if not config.force and samples_path.exists() and embeddings_path.exists():
        return load_frame(samples_path, config.themes), np.load(embeddings_path)
    samples = load_puzzle_samples(
        config.puzzles,
        config.n_puzzles,
        themes=config.themes,
        position=position,
        seed=config.seed,
    )
    return save_sample_artifacts(config, name, samples)


def fit_umap(config: AnalysisConfig, name: str, values: np.ndarray) -> np.ndarray:
    path = config.out_dir / f"umap_{name}.npy"
    if not config.force and path.exists():
        return np.load(path)
    try:
        import umap
    except ImportError as exc:
        raise RuntimeError(
            "umap-learn is required; run with `uv run --with umap-learn ...`"
        ) from exc
    reducer = umap.UMAP(
        n_neighbors=config.umap_neighbors,
        min_dist=config.umap_min_dist,
        n_components=2,
        metric="cosine",
        random_state=config.seed,
        n_jobs=config.umap_n_jobs,
        verbose=True,
    )
    coords = reducer.fit_transform(values)
    np.save(path, coords)
    return coords


def neighbors(
    values: np.ndarray, k: int, *, metric: str = "cosine"
) -> tuple[np.ndarray, np.ndarray]:
    model = NearestNeighbors(
        n_neighbors=k + 1,
        metric=metric,
        algorithm="brute",
        n_jobs=-1,
    )
    model.fit(values)
    distances, indices = model.kneighbors(values)
    return distances[:, 1:], indices[:, 1:]


def label_purity(indices: np.ndarray, labels: np.ndarray) -> dict[str, Any]:
    labels = np.asarray(labels)
    values = []
    per_class: dict[str, list[float]] = {}
    for index, row in enumerate(indices):
        matches = labels[row] == labels[index]
        value = float(np.mean(matches))
        values.append(value)
        per_class.setdefault(str(labels[index]), []).append(value)
    overall = float(np.mean(values))
    prevalence = {
        str(label): float(np.mean(labels == label)) for label in np.unique(labels)
    }
    baseline = float(np.mean([prevalence[str(label)] for label in labels]))
    return {
        "purity": overall,
        "random_baseline": baseline,
        "lift": overall - baseline,
        "per_class": {
            key: float(np.mean(items)) for key, items in sorted(per_class.items())
        },
        "class_counts": {
            str(key): int(value) for key, value in sorted(Counter(labels).items())
        },
    }


def binary_theme_purity(
    indices: np.ndarray, positive: np.ndarray
) -> dict[str, float | int]:
    positive = np.asarray(positive, dtype=bool)
    mask = np.flatnonzero(positive)
    if len(mask) == 0:
        return {"n": 0, "purity": 0.0, "prevalence": 0.0, "lift": 0.0}
    values = [float(np.mean(positive[indices[index]])) for index in mask]
    prevalence = float(np.mean(positive))
    purity = float(np.mean(values))
    return {
        "n": int(len(mask)),
        "purity": purity,
        "prevalence": prevalence,
        "lift": purity - prevalence,
    }


def same_game_fraction(indices: np.ndarray, game_ids: np.ndarray) -> dict[str, float]:
    game_ids = np.asarray(game_ids)
    values = [
        float(np.mean(game_ids[row] == game_ids[index]))
        for index, row in enumerate(indices)
    ]
    overall = float(np.mean(values))
    counts = Counter(game_ids)
    expected = float(
        np.mean([(counts[game] - 1) / max(1, len(game_ids) - 1) for game in game_ids])
    )
    return {
        "same_game_fraction": overall,
        "random_baseline": expected,
        "lift": overall - expected,
    }


def _game_rows(frame: pd.DataFrame) -> dict[str, list[tuple[int, int]]]:
    by_game: dict[str, list[tuple[int, int]]] = {}
    for row, (game, ply) in enumerate(zip(frame["game_id"], frame["ply"])):
        if pd.isna(game) or pd.isna(ply):
            continue
        by_game.setdefault(str(game), []).append((int(ply), row))
    for rows in by_game.values():
        rows.sort()
    return by_game


def adjacent_stats(
    values: np.ndarray,
    frame: pd.DataFrame,
    indices: np.ndarray,
    *,
    seed: int,
    k: int,
) -> dict[str, float | int]:
    by_game = _game_rows(frame)
    adjacent_pairs = []
    for rows in by_game.values():
        for (ply_a, index_a), (ply_b, index_b) in zip(rows, rows[1:]):
            if ply_b - ply_a == 1:
                adjacent_pairs.append((index_a, index_b))
    if not adjacent_pairs:
        return {"n_pairs": 0}
    adjacent_distances = np.array(
        [1.0 - float(values[a] @ values[b]) for a, b in adjacent_pairs]
    )
    adjacent_hits = np.array(
        [float(b in set(indices[a].tolist())) for a, b in adjacent_pairs]
    )
    rng = np.random.default_rng(seed)
    random_pairs = rng.integers(
        0,
        len(values),
        size=(min(50_000, len(adjacent_pairs) * 20), 2),
    )
    random_distances = 1.0 - np.einsum(
        "ij,ij->i", values[random_pairs[:, 0]], values[random_pairs[:, 1]]
    )
    same_game_distances = []
    for rows in by_game.values():
        if len(rows) < 2:
            continue
        for _ in range(min(3, len(rows))):
            a, b = rng.choice(len(rows), size=2, replace=False)
            same_game_distances.append(
                1.0 - float(values[rows[a][1]] @ values[rows[b][1]])
            )
    same_game_distances = np.asarray(same_game_distances)
    hit_rate = float(np.mean(adjacent_hits))
    return {
        "n_pairs": int(len(adjacent_pairs)),
        "adjacent_mean": float(np.mean(adjacent_distances)),
        "adjacent_median": float(np.median(adjacent_distances)),
        "adjacent_p90": float(np.percentile(adjacent_distances, 90)),
        "random_mean": float(np.mean(random_distances)),
        "random_median": float(np.median(random_distances)),
        "same_game_mean": float(np.mean(same_game_distances)),
        "same_game_median": float(np.median(same_game_distances)),
        "adjacent_vs_random_ratio": float(
            np.median(adjacent_distances) / np.median(random_distances)
        ),
        "knn_adjacent_hit_rate": hit_rate,
        "knn_random_baseline": float(k / max(1, len(values) - 1)),
        "knn_lift": float(hit_rate - k / max(1, len(values) - 1)),
    }


def umap_step_stats(
    coords: np.ndarray, frame: pd.DataFrame, *, seed: int
) -> dict[str, float]:
    by_game = _game_rows(frame)
    adjacent = []
    for rows in by_game.values():
        for (ply_a, index_a), (ply_b, index_b) in zip(rows, rows[1:]):
            if ply_b - ply_a == 1:
                adjacent.append(
                    float(np.linalg.norm(coords[index_a] - coords[index_b]))
                )
    rng = np.random.default_rng(seed)
    random = np.linalg.norm(
        coords[rng.integers(0, len(coords), 50_000)]
        - coords[rng.integers(0, len(coords), 50_000)],
        axis=1,
    )
    adjacent_values = np.asarray(adjacent)
    return {
        "adjacent_mean": float(np.mean(adjacent_values)),
        "adjacent_median": float(np.median(adjacent_values)),
        "random_mean": float(np.mean(random)),
        "random_median": float(np.median(random)),
        "ratio": float(np.median(adjacent_values) / np.median(random)),
    }


def silhouette(
    values: np.ndarray,
    labels: np.ndarray,
    *,
    seed: int,
    sample_size: int = 3_000,
) -> float | None:
    labels = np.asarray(labels)
    if len(np.unique(labels)) < 2:
        return None
    size = min(sample_size, len(values))
    return float(
        silhouette_score(
            values,
            labels,
            metric="cosine" if values.shape[1] > 2 else "euclidean",
            sample_size=size,
            random_state=seed,
        )
    )


def scatter_panel(
    ax: plt.Axes,
    coords: np.ndarray,
    labels: Sequence[Any],
    title: str,
    *,
    cmap: str = "tab10",
    s: float = 1.5,
    alpha: float = 0.5,
) -> np.ndarray:
    codes, uniques = pd.factorize(pd.Series(labels).astype(str))
    ax.scatter(
        coords[:, 0],
        coords[:, 1],
        c=codes,
        cmap=cmap,
        s=s,
        alpha=alpha,
        linewidths=0,
        rasterized=True,
    )
    ax.set_title(title, fontsize=10)
    ax.set_xticks([])
    ax.set_yticks([])
    return np.asarray(uniques)


def cluster_rows(
    frame: pd.DataFrame, clusters: np.ndarray, *, source: str
) -> pd.DataFrame:
    rows = []
    for cluster in sorted(set(clusters)):
        if cluster == -1:
            continue
        mask = clusters == cluster
        part = frame.loc[mask]
        if part.empty:
            continue
        row = {
            "cluster": int(cluster),
            "n": int(mask.sum()),
            "source": source,
            "top_game": part["game_id"].dropna().mode().iloc[0]
            if part["game_id"].notna().any()
            else None,
            "top_game_fraction": float(
                part["game_id"].value_counts(normalize=True).iloc[0]
            )
            if part["game_id"].notna().any()
            else None,
            "top_phase": part["class_phase"].mode().iloc[0]
            if "class_phase" in part and not part.empty
            else None,
            "top_theme": part["primary_theme"].mode().iloc[0]
            if "primary_theme" in part and part["primary_theme"].notna().any()
            else None,
            "mean_ply": float(part["ply"].dropna().mean())
            if part["ply"].notna().any()
            else None,
            "mean_rating": float(part["rating"].dropna().mean())
            if part["rating"].notna().any()
            else None,
        }
        rows.append(row)
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values("n", ascending=False)


def cluster_phase_purity(
    frame: pd.DataFrame, indices: np.ndarray, clusters: np.ndarray
) -> pd.DataFrame:
    rows = cluster_rows(frame, clusters, source="test")
    if rows.empty:
        return rows
    phase = frame["class_phase"].to_numpy()
    purities = []
    top_phase_fractions = []
    for cluster in rows["cluster"]:
        indexes = np.flatnonzero(clusters == cluster)
        values = [
            float(np.mean(phase[indices[index]] == phase[index])) for index in indexes
        ]
        purities.append(float(np.mean(values)) if values else 0.0)
        counts = pd.Series(phase[indexes]).value_counts(normalize=True)
        top_phase_fractions.append(float(counts.iloc[0]) if len(counts) else 0.0)
    rows["phase_purity"] = purities
    rows["top_phase_fraction"] = top_phase_fractions
    rows["mean_ply"] = [
        float(frame.loc[clusters == cluster, "ply"].mean())
        for cluster in rows["cluster"]
    ]
    return rows


def cluster_theme_purity(
    frame: pd.DataFrame,
    indices: np.ndarray,
    clusters: np.ndarray,
    *,
    source: str,
) -> pd.DataFrame:
    rows = cluster_rows(frame, clusters, source=source)
    if rows.empty:
        return rows
    purities = []
    top_theme_fractions = []
    for cluster in rows["cluster"]:
        indexes = np.flatnonzero(clusters == cluster)
        top_theme = frame.loc[indexes, "primary_theme"].mode().iloc[0]
        positives = (
            frame["theme_list"].map(lambda values: top_theme in values).to_numpy()
        )
        values = [float(np.mean(positives[indices[index]])) for index in indexes]
        purities.append(float(np.mean(values)) if values else 0.0)
        counts = frame.loc[indexes, "primary_theme"].value_counts(normalize=True)
        top_theme_fractions.append(float(counts.iloc[0]) if len(counts) else 0.0)
    rows["theme_purity"] = purities
    rows["top_theme_fraction"] = top_theme_fractions
    return rows


def global_theme_purity(frame: pd.DataFrame, indices: np.ndarray) -> float:
    values = []
    for index in range(len(frame)):
        theme = frame.loc[index, "primary_theme"]
        positive = frame["theme_list"].map(lambda items: theme in items).to_numpy()
        values.append(float(np.mean(positive[indices[index]])))
    return float(np.mean(values)) if values else 0.0


def theme_table(
    config: AnalysisConfig,
    frame: pd.DataFrame,
    embeddings: np.ndarray,
    umap_coords: np.ndarray,
    indices: np.ndarray,
    umap_indices: np.ndarray,
) -> dict[str, Any]:
    table = {}
    for theme in config.themes:
        positive = frame["theme_list"].map(lambda values: theme in values).to_numpy()
        table[theme] = {
            "original": binary_theme_purity(indices, positive),
            "umap": binary_theme_purity(umap_indices, positive),
            "silhouette_original": silhouette(
                embeddings, positive.astype(int), seed=config.seed
            ),
            "silhouette_umap": silhouette(
                umap_coords, positive.astype(int), seed=config.seed
            ),
        }
    return table


def plot_global_manifold(
    config: AnalysisConfig,
    frame: pd.DataFrame,
    coords: np.ndarray,
) -> Path:
    phase = frame["class_phase"].to_numpy()
    fig, axes = plt.subplots(2, 2, figsize=(13, 10), constrained_layout=True)
    scatter_panel(axes[0, 0], coords, phase, "Global test: phase")
    axes[0, 0].legend(
        handles=[
            plt.Line2D(
                [],
                [],
                marker="o",
                linestyle="",
                color=plt.cm.tab10(index),
                label=label,
                markersize=6,
            )
            for index, label in enumerate(sorted(set(phase)))
        ],
        fontsize=8,
        loc="best",
    )
    scatter_panel(axes[0, 1], coords, frame["ply"], "Global test: ply", cmap="viridis")
    scatter_panel(
        axes[1, 0],
        coords,
        frame["sur_material"],
        "Global test: material",
        cmap="coolwarm",
    )
    scatter_panel(
        axes[1, 1],
        coords,
        frame["sur_piece_count"],
        "Global test: piece count",
        cmap="plasma",
    )
    fig.suptitle("Experiment 1: global test-position manifold (one position per game)")
    path = config.out_dir / "exp1_global_manifold.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def plot_theme_first_last(
    config: AnalysisConfig,
    first_frame: pd.DataFrame,
    first_coords: np.ndarray,
    last_frame: pd.DataFrame | None,
    last_coords: np.ndarray | None,
) -> Path:
    columns = 2 if last_frame is not None else 1
    fig, axes = plt.subplots(
        1, columns, figsize=(15 if columns == 2 else 8, 6.5), constrained_layout=True
    )
    first_axis = axes[0] if columns == 2 else axes
    uniques = scatter_panel(
        first_axis,
        first_coords,
        first_frame["primary_theme"],
        "Puzzle first position: primary theme",
    )
    first_axis.legend(
        handles=[
            plt.Line2D(
                [],
                [],
                marker="o",
                linestyle="",
                color=plt.cm.tab10(index),
                label=label,
                markersize=6,
            )
            for index, label in enumerate(uniques)
        ],
        fontsize=8,
        loc="best",
    )
    if last_frame is not None and last_coords is not None:
        scatter_panel(
            axes[1],
            last_coords,
            last_frame["primary_theme"],
            "Puzzle last position: primary theme",
        )
    fig.suptitle("Experiment 2: tactical-theme islands (first vs last position)")
    path = config.out_dir / "exp2_theme_first_last.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def plot_temporal(
    config: AnalysisConfig,
    frame: pd.DataFrame,
    embeddings: np.ndarray,
    coords: np.ndarray,
) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(15, 6.5), constrained_layout=True)
    axes[0].scatter(
        coords[:, 0],
        coords[:, 1],
        s=1,
        alpha=0.15,
        c="#bbbbbb",
        rasterized=True,
    )
    games = [game for game, part in frame.groupby("game_id") if len(part) >= 10]
    rng = np.random.default_rng(config.seed)
    for game in (
        list(rng.choice(games, size=min(12, len(games)), replace=False))
        if games
        else []
    ):
        part = frame[frame["game_id"] == game].sort_values("ply")
        indices = part.index.to_numpy()
        axes[0].plot(
            coords[indices, 0],
            coords[indices, 1],
            "-o",
            linewidth=0.8,
            markersize=1.5,
            alpha=0.8,
        )
    axes[0].set_title("Experiment 3: 12 same-game trajectories")
    axes[0].set_xticks([])
    axes[0].set_yticks([])
    adjacent = []
    for rows in _game_rows(frame).values():
        for (ply_a, index_a), (ply_b, index_b) in zip(rows, rows[1:]):
            if ply_b - ply_a == 1:
                adjacent.append(1.0 - float(embeddings[index_a] @ embeddings[index_b]))
    axes[1].hist(
        adjacent,
        bins=50,
        alpha=0.7,
        label="adjacent, original cosine",
        density=True,
    )
    random_pairs = rng.integers(0, len(embeddings), size=(20_000, 2))
    axes[1].hist(
        1.0
        - np.einsum(
            "ij,ij->i",
            embeddings[random_pairs[:, 0]],
            embeddings[random_pairs[:, 1]],
        ),
        bins=50,
        alpha=0.45,
        label="random, original cosine",
        density=True,
    )
    axes[1].set_title("Adjacent vs random step distances")
    axes[1].legend(fontsize=8)
    path = config.out_dir / "exp3_temporal_trajectories.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def plot_factors(
    config: AnalysisConfig,
    frame: pd.DataFrame,
    coords: np.ndarray,
    ranked: Sequence[tuple[str, dict[str, Any]]],
) -> Path:
    fig, axes = plt.subplots(2, 3, figsize=(16, 10), constrained_layout=True)
    for ax, (name, row) in zip(axes.flat, ranked[:6]):
        scatter_panel(
            ax,
            coords,
            frame[f"class_{name}"].to_numpy(),
            f"{row['label']}\npurity={row['knn_global']['purity']:.2f} "
            f"lift={row['knn_global']['lift']:.2f}",
        )
    fig.suptitle(
        "Experiment 4: strongest surrogate factors in global test-position space"
    )
    path = config.out_dir / "exp4_dominant_factors_global.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def plot_clusters(
    config: AnalysisConfig,
    global_coords: np.ndarray,
    global_clusters: np.ndarray,
    puzzle_coords: np.ndarray | None,
    puzzle_clusters: np.ndarray | None,
    combined_coords: np.ndarray,
    source_labels: np.ndarray,
) -> Path:
    panels = 3 if puzzle_coords is not None and puzzle_clusters is not None else 2
    fig, axes = plt.subplots(
        1, panels, figsize=(6 * panels, 6), constrained_layout=True
    )
    scatter_panel(
        axes[0],
        global_coords,
        global_clusters.astype(str),
        "Global test HDBSCAN clusters",
    )
    if panels == 3 and puzzle_coords is not None and puzzle_clusters is not None:
        scatter_panel(
            axes[1],
            puzzle_coords,
            puzzle_clusters.astype(str),
            "Puzzle-last HDBSCAN clusters",
        )
        scatter_panel(
            axes[2],
            combined_coords,
            source_labels,
            "Combined map: source",
        )
    else:
        scatter_panel(
            axes[1],
            combined_coords,
            source_labels,
            "Combined map: source",
        )
    fig.suptitle("Experiment 5: cluster structure and failure regions")
    path = config.out_dir / "exp5_clusters.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def run(config: AnalysisConfig) -> dict[str, Any]:
    config.out_dir.mkdir(parents=True, exist_ok=True)
    global_frame, global_emb = load_test_dataset(
        config,
        "global",
        config.n_global,
        per_game_cap=1,
    )
    trajectory_frame, trajectory_emb = load_test_dataset(
        config,
        "trajectory",
        config.n_trajectory,
        per_game_cap=config.trajectory_per_game_cap,
    )
    puzzle_frames: dict[str, pd.DataFrame] = {}
    puzzle_embeddings: dict[str, np.ndarray] = {}
    for position in config.puzzle_positions:
        frame, embeddings = load_puzzle_dataset(config, position)
        puzzle_frames[position] = frame
        puzzle_embeddings[position] = embeddings
    first_frame = puzzle_frames.get(config.puzzle_positions[0])
    first_emb = puzzle_embeddings[config.puzzle_positions[0]]

    umap_global = fit_umap(config, "global", global_emb)
    umap_trajectory = fit_umap(config, "trajectory", trajectory_emb)
    umap_first = fit_umap(config, "puzzle_first", first_emb)
    umap_last = (
        fit_umap(config, "puzzle_last", puzzle_embeddings["last"])
        if "last" in puzzle_frames
        else None
    )
    combined_emb = np.vstack([global_emb, first_emb])
    combined_frame = pd.concat([global_frame, first_frame], ignore_index=True)
    umap_combined = fit_umap(config, "combined_global", combined_emb)

    _, global_neighbors = neighbors(global_emb, config.k)
    _, global_umap_neighbors = neighbors(umap_global, config.k, metric="euclidean")
    _, trajectory_neighbors = neighbors(trajectory_emb, config.k)
    _, first_neighbors = neighbors(first_emb, config.k)
    _, first_umap_neighbors = neighbors(umap_first, config.k, metric="euclidean")
    _, combined_neighbors = neighbors(combined_emb, config.k)
    _, combined_umap_neighbors = neighbors(umap_combined, config.k, metric="euclidean")
    last_neighbors = last_umap_neighbors = None
    if "last" in puzzle_frames and umap_last is not None:
        _, last_neighbors = neighbors(puzzle_embeddings["last"], config.k)
        _, last_umap_neighbors = neighbors(umap_last, config.k, metric="euclidean")

    results: dict[str, Any] = {
        "config": {
            "checkpoint": config.checkpoint,
            "test_dir": config.test_dir,
            "puzzle_dir": config.puzzles,
            "seed": config.seed,
            "k": config.k,
            "global": {
                "n": int(len(global_frame)),
                "games": int(global_frame["game_id"].nunique()),
            },
            "trajectory": {
                "n": int(len(trajectory_frame)),
                "games": int(trajectory_frame["game_id"].nunique()),
                "per_game_cap": config.trajectory_per_game_cap,
            },
            "puzzles": {
                position: int(len(frame)) for position, frame in puzzle_frames.items()
            },
            "themes": list(config.themes),
            "umap": {
                "n_neighbors": config.umap_neighbors,
                "min_dist": config.umap_min_dist,
                "metric": "cosine",
            },
        }
    }

    phase = global_frame["class_phase"].to_numpy()
    clusters_global = HDBSCAN(
        min_cluster_size=50, min_samples=10, copy=True
    ).fit_predict(umap_global)
    global_clusters_table = cluster_phase_purity(
        global_frame, global_neighbors, clusters_global
    )
    same_game = same_game_fraction(
        trajectory_neighbors, trajectory_frame["game_id"].to_numpy()
    )
    results["exp1_phase_manifold"] = {
        "phase_knn_global": label_purity(global_neighbors, phase),
        "phase_knn_umap": label_purity(global_umap_neighbors, phase),
        "phase_silhouette_original": silhouette(global_emb, phase, seed=config.seed),
        "phase_silhouette_umap": silhouette(umap_global, phase, seed=config.seed),
        "umap_corr_ply": {
            "dim0": float(
                pd.Series(umap_global[:, 0]).corr(
                    pd.Series(global_frame["ply"]), method="spearman"
                )
            ),
            "dim1": float(
                pd.Series(umap_global[:, 1]).corr(
                    pd.Series(global_frame["ply"]), method="spearman"
                )
            ),
        },
        "phase_by_ply": {
            label: {
                "n": int((phase == label).sum()),
                "mean_ply": float(global_frame.loc[phase == label, "ply"].mean()),
                "median_ply": float(global_frame.loc[phase == label, "ply"].median()),
            }
            for label in sorted(set(phase))
        },
        "phase_by_material_abs": {
            label: float(global_frame.loc[phase == label, "sur_material"].abs().mean())
            for label in sorted(set(phase))
        },
        "hdbscan": {
            "n_clusters": int(
                len([cluster for cluster in set(clusters_global) if cluster != -1])
            ),
            "noise_fraction": float(np.mean(clusters_global == -1)),
            "weighted_phase_purity": float(
                np.average(
                    global_clusters_table["phase_purity"],
                    weights=global_clusters_table["n"],
                )
            )
            if not global_clusters_table.empty
            else None,
            "min_phase_purity": float(global_clusters_table["phase_purity"].min())
            if not global_clusters_table.empty
            else None,
        },
        "trajectory_same_game_knn": same_game,
    }

    first_theme_table = theme_table(
        config,
        first_frame,
        first_emb,
        umap_first,
        first_neighbors,
        first_umap_neighbors,
    )
    first_primary = first_frame["primary_theme"].to_numpy()
    clusters_first = HDBSCAN(
        min_cluster_size=50, min_samples=10, copy=True
    ).fit_predict(umap_first)
    results["exp2_theme_islands"] = {
        "first": {
            "themes": first_theme_table,
            "primary_silhouette_original": silhouette(
                first_emb, first_primary, seed=config.seed
            ),
            "primary_silhouette_umap": silhouette(
                umap_first, first_primary, seed=config.seed
            ),
            "hdbscan_theme_ami": float(
                adjusted_mutual_info_score(first_primary, clusters_first)
            ),
            "hdbscan_n_clusters": int(
                len([cluster for cluster in set(clusters_first) if cluster != -1])
            ),
            "hdbscan_noise_fraction": float(np.mean(clusters_first == -1)),
        }
    }
    clusters_last = None
    last_theme_table = None
    if (
        "last" in puzzle_frames
        and umap_last is not None
        and last_neighbors is not None
        and last_umap_neighbors is not None
    ):
        last_frame = puzzle_frames["last"]
        last_emb = puzzle_embeddings["last"]
        last_theme_table = theme_table(
            config,
            last_frame,
            last_emb,
            umap_last,
            last_neighbors,
            last_umap_neighbors,
        )
        last_primary = last_frame["primary_theme"].to_numpy()
        clusters_last = HDBSCAN(
            min_cluster_size=50, min_samples=10, copy=True
        ).fit_predict(umap_last)
        results["exp2_theme_islands"]["last"] = {
            "themes": last_theme_table,
            "primary_silhouette_original": silhouette(
                last_emb, last_primary, seed=config.seed
            ),
            "primary_silhouette_umap": silhouette(
                umap_last, last_primary, seed=config.seed
            ),
            "hdbscan_theme_ami": float(
                adjusted_mutual_info_score(last_primary, clusters_last)
            ),
            "hdbscan_n_clusters": int(
                len([cluster for cluster in set(clusters_last) if cluster != -1])
            ),
            "hdbscan_noise_fraction": float(np.mean(clusters_last == -1)),
        }

    original_stats = adjacent_stats(
        trajectory_emb,
        trajectory_frame,
        trajectory_neighbors,
        seed=config.seed,
        k=config.k,
    )
    original_stats["game_aware_knn_baseline"] = (
        same_game["same_game_fraction"] * float(config.k / 19.0)
        + (1.0 - same_game["same_game_fraction"])
        * original_stats["knn_random_baseline"]
    )
    original_stats["game_aware_knn_lift"] = (
        original_stats["knn_adjacent_hit_rate"]
        - original_stats["game_aware_knn_baseline"]
    )
    results["exp3_temporal"] = {
        "original": original_stats,
        "umap": umap_step_stats(umap_trajectory, trajectory_frame, seed=config.seed),
        "note": (
            "The trajectory sample has many contiguous positions per game, "
            "so same-game neighbours are intentionally over-represented."
        ),
    }

    global_factors: dict[str, Any] = {}
    trajectory_factors: dict[str, Any] = {}
    for task in SURROGATE_TASKS:
        global_labels = global_frame[f"class_{task.name}"].to_numpy()
        trajectory_labels = trajectory_frame[f"class_{task.name}"].to_numpy()
        global_factors[task.name] = {
            "label": task.label,
            "knn_global": label_purity(global_neighbors, global_labels),
            "knn_umap": label_purity(global_umap_neighbors, global_labels),
            "silhouette_original": silhouette(
                global_emb,
                global_labels,
                seed=config.seed,
                sample_size=2_000,
            ),
            "silhouette_umap": silhouette(
                umap_global,
                global_labels,
                seed=config.seed,
                sample_size=2_000,
            ),
        }
        trajectory_factors[task.name] = {
            "label": task.label,
            "knn_trajectory": label_purity(trajectory_neighbors, trajectory_labels),
        }
    ranked = sorted(
        global_factors.items(),
        key=lambda item: item[1]["knn_global"]["lift"],
        reverse=True,
    )
    results["exp4_dominant_factors"] = {
        "global": global_factors,
        "trajectory": trajectory_factors,
        "ranked_by_global_knn_lift": [name for name, _ in ranked],
    }

    global_phase_purity = results["exp1_phase_manifold"]["phase_knn_global"]["purity"]
    if not global_clusters_table.empty:
        global_clusters_table["global_phase_purity"] = global_phase_purity
        global_clusters_table["phase_gap"] = (
            global_clusters_table["phase_purity"] - global_phase_purity
        )
        global_clusters_table["failure"] = (global_clusters_table["n"] >= 100) & (
            global_clusters_table["phase_gap"] < -0.10
        )
    first_cluster_table = cluster_theme_purity(
        first_frame, first_neighbors, clusters_first, source="puzzle_first"
    )
    if not first_cluster_table.empty:
        first_global_purity = global_theme_purity(first_frame, first_neighbors)
        first_cluster_table["global_theme_purity"] = first_global_purity
        first_cluster_table["theme_gap"] = (
            first_cluster_table["theme_purity"] - first_global_purity
        )
        first_cluster_table["failure"] = (first_cluster_table["n"] >= 100) & (
            first_cluster_table["theme_gap"] < -0.10
        )
    last_cluster_table = None
    if (
        clusters_last is not None
        and last_neighbors is not None
        and "last" in puzzle_frames
    ):
        last_frame = puzzle_frames["last"]
        last_cluster_table = cluster_theme_purity(
            last_frame, last_neighbors, clusters_last, source="puzzle_last"
        )
        if not last_cluster_table.empty:
            last_global_purity = global_theme_purity(last_frame, last_neighbors)
            last_cluster_table["global_theme_purity"] = last_global_purity
            last_cluster_table["theme_gap"] = (
                last_cluster_table["theme_purity"] - last_global_purity
            )
            last_cluster_table["failure"] = (last_cluster_table["n"] >= 100) & (
                last_cluster_table["theme_gap"] < -0.10
            )
    source_labels = combined_frame["source"].to_numpy()
    clusters_combined = HDBSCAN(
        min_cluster_size=100, min_samples=10, copy=True
    ).fit_predict(umap_combined)
    combined_source_purity = label_purity(combined_umap_neighbors, source_labels)
    combined_source_silhouette = silhouette(
        umap_combined, source_labels, seed=config.seed
    )
    combined_cluster_table = cluster_rows(
        combined_frame, clusters_combined, source="combined"
    )
    if not combined_cluster_table.empty:
        combined_cluster_table["test_fraction"] = [
            float(np.mean(source_labels[clusters_combined == cluster] == "test"))
            for cluster in combined_cluster_table["cluster"]
        ]
        combined_cluster_table["puzzle_fraction"] = (
            1.0 - combined_cluster_table["test_fraction"]
        )
    results["exp5_clusters"] = {
        "global_test": {
            "global_phase_purity": global_phase_purity,
            "clusters": global_clusters_table.to_dict(orient="records")
            if not global_clusters_table.empty
            else [],
            "failures": global_clusters_table[global_clusters_table["failure"]].to_dict(
                orient="records"
            )
            if not global_clusters_table.empty
            else [],
        },
        "puzzle_first": {
            "global_theme_purity": float(
                first_cluster_table["global_theme_purity"].iloc[0]
            )
            if not first_cluster_table.empty
            else None,
            "clusters": first_cluster_table.to_dict(orient="records")
            if not first_cluster_table.empty
            else [],
            "failures": first_cluster_table[first_cluster_table["failure"]].to_dict(
                orient="records"
            )
            if not first_cluster_table.empty
            else [],
        },
        "puzzle_last": {
            "global_theme_purity": float(
                last_cluster_table["global_theme_purity"].iloc[0]
            )
            if last_cluster_table is not None and not last_cluster_table.empty
            else None,
            "clusters": last_cluster_table.to_dict(orient="records")
            if last_cluster_table is not None and not last_cluster_table.empty
            else [],
            "failures": last_cluster_table[last_cluster_table["failure"]].to_dict(
                orient="records"
            )
            if last_cluster_table is not None and not last_cluster_table.empty
            else [],
        },
        "combined": {
            "source_knn": combined_source_purity,
            "source_silhouette_umap": combined_source_silhouette,
            "hdbscan_n_clusters": int(
                len([cluster for cluster in set(clusters_combined) if cluster != -1])
            ),
            "hdbscan_noise_fraction": float(np.mean(clusters_combined == -1)),
            "clusters": combined_cluster_table.to_dict(orient="records")
            if not combined_cluster_table.empty
            else [],
        },
    }

    global_clusters_table.to_csv(
        config.out_dir / "global_test_clusters.csv", index=False
    )
    first_cluster_table.to_csv(
        config.out_dir / "puzzle_first_clusters.csv", index=False
    )
    if last_cluster_table is not None:
        last_cluster_table.to_csv(
            config.out_dir / "puzzle_last_clusters.csv", index=False
        )
    combined_cluster_table.to_csv(
        config.out_dir / "combined_global_clusters.csv", index=False
    )
    plot_global_manifold(config, global_frame, umap_global)
    plot_theme_first_last(
        config,
        first_frame,
        umap_first,
        puzzle_frames.get("last"),
        umap_last,
    )
    plot_temporal(config, trajectory_frame, trajectory_emb, umap_trajectory)
    plot_factors(config, global_frame, umap_global, ranked)
    plot_clusters(
        config,
        umap_global,
        clusters_global,
        umap_last,
        clusters_last,
        umap_combined,
        source_labels,
    )
    results = jsonable(results)
    (config.out_dir / "results.json").write_text(
        json.dumps(results, indent=2, sort_keys=True)
    )
    return results


def main(argv: Sequence[str] | None = None) -> int:
    config = parse_args(argv)
    results = run(config)
    print(
        json.dumps(
            {
                "out_dir": str(config.out_dir),
                "n_global": results["config"]["global"]["n"],
                "n_trajectory": results["config"]["trajectory"]["n"],
                "puzzles": results["config"]["puzzles"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
