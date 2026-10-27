from __future__ import annotations

import json
import zlib
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Iterable, Iterator, Sequence

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from huggingface_hub import CommitInfo, HfApi, snapshot_download

from src.dataset.puzzle_variants import _split_stream, split_for_puzzle
from src.dataset.token_stream import TokenStreamEncoder

POSITIONS_FILE = "positions.parquet"
NEAR_QUERIES_FILE = "near_queries.parquet"
NEAR_RELEVANCE_FILE = "near_relevance.parquet"
STRATEGIC_QUERIES_FILE = "strategic_queries.parquet"
MANIFEST_FILE = "manifest.json"
BENCHMARK_SUBDIR = "benchmarks/retrieval"

BENCHMARK_VERSION = 1
DEFAULT_KS = (1, 5, 10)
DEFAULT_CLUSTER_THEMES = (
    "rookEndgame",
    "pawnEndgame",
    "queenRookEndgame",
    "bishopEndgame",
    "knightEndgame",
    "promotion",
    "zugzwang",
)
MAX_STRATEGIC_QUERIES_PER_THEME = 200
DEFAULT_STRATEGIC_POSITIVES = 10
DEFAULT_STRATEGIC_NEGATIVES = 490


@dataclass(frozen=True)
class BenchmarkConfig:
    data_dir: str
    output_dir: str
    val_ratio: float = 0.1
    max_positions: int = 10_000
    max_queries: int = 1_000
    near_hamming: int = 2
    strategic_positives: int = DEFAULT_STRATEGIC_POSITIVES
    strategic_negatives: int = DEFAULT_STRATEGIC_NEGATIVES
    cluster_themes: tuple[str, ...] = DEFAULT_CLUSTER_THEMES
    seed: int = 0

    def __post_init__(self):
        if not 0 < self.val_ratio < 1:
            raise ValueError("val_ratio must be between 0 and 1")
        if self.max_positions <= 0:
            raise ValueError("max_positions must be positive")
        if self.max_queries <= 0:
            raise ValueError("max_queries must be positive")
        if self.near_hamming < 0:
            raise ValueError("near_hamming must be non-negative")
        if self.strategic_positives <= 0:
            raise ValueError("strategic_positives must be positive")
        if self.strategic_negatives <= 0:
            raise ValueError("strategic_negatives must be positive")
        return


@dataclass(frozen=True)
class BenchPosition:
    position_id: int
    puzzle_id: str
    ply: int
    fen: str
    tokens: tuple[int, ...]
    themes: tuple[str, ...]


@dataclass(frozen=True)
class RetrievalBenchmark:
    positions: tuple[BenchPosition, ...]
    near_queries: np.ndarray
    near_relevance: dict[int, dict[int, int]]
    strategic_queries: dict[str, np.ndarray]
    manifest: dict


def _iter_validation_positions(
    data_dir: str | Path,
    val_ratio: float,
    cluster_themes: Sequence[str],
) -> Iterator[BenchPosition]:
    encoder = TokenStreamEncoder()
    source_paths: list[tuple[Path, set[str]]] = []
    for path in sorted(Path(data_dir).rglob("*.parquet")):
        names = set(pq.read_schema(path).names)
        if {"puzzle_id", "packed"}.issubset(names):
            source_paths.append((path, names))
    if not source_paths:
        raise ValueError(f"No source parquet shards found under {data_dir}")

    for path, names in source_paths:
        theme_columns = [theme for theme in cluster_themes if theme in names]
        table = pq.read_table(path, columns=["puzzle_id", "packed", *theme_columns])
        theme_values = {
            theme: table.column(theme).to_pylist() for theme in theme_columns
        }
        puzzle_ids = table.column("puzzle_id").to_pylist()
        packed = table.column("packed").combine_chunks()
        offsets = np.asarray(packed.offsets)
        values = packed.flatten().to_numpy(zero_copy_only=False).astype(np.int16)

        for row, puzzle_id in enumerate(puzzle_ids):
            puzzle_id = str(puzzle_id)
            if split_for_puzzle(puzzle_id, val_ratio) != "val":
                continue
            segments = _split_stream(values[offsets[row] : offsets[row + 1]])
            themes = tuple(
                theme for theme in theme_columns if theme_values[theme][row] == 1
            )
            for ply, segment in enumerate(segments):
                board = encoder.decode(segment)
                yield BenchPosition(
                    position_id=-1,
                    puzzle_id=puzzle_id,
                    ply=ply,
                    fen=board.fen(),
                    tokens=tuple(int(token) for token in segment),
                    themes=themes,
                )
        continue


def _reservoir_sample(
    positions: Iterable[BenchPosition], limit: int, rng: np.random.Generator
) -> list[BenchPosition]:
    reservoir: list[BenchPosition] = []
    for index, position in enumerate(positions):
        if len(reservoir) < limit:
            reservoir.append(position)
        else:
            slot = int(rng.integers(0, index + 1))
            if slot < limit:
                reservoir[slot] = position
    return reservoir


def _mine_near(
    positions: Sequence[BenchPosition],
    max_queries: int,
    near_hamming: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, dict[int, dict[int, int]]]:
    tokens = np.asarray([position.tokens for position in positions], dtype=np.int16)
    squares = tokens[:, :64]
    puzzle_ids = np.asarray([position.puzzle_id for position in positions])

    query_ids: list[int] = []
    relevance: dict[int, dict[int, int]] = {}
    claimed: set[tuple[int, int]] = set()
    for query in rng.permutation(len(positions)):
        if len(query_ids) >= max_queries:
            break
        cross_puzzle = puzzle_ids != puzzle_ids[query]
        differences = (squares != squares[query]).sum(axis=1)
        candidates = np.flatnonzero(cross_puzzle & (differences <= near_hamming))
        if candidates.size == 0:
            continue
        fresh = [
            int(candidate)
            for candidate in candidates
            if (
                min(int(query), int(candidate)),
                max(int(query), int(candidate)),
            )
            not in claimed
        ]
        if not fresh:
            continue
        fresh_ids = np.asarray(fresh, dtype=np.int64)
        exact = (tokens[fresh_ids] == tokens[query]).all(axis=1)
        grades = np.maximum(1, differences[fresh_ids]).astype(np.int8)
        grades[exact] = 0
        relevance[int(query)] = {
            candidate: int(grade) for candidate, grade in zip(fresh, grades)
        }
        claimed.update(
            (min(int(query), candidate), max(int(query), candidate))
            for candidate in fresh
        )
        query_ids.append(int(query))

    return np.asarray(query_ids, dtype=np.int64), relevance


def _mine_strategic(
    positions: Sequence[BenchPosition],
    max_queries: int,
    rng: np.random.Generator,
) -> dict[str, np.ndarray]:
    themes = sorted({theme for position in positions for theme in position.themes})
    strategic: dict[str, np.ndarray] = {}
    for theme in themes:
        indices = [
            index
            for index, position in enumerate(positions)
            if theme in position.themes
        ]
        candidates = [
            index
            for index in indices
            if any(
                positions[other].puzzle_id != positions[index].puzzle_id
                for other in indices
            )
        ]
        if not candidates:
            continue
        order = rng.permutation(len(candidates))
        selected = [candidates[int(index)] for index in order[:max_queries]]
        strategic[theme] = np.asarray(selected, dtype=np.int64)
    return strategic


def _write_positions(path: Path, positions: Sequence[BenchPosition]) -> None:
    table = pa.table(
        {
            "position_id": pa.array(
                [position.position_id for position in positions], type=pa.int64()
            ),
            "puzzle_id": pa.array(
                [position.puzzle_id for position in positions], type=pa.string()
            ),
            "ply": pa.array([position.ply for position in positions], type=pa.int32()),
            "fen": pa.array([position.fen for position in positions], type=pa.string()),
            "tokens": pa.array(
                [list(position.tokens) for position in positions],
                type=pa.list_(pa.int16()),
            ),
            "themes": pa.array(
                [list(position.themes) for position in positions],
                type=pa.list_(pa.string()),
            ),
        }
    )
    pq.write_table(table, path)
    return


def _write_near(
    query_path: Path,
    relevance_path: Path,
    query_ids: np.ndarray,
    relevance: dict[int, dict[int, int]],
) -> None:
    pq.write_table(
        pa.table({"query_position_id": pa.array(query_ids, type=pa.int64())}),
        query_path,
    )
    query_column: list[int] = []
    candidate_column: list[int] = []
    grade_column: list[int] = []
    for query in query_ids:
        for candidate, grade in sorted(relevance[int(query)].items()):
            query_column.append(int(query))
            candidate_column.append(candidate)
            grade_column.append(grade)
    pq.write_table(
        pa.table(
            {
                "query_position_id": pa.array(query_column, type=pa.int64()),
                "candidate_position_id": pa.array(candidate_column, type=pa.int64()),
                "grade": pa.array(grade_column, type=pa.int8()),
            }
        ),
        relevance_path,
    )
    return


def _write_strategic(path: Path, strategic: dict[str, np.ndarray]) -> None:
    query_column: list[int] = []
    theme_column: list[str] = []
    for theme in sorted(strategic):
        for query in strategic[theme]:
            query_column.append(int(query))
            theme_column.append(theme)
    pq.write_table(
        pa.table(
            {
                "query_position_id": pa.array(query_column, type=pa.int64()),
                "theme": pa.array(theme_column, type=pa.string()),
            }
        ),
        path,
    )
    return


def build_benchmark(config: BenchmarkConfig) -> Path:
    rng = np.random.default_rng(config.seed)
    sampled = _reservoir_sample(
        _iter_validation_positions(
            config.data_dir, config.val_ratio, config.cluster_themes
        ),
        config.max_positions,
        rng,
    )
    if not sampled:
        raise ValueError("benchmark has no validation positions")

    sampled.sort(key=lambda position: (position.puzzle_id, position.ply))
    positions = [
        replace(position, position_id=index) for index, position in enumerate(sampled)
    ]
    near_queries, near_relevance = _mine_near(
        positions, config.max_queries, config.near_hamming, rng
    )
    strategic_queries = _mine_strategic(
        positions,
        min(config.max_queries, MAX_STRATEGIC_QUERIES_PER_THEME),
        rng,
    )

    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_positions(output_dir / POSITIONS_FILE, positions)
    _write_near(
        output_dir / NEAR_QUERIES_FILE,
        output_dir / NEAR_RELEVANCE_FILE,
        near_queries,
        near_relevance,
    )
    _write_strategic(output_dir / STRATEGIC_QUERIES_FILE, strategic_queries)

    manifest = {
        "version": BENCHMARK_VERSION,
        "seed": config.seed,
        "val_ratio": config.val_ratio,
        "max_positions": config.max_positions,
        "max_queries": config.max_queries,
        "near_hamming": config.near_hamming,
        "strategic_positives": config.strategic_positives,
        "strategic_negatives": config.strategic_negatives,
        "cluster_themes": list(config.cluster_themes),
        "n_positions": len(positions),
        "n_near_queries": int(near_queries.size),
        "n_near_pairs": sum(len(pairs) for pairs in near_relevance.values()),
        "n_strategic_queries": int(
            sum(queries.size for queries in strategic_queries.values())
        ),
    }
    (output_dir / MANIFEST_FILE).write_text(
        json.dumps(manifest, indent=2, sort_keys=True)
    )
    return output_dir


def resolve_benchmark(
    source: str | Path,
    revision: str | None = None,
    cache_dir: str | None = None,
) -> Path:
    """Resolve a local benchmark directory or download it from the Hub."""
    path = Path(source)
    if path.is_dir():
        resolved = path
    else:
        snapshot = snapshot_download(
            repo_id=str(source),
            repo_type="dataset",
            revision=revision,
            allow_patterns=[f"{BENCHMARK_SUBDIR}/**"],
            cache_dir=cache_dir,
        )
        resolved = Path(snapshot) / BENCHMARK_SUBDIR
    if not (resolved / MANIFEST_FILE).exists():
        raise ValueError(
            f"No retrieval benchmark manifest found at {resolved}; "
            f"build one with src.run.build_retrieval_benchmark"
        )
    return resolved


def upload_benchmark(
    path: str | Path,
    repo_id: str,
    subfolder: str = BENCHMARK_SUBDIR,
    revision: str | None = None,
    commit_message: str | None = None,
    token: str | None = None,
) -> CommitInfo:
    """Upload a benchmark directory into a HuggingFace dataset repository."""
    api = HfApi()
    return api.upload_folder(
        folder_path=str(path),
        repo_id=repo_id,
        repo_type="dataset",
        path_in_repo=subfolder,
        revision=revision,
        commit_message=commit_message,
        token=token,
    )


def load_benchmark(
    source: str | Path,
    revision: str | None = None,
    cache_dir: str | None = None,
) -> RetrievalBenchmark:
    path = resolve_benchmark(source, revision=revision, cache_dir=cache_dir)
    manifest = json.loads((path / MANIFEST_FILE).read_text())

    table = pq.read_table(path / POSITIONS_FILE)
    position_ids = table.column("position_id").to_pylist()
    puzzle_ids = table.column("puzzle_id").to_pylist()
    plies = table.column("ply").to_pylist()
    fens = table.column("fen").to_pylist()
    token_rows = table.column("tokens").to_pylist()
    theme_rows = table.column("themes").to_pylist()
    positions = tuple(
        BenchPosition(
            position_id=int(position_id),
            puzzle_id=str(puzzle_id),
            ply=int(ply),
            fen=str(fen),
            tokens=tuple(int(token) for token in tokens),
            themes=tuple(str(theme) for theme in themes),
        )
        for position_id, puzzle_id, ply, fen, tokens, themes in zip(
            position_ids, puzzle_ids, plies, fens, token_rows, theme_rows
        )
    )

    near_queries = np.asarray(
        pq.read_table(path / NEAR_QUERIES_FILE).column("query_position_id").to_pylist(),
        dtype=np.int64,
    )
    near_relevance: dict[int, dict[int, int]] = {}
    relevance_table = pq.read_table(path / NEAR_RELEVANCE_FILE)
    for query, candidate, grade in zip(
        relevance_table.column("query_position_id").to_pylist(),
        relevance_table.column("candidate_position_id").to_pylist(),
        relevance_table.column("grade").to_pylist(),
    ):
        near_relevance.setdefault(int(query), {})[int(candidate)] = int(grade)

    strategic_queries: dict[str, np.ndarray] = {}
    strategic_table = pq.read_table(path / STRATEGIC_QUERIES_FILE)
    themes = strategic_table.column("theme").to_pylist()
    queries = strategic_table.column("query_position_id").to_pylist()
    for theme, query in zip(themes, queries):
        strategic_queries.setdefault(str(theme), []).append(int(query))
    strategic_queries = {
        theme: np.asarray(values, dtype=np.int64)
        for theme, values in strategic_queries.items()
    }

    return RetrievalBenchmark(
        positions=positions,
        near_queries=near_queries,
        near_relevance=near_relevance,
        strategic_queries=strategic_queries,
        manifest=manifest,
    )


def _normalize(embeddings: np.ndarray) -> np.ndarray:
    embeddings = np.asarray(embeddings, dtype=np.float64)
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    return embeddings / np.clip(norms, 1e-12, None)


def retrieval_scores(
    embeddings: np.ndarray,
    query_ids: np.ndarray,
    gallery_ids: np.ndarray,
    puzzle_ids: Sequence[str],
) -> np.ndarray:
    """Cosine scores for queries against a gallery, masking same-puzzle items."""
    normalized = _normalize(embeddings)
    scores = normalized[np.asarray(query_ids)] @ normalized[np.asarray(gallery_ids)].T
    puzzles = np.asarray(puzzle_ids)
    masked = (
        puzzles[np.asarray(gallery_ids)][None, :]
        == puzzles[np.asarray(query_ids)][:, None]
    )
    scores = np.array(scores, dtype=np.float64, copy=True)
    scores[masked] = -np.inf
    return scores


def _recall_hits(scores: np.ndarray, relevance: Sequence[dict], k: int) -> np.ndarray:
    hits = []
    for row, relevant in enumerate(relevance):
        if not relevant:
            continue
        order = np.argsort(-scores[row])[:k]
        hits.append(1.0 if any(int(index) in relevant for index in order) else 0.0)
    return np.asarray(hits, dtype=np.float64)


def ranking_metrics(
    scores: np.ndarray, relevance: Sequence[dict], ks: Sequence[int] = DEFAULT_KS
) -> dict[str, float]:
    scores = np.asarray(scores, dtype=np.float64)
    covered = [row for row, relevant in enumerate(relevance) if relevant]
    coverage = len(covered) / len(relevance) if relevance else 0.0
    metrics: dict[str, float] = {"coverage": float(coverage)}
    if not covered:
        for k in ks:
            metrics[f"recall@{k}"] = 0.0
            metrics[f"ndcg@{k}"] = 0.0
        metrics["map"] = 0.0
        return metrics

    for k in ks:
        metrics[f"recall@{k}"] = float(_recall_hits(scores, relevance, k).mean())

    average_precisions = []
    ndcgs = {k: [] for k in ks}
    for row in covered:
        relevant = relevance[row]
        order = np.argsort(-scores[row])
        hits = 0
        precision_sum = 0.0
        for rank, index in enumerate(order, start=1):
            if int(index) in relevant:
                hits += 1
                precision_sum += hits / rank
        average_precisions.append(precision_sum / len(relevant))

        gains = {int(index): 1.0 / (1.0 + grade) for index, grade in relevant.items()}
        ideal = sorted(gains.values(), reverse=True)
        for k in ks:
            dcg = sum(
                gains.get(int(index), 0.0) / np.log2(rank + 1)
                for rank, index in enumerate(order[:k], start=1)
            )
            idcg = sum(
                gain / np.log2(rank + 1) for rank, gain in enumerate(ideal[:k], start=1)
            )
            ndcgs[k].append(dcg / idcg if idcg > 0 else 0.0)

    metrics["map"] = float(np.mean(average_precisions))
    for k in ks:
        metrics[f"ndcg@{k}"] = float(np.mean(ndcgs[k]))
    return metrics


def bootstrap_ci(
    values: np.ndarray, seed: int = 0, n_boot: int = 1_000
) -> tuple[float, float]:
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        return 0.0, 0.0
    rng = np.random.default_rng(seed)
    means = np.array(
        [
            rng.choice(values, size=values.size, replace=True).mean()
            for _ in range(n_boot)
        ]
    )
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def effective_rank(embeddings: np.ndarray) -> float:
    embeddings = np.asarray(embeddings, dtype=np.float64)
    centered = embeddings - embeddings.mean(axis=0, keepdims=True)
    covariance = centered.T @ centered / max(1, len(centered) - 1)
    eigenvalues = np.clip(np.linalg.eigvalsh(covariance), 0.0, None)
    total = float(eigenvalues.sum())
    if total <= 1e-12:
        return 1.0
    return float(total**2 / float((eigenvalues**2).sum()))


def alignment_uniformity(
    embeddings: np.ndarray,
    positive_pairs: Sequence[tuple[int, int]],
    seed: int = 0,
) -> dict[str, float]:
    normalized = _normalize(embeddings)
    if positive_pairs:
        left = normalized[[pair[0] for pair in positive_pairs]]
        right = normalized[[pair[1] for pair in positive_pairs]]
        alignment = float(((left - right) ** 2).sum(axis=1).mean())
    else:
        alignment = 0.0

    if len(normalized) < 2:
        uniformity = 0.0
    else:
        rng = np.random.default_rng(seed)
        left_ids = rng.integers(0, len(normalized), size=10_000)
        right_ids = rng.integers(0, len(normalized), size=10_000)
        keep = left_ids != right_ids
        distances = (
            (normalized[left_ids[keep]] - normalized[right_ids[keep]]) ** 2
        ).sum(axis=1)
        uniformity = float(np.log(np.exp(-2.0 * distances).mean()))
    return {"alignment": alignment, "uniformity": uniformity}


def evaluate_near(
    benchmark: RetrievalBenchmark,
    embeddings: np.ndarray,
    ks: Sequence[int] = DEFAULT_KS,
) -> dict[str, float]:
    gallery_ids = np.arange(len(benchmark.positions))
    puzzle_ids = [position.puzzle_id for position in benchmark.positions]
    query_ids = np.asarray(benchmark.near_queries, dtype=np.int64)
    if query_ids.size == 0:
        metrics = ranking_metrics(np.zeros((0, len(benchmark.positions))), [], ks=ks)
        return metrics

    scores = retrieval_scores(embeddings, query_ids, gallery_ids, puzzle_ids)
    relevance = [benchmark.near_relevance.get(int(query), {}) for query in query_ids]
    metrics = ranking_metrics(scores, relevance, ks=ks)
    seed = int(benchmark.manifest.get("seed", 0))
    for k in ks:
        low, high = bootstrap_ci(_recall_hits(scores, relevance, k), seed=seed)
        metrics[f"recall@{k}_ci_low"] = low
        metrics[f"recall@{k}_ci_high"] = high
    return metrics


def _sample_ids(ids: Sequence[int], count: int, rng: np.random.Generator) -> list[int]:
    ids = list(ids)
    if len(ids) <= count:
        return sorted(ids)
    order = rng.permutation(len(ids))
    return [ids[int(index)] for index in order[:count]]


def strategic_pool(
    benchmark: RetrievalBenchmark,
    theme: str,
    query: int,
    positives: int | None = None,
    negatives: int | None = None,
) -> tuple[np.ndarray, dict[int, int]]:
    """Sample a balanced gallery for one strategic query.

    Positives are cross-puzzle positions carrying the theme; negatives are
    cross-puzzle positions without it. The relevance map uses local pool
    indices so it can be passed straight to :func:`ranking_metrics`.
    """
    query = int(query)
    if theme not in benchmark.strategic_queries:
        raise ValueError(f"Theme {theme!r} is not in the benchmark")
    if theme not in benchmark.positions[query].themes:
        raise ValueError(f"Position {query} does not carry theme {theme!r}")
    if positives is None:
        positives = int(
            benchmark.manifest.get("strategic_positives", DEFAULT_STRATEGIC_POSITIVES)
        )
    if negatives is None:
        negatives = int(
            benchmark.manifest.get("strategic_negatives", DEFAULT_STRATEGIC_NEGATIVES)
        )
    if positives <= 0 or negatives <= 0:
        raise ValueError("positives and negatives must be positive")

    query_puzzle = benchmark.positions[query].puzzle_id
    positive_ids: list[int] = []
    negative_ids: list[int] = []
    for index, position in enumerate(benchmark.positions):
        if position.puzzle_id == query_puzzle:
            continue
        if theme in position.themes:
            positive_ids.append(index)
        else:
            negative_ids.append(index)

    seed = int(benchmark.manifest.get("seed", 0))
    theme_seed = zlib.crc32(theme.encode("utf-8"))
    rng = np.random.default_rng([seed, theme_seed, query])
    chosen_positives = _sample_ids(positive_ids, positives, rng)
    chosen_negatives = _sample_ids(negative_ids, negatives, rng)
    gallery = np.asarray(chosen_positives + chosen_negatives, dtype=np.int64)
    relevance = {index: 0 for index in range(len(chosen_positives))}
    return gallery, relevance


def strategic_theme_metrics(
    benchmark: RetrievalBenchmark,
    embeddings: np.ndarray,
    ks: Sequence[int] = DEFAULT_KS,
) -> dict[str, dict[str, float]]:
    """Per-theme retrieval metrics over balanced same-theme pools."""
    normalized = _normalize(embeddings)
    theme_metrics: dict[str, dict[str, float]] = {}
    for theme, query_ids in sorted(benchmark.strategic_queries.items()):
        per_query: list[dict[str, float]] = []
        chance_values: list[float] = []
        for query in query_ids:
            gallery, relevance = strategic_pool(benchmark, theme, int(query))
            if not relevance or len(relevance) == len(gallery):
                continue
            scores = normalized[[int(query)]] @ normalized[gallery].T
            per_query.append(ranking_metrics(scores, [relevance], ks=ks))
            chance_values.append(len(relevance) / len(gallery))
        if not per_query:
            continue
        metrics = {
            key: float(np.mean([row[key] for row in per_query])) for key in per_query[0]
        }
        metrics["n_queries"] = len(per_query)
        metrics["chance_map"] = float(np.mean(chance_values))
        metrics["map_lift"] = float(metrics["map"] - metrics["chance_map"])
        theme_metrics[theme] = metrics
    return theme_metrics


def _aggregate_strategic_metrics(
    per_theme: dict[str, dict[str, float]], ks: Sequence[int]
) -> dict[str, float]:
    if not per_theme:
        metrics = {f"recall@{k}": 0.0 for k in ks}
        metrics.update({f"ndcg@{k}": 0.0 for k in ks})
        metrics["map"] = 0.0
        metrics["coverage"] = 0.0
        metrics["chance_map"] = 0.0
        metrics["map_lift"] = 0.0
        metrics["n_queries"] = 0
        return metrics
    keys = next(iter(per_theme.values())).keys()
    metrics = {
        key: float(np.mean([theme[key] for theme in per_theme.values()]))
        for key in keys
        if key != "n_queries"
    }
    metrics["n_queries"] = int(sum(theme["n_queries"] for theme in per_theme.values()))
    return metrics


def evaluate_strategic(
    benchmark: RetrievalBenchmark,
    embeddings: np.ndarray,
    ks: Sequence[int] = DEFAULT_KS,
) -> dict[str, float]:
    per_theme = strategic_theme_metrics(benchmark, embeddings, ks=ks)
    return _aggregate_strategic_metrics(per_theme, ks)


def embed_positions(
    encoder: torch.nn.Module,
    tokens: Sequence[tuple[int, ...]],
    device: torch.device,
    batch_size: int = 128,
) -> np.ndarray:
    from src.dataset.puzzle_variants import pack_segments

    was_training = encoder.training
    encoder.eval()
    rows: list[np.ndarray] = []
    try:
        with (
            torch.no_grad(),
            torch.autocast(
                device_type=device.type,
                dtype=torch.bfloat16,
                enabled=device.type == "cuda",
            ),
        ):
            for start in range(0, len(tokens), batch_size):
                batch = tokens[start : start + batch_size]
                packed = [
                    pack_segments([np.asarray(row, dtype=np.int16)]) for row in batch
                ]
                token_tensor = torch.from_numpy(
                    np.stack([item[0] for item in packed])
                ).long()
                segment_tensor = torch.from_numpy(
                    np.stack([item[1] for item in packed])
                ).long()
                embeddings, _ = encoder.embed(
                    token_tensor.to(device), segment_tensor.to(device)
                )
                rows.append(embeddings[:, 0].float().cpu().numpy())
    finally:
        encoder.train(was_training)
    return np.concatenate(rows) if rows else np.zeros((0, 0))


def evaluate_retrieval(
    encoder: torch.nn.Module,
    benchmark: RetrievalBenchmark,
    device: torch.device | str,
    batch_size: int = 128,
    ks: Sequence[int] = DEFAULT_KS,
) -> dict[str, float]:
    device = torch.device(device) if isinstance(device, str) else device
    tokens = [position.tokens for position in benchmark.positions]
    embeddings = embed_positions(encoder, tokens, device, batch_size=batch_size)
    metrics = {
        f"near_{key}": value
        for key, value in evaluate_near(benchmark, embeddings, ks=ks).items()
    }
    theme_metrics = strategic_theme_metrics(benchmark, embeddings, ks=ks)
    metrics.update(
        {
            f"strategic_{key}": value
            for key, value in _aggregate_strategic_metrics(theme_metrics, ks).items()
        }
    )
    for theme, values in theme_metrics.items():
        for key in ("map", "chance_map", "map_lift", "n_queries"):
            metrics[f"strategic_theme_{theme}_{key}"] = values[key]
    metrics["rank_effective"] = effective_rank(embeddings)
    positive_pairs = [
        (int(query), int(candidate))
        for query, candidates in benchmark.near_relevance.items()
        for candidate in candidates
    ]
    metrics.update(
        alignment_uniformity(
            embeddings,
            positive_pairs,
            seed=int(benchmark.manifest.get("seed", 0)),
        )
    )
    return metrics
