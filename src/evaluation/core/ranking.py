from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np

DEFAULT_KS = (1, 5, 10)


def normalize(embeddings: np.ndarray) -> np.ndarray:
    """Return row-wise L2-normalized float64 embeddings, zero-safe."""
    values = np.asarray(embeddings, dtype=np.float64)
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    return values / np.clip(norms, 1e-12, None)


def cosine_scores(
    embeddings: np.ndarray,
    query_ids: np.ndarray,
    gallery_ids: np.ndarray,
) -> np.ndarray:
    """Cosine scores for query rows against gallery rows."""
    values = normalize(embeddings)
    return values[np.asarray(query_ids)] @ values[np.asarray(gallery_ids)].T


def rank_of_first_relevant(scores: np.ndarray, relevant: Mapping[int, int]) -> float:
    """Average rank of the highest-scored relevant item, with ties split."""
    values = np.asarray(scores, dtype=np.float64)
    relevant_ids = np.fromiter((int(index) for index in relevant), dtype=np.int64)
    if relevant_ids.size == 0:
        raise ValueError("relevant cannot be empty")
    best = float(values[relevant_ids].max())
    better = float(np.sum(values > best))
    tied = float(np.sum(values == best)) - 1.0
    return 1.0 + better + 0.5 * max(0.0, tied)


def recall_hits(
    scores: np.ndarray, relevance: Sequence[Mapping[int, int]], k: int
) -> np.ndarray:
    """Return one hit indicator per query that has relevant candidates."""
    values = np.asarray(scores, dtype=np.float64)
    hits = []
    for row, relevant in enumerate(relevance):
        if not relevant:
            continue
        order = np.argsort(-values[row])[:k]
        hits.append(1.0 if any(int(index) in relevant for index in order) else 0.0)
    return np.asarray(hits, dtype=np.float64)


def chance_metrics(k: int) -> dict[str, float]:
    """Random-ranking baselines for a query with k candidates."""
    return {
        "chance_top1": 1.0 / k,
        "chance_mrr": sum(1.0 / rank for rank in range(1, k + 1)) / k,
        "chance_mean_rank": (k + 1) / 2.0,
    }


def ranking_metrics(
    scores: np.ndarray,
    relevance: Sequence[Mapping[int, int]],
    ks: Sequence[int] = DEFAULT_KS,
) -> dict[str, float]:
    """Unified fixed-size ranking and graded retrieval metrics."""
    values = np.asarray(scores, dtype=np.float64)
    if values.ndim != 2:
        raise ValueError("scores must be a 2D array")
    if len(relevance) != values.shape[0]:
        raise ValueError("relevance must have one entry per score row")

    covered = [row for row, relevant in enumerate(relevance) if relevant]
    metrics: dict[str, float] = {
        "coverage": len(covered) / len(relevance) if relevance else 0.0,
        "n_queries": len(covered),
        "top1": 0.0,
        "mrr": 0.0,
        "mean_rank": 0.0,
        "map": 0.0,
    }
    for k in ks:
        metrics[f"recall@{k}"] = 0.0
        metrics[f"ndcg@{k}"] = 0.0
    if not covered:
        return metrics

    ranks = np.asarray(
        [rank_of_first_relevant(values[row], relevance[row]) for row in covered],
        dtype=np.float64,
    )
    metrics["top1"] = float(np.mean(ranks <= 1.0))
    metrics["mrr"] = float(np.mean(1.0 / ranks))
    metrics["mean_rank"] = float(np.mean(ranks))

    for k in ks:
        metrics[f"recall@{k}"] = float(recall_hits(values, relevance, k).mean())

    average_precisions = []
    ndcgs = {k: [] for k in ks}
    for row in covered:
        relevant = relevance[row]
        order = np.argsort(-values[row])
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
