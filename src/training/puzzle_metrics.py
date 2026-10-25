from __future__ import annotations

import numpy as np


def _auroc(scores: np.ndarray, labels: np.ndarray) -> float:
    n_pos = int(labels.sum())
    n_neg = int(labels.size - n_pos)
    if n_pos == 0 or n_neg == 0:
        return 0.5

    order = np.argsort(scores, kind="mergesort")
    sorted_scores = scores[order]
    ranks = np.empty(scores.size, dtype=np.float64)
    start = 0
    while start < scores.size:
        end = start
        while end + 1 < scores.size and sorted_scores[end + 1] == sorted_scores[start]:
            end += 1
        ranks[order[start : end + 1]] = (start + end) / 2.0 + 1.0
        start = end + 1

    rank_sum = ranks[labels == 1].sum()
    return float((rank_sum - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def _average_precision(scores: np.ndarray, labels: np.ndarray) -> float:
    n_pos = int(labels.sum())
    if n_pos == 0:
        return 0.0

    order = np.argsort(-scores, kind="mergesort")
    sorted_scores = scores[order]
    sorted_labels = labels[order].astype(np.float64)
    true_positives = np.cumsum(sorted_labels)
    false_positives = np.cumsum(1.0 - sorted_labels)
    group_ends = np.r_[np.flatnonzero(np.diff(sorted_scores) != 0), scores.size - 1]
    precision = true_positives[group_ends] / (
        true_positives[group_ends] + false_positives[group_ends]
    )
    recall = true_positives[group_ends] / n_pos
    return float(np.sum(np.diff(np.r_[0.0, recall]) * precision))


def _threshold_metrics(
    scores: np.ndarray, labels: np.ndarray, threshold: float
) -> dict[str, float]:
    predictions = scores >= threshold
    positives = labels == 1
    true_positives = int(np.sum(predictions & positives))
    false_positives = int(np.sum(predictions & ~positives))
    false_negatives = int(np.sum(~predictions & positives))
    true_negatives = int(np.sum(~predictions & ~positives))

    precision = true_positives / max(1, true_positives + false_positives)
    recall = true_positives / max(1, true_positives + false_negatives)
    f1 = (
        2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0
    )
    accuracy = (true_positives + true_negatives) / max(1, labels.size)
    return {
        "threshold": float(threshold),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "accuracy": float(accuracy),
        "true_positives": float(true_positives),
        "false_positives": float(false_positives),
        "false_negatives": float(false_negatives),
        "true_negatives": float(true_negatives),
    }


def binary_metrics(
    scores: np.ndarray | list[float],
    labels: np.ndarray | list[int],
    threshold: float | None = None,
) -> dict[str, float]:
    """Ranking and threshold metrics for a binary classifier."""
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)
    if scores.shape != labels.shape:
        raise ValueError("scores and labels must have the same shape")
    if scores.size == 0:
        raise ValueError("cannot compute metrics on an empty set")

    if threshold is None:
        candidates = np.unique(scores)[::-1]
        best = _threshold_metrics(scores, labels, float(candidates[0]))
        for candidate in candidates[1:]:
            current = _threshold_metrics(scores, labels, float(candidate))
            if (current["f1"], current["threshold"]) > (
                best["f1"],
                best["threshold"],
            ):
                best = current
    else:
        best = _threshold_metrics(scores, labels, float(threshold))

    return {
        "auroc": _auroc(scores, labels),
        "pr_auc": _average_precision(scores, labels),
        **best,
    }
