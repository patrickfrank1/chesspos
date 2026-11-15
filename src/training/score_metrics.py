from __future__ import annotations

import numpy as np

from src.training.puzzle_metrics import auroc


def _as_arrays(
    predictions: np.ndarray | list[float],
    targets: np.ndarray | list[float],
) -> tuple[np.ndarray, np.ndarray]:
    predictions = np.asarray(predictions, dtype=np.float64).ravel()
    targets = np.asarray(targets, dtype=np.float64).ravel()
    if predictions.shape != targets.shape:
        raise ValueError("predictions and targets must have the same shape")
    if predictions.size == 0:
        raise ValueError("cannot compute score metrics on an empty set")
    return predictions, targets


def _average_ranks(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    sorted_values = values[order]
    ranks = np.empty(values.size, dtype=np.float64)
    start = 0
    while start < values.size:
        end = start
        while end + 1 < values.size and sorted_values[end + 1] == sorted_values[start]:
            end += 1
        ranks[order[start : end + 1]] = (start + end) / 2.0 + 1.0
        start = end + 1
    return ranks


def _correlation(left: np.ndarray, right: np.ndarray) -> float:
    if left.size < 2 or left.std() == 0 or right.std() == 0:
        return 0.0
    return float(np.corrcoef(left, right)[0, 1])


def score_metrics(
    predictions: np.ndarray | list[float],
    targets: np.ndarray | list[float],
) -> dict[str, float]:
    """Regression metrics for White-POV scores in [-1, 1]."""
    predictions, targets = _as_arrays(predictions, targets)
    residuals = predictions - targets
    mse = float(np.mean(residuals**2))
    target_variance = float(np.mean((targets - targets.mean()) ** 2))
    if target_variance > 0:
        r2 = 1.0 - mse / target_variance
    else:
        r2 = 1.0 if mse == 0 else 0.0

    decisive = targets != 0
    if decisive.any() and np.unique(targets[decisive]).size > 1:
        decisive_auc = auroc(
            predictions[decisive], (targets[decisive] > 0).astype(np.int64)
        )
    else:
        decisive_auc = 0.5

    return {
        "mse": mse,
        "mae": float(np.mean(np.abs(residuals))),
        "r2": float(r2),
        "pearson": _correlation(predictions, targets),
        "spearman": _correlation(_average_ranks(predictions), _average_ranks(targets)),
        "auroc_decisive": float(decisive_auc),
        "n": float(predictions.size),
        "target_mean": float(targets.mean()),
        "prediction_mean": float(predictions.mean()),
        "prediction_std": float(predictions.std()),
    }


def calibration_bins(
    predictions: np.ndarray | list[float],
    targets: np.ndarray | list[float],
    n_bins: int = 10,
) -> dict[str, np.ndarray]:
    """Fixed-width bins over [-1, 1] with mean prediction and realised score."""
    if n_bins < 1:
        raise ValueError("n_bins must be positive")
    predictions, targets = _as_arrays(predictions, targets)
    edges = np.linspace(-1.0, 1.0, n_bins + 1)
    indices = np.clip(np.digitize(predictions, edges[1:-1]), 0, n_bins - 1)
    counts = np.bincount(indices, minlength=n_bins).astype(np.int64)
    prediction_sum = np.bincount(indices, weights=predictions, minlength=n_bins)
    target_sum = np.bincount(indices, weights=targets, minlength=n_bins)
    valid = counts > 0
    prediction_mean = np.full(n_bins, np.nan)
    target_mean = np.full(n_bins, np.nan)
    prediction_mean[valid] = prediction_sum[valid] / counts[valid]
    target_mean[valid] = target_sum[valid] / counts[valid]
    return {
        "edges": edges,
        "counts": counts,
        "prediction_mean": prediction_mean,
        "target_mean": target_mean,
    }


def expected_calibration_error(
    predictions: np.ndarray | list[float],
    targets: np.ndarray | list[float],
    n_bins: int = 10,
) -> float:
    bins = calibration_bins(predictions, targets, n_bins=n_bins)
    counts = bins["counts"]
    valid = counts > 0
    gaps = np.abs(bins["prediction_mean"][valid] - bins["target_mean"][valid])
    return float((gaps * counts[valid]).sum() / counts.sum())
