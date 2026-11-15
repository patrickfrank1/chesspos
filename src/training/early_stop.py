from __future__ import annotations

import math


def auroc_standard_error(auroc: float, n_positive: int, n_negative: int) -> float:
    """Hanley-McNeil standard error of a rank-based AUC estimate."""
    if n_positive <= 0 or n_negative <= 0:
        return 0.0
    q1 = auroc / (2.0 - auroc)
    q2 = 2.0 * auroc * auroc / (1.0 + auroc)
    variance = (
        auroc * (1.0 - auroc)
        + (n_positive - 1) * (q1 - auroc * auroc)
        + (n_negative - 1) * (q2 - auroc * auroc)
    ) / (n_positive * n_negative)
    return math.sqrt(max(variance, 0.0))


def is_at_chance(
    auroc: float,
    n_positive: int,
    n_negative: int,
    z: float = 1.645,
) -> bool:
    """True when the AUC upper confidence bound does not clear 0.5.

    ``z`` is the one-sided normal quantile for the confidence bound (1.645 for
    95%). A run whose AUC is not significantly above chance should be aborted
    rather than trained to completion.
    """
    if not 0.0 <= auroc <= 1.0:
        raise ValueError(f"auroc must be in [0, 1], got {auroc}")
    if n_positive <= 0 or n_negative <= 0:
        return True
    margin = z * auroc_standard_error(auroc, n_positive, n_negative)
    return auroc <= 0.5 + margin
