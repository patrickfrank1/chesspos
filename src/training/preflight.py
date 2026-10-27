"""Startup dependency checks for the standalone training entry point.

The training/evaluation path imports some heavy or optional packages lazily
(for example matplotlib only when saving the PCA plot). Checking them before
training starts turns a missing package from a failure after the first
evaluation into an immediate, actionable error.
"""

from __future__ import annotations

import importlib
from collections.abc import Sequence

REQUIRED_MODULES: tuple[str, ...] = (
    "chess",
    "huggingface_hub",
    "matplotlib",
    "mlflow",
    "numpy",
    "pyarrow",
    "torch",
)


def check_dependencies(
    modules: Sequence[str] = REQUIRED_MODULES,
) -> None:
    """Import every required module and raise one error listing failures.

    ``modules`` is injectable so tests can exercise missing and present
    dependencies without touching the real environment.
    """
    failures: list[str] = []
    for name in modules:
        try:
            importlib.import_module(name)
        except Exception as exc:
            failures.append(f"{name}: {type(exc).__name__}: {exc}")
    if failures:
        details = "\n".join(f"  - {failure}" for failure in failures)
        raise RuntimeError(
            f"Cannot start training; missing or broken dependencies:\n{details}"
        )
    return
