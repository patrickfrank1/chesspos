"""Thin MLflow wrapper for training runs.

The tracker is a no-op unless enabled, so training runs work without MLflow
configured. When enabled it expects a tracking server URI (CLI flag, falling
back to the ``MLFLOW_TRACKING_URI`` environment variable); the token, if any,
is consumed by the MLflow client from its own environment variables and is
never logged here.
"""

from __future__ import annotations

import os

import mlflow


def resolve_tracking_uri(flag: str | None) -> str | None:
    """Resolve the MLflow tracking URI: CLI flag, then env, then disabled."""
    if flag:
        return flag
    return os.environ.get("MLFLOW_TRACKING_URI") or None


class MLflowTracker:
    """No-op unless enabled; forwards to the ``mlflow`` module when enabled."""

    def __init__(
        self,
        enabled: bool,
        tracking_uri: str | None = None,
        experiment: str = "chesspos",
        run_name: str | None = None,
    ) -> None:
        self.enabled = enabled
        if not enabled:
            return
        if tracking_uri:
            mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(experiment)
        mlflow.start_run(run_name=run_name)
        return

    def log_params(self, params: dict) -> None:
        if not self.enabled:
            return
        mlflow.log_params(params)
        return

    def log_metrics(self, metrics: dict, step: int | None = None) -> None:
        if not self.enabled:
            return
        mlflow.log_metrics(metrics, step=step)
        return

    def log_artifact(self, local_path: str | os.PathLike) -> None:
        if not self.enabled:
            return
        mlflow.log_artifact(str(local_path))
        return

    def end(self) -> None:
        if not self.enabled:
            return
        mlflow.end_run()
        return
