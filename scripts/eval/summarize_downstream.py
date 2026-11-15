"""Aggregate downstream probe runs into tables and plots for the report."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

from src.training.score_metrics import calibration_bins, score_metrics

PLY_BUCKETS = ((1, 10, "ply:1-10"), (11, 30, "ply:11-30"), (31, 60, "ply:31-60"))
MATERIAL_BUCKETS = (
    (0, 0, "material:0"),
    (1, 3, "material:1-3"),
    (4, 6, "material:4-6"),
    (7, 10**6, "material:7+"),
)


@dataclass(frozen=True)
class RunRecord:
    task: str
    model: str
    recipe: str
    seed: int
    split: str
    metrics: dict[str, float]


def _seed_from(name: str) -> int:
    return int(name.removeprefix("seed"))


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text())


def collect_score_runs(
    training_root: str | Path, eval_root: str | Path
) -> list[RunRecord]:
    runs: list[RunRecord] = []
    training_root = Path(training_root)
    if training_root.exists():
        for path in sorted(training_root.glob("*/*/seed*/metrics.json")):
            recipe, model, seed_dir = path.relative_to(training_root).parts[:3]
            history = _load_json(path).get("history") or []
            if not history:
                continue
            runs.append(
                RunRecord(
                    task="score",
                    model=model,
                    recipe=recipe,
                    seed=_seed_from(seed_dir),
                    split="val",
                    metrics=history[-1]["metrics"],
                )
            )
    eval_root = Path(eval_root)
    if eval_root.exists():
        for path in sorted(eval_root.glob("*/*/seed*/metrics.json")):
            recipe, model, seed_dir = path.relative_to(eval_root).parts[:3]
            runs.append(
                RunRecord(
                    task="score",
                    model=model,
                    recipe=recipe,
                    seed=_seed_from(seed_dir),
                    split="test",
                    metrics=_load_json(path)["metrics"],
                )
            )
    return runs


def collect_sacrifice_runs(root: str | Path) -> list[RunRecord]:
    import torch

    root = Path(root)
    if not root.exists():
        return []
    runs: list[RunRecord] = []
    for path in sorted(root.glob("*/*/seed*/best.pt")):
        recipe, model, seed_dir = path.relative_to(root).parts[:3]
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        runs.append(
            RunRecord(
                task="sacrifice",
                model=model,
                recipe=recipe,
                seed=_seed_from(seed_dir),
                split="val",
                metrics=checkpoint["metrics"],
            )
        )
    return runs


def aggregate(runs: Sequence[RunRecord]) -> list[dict]:
    groups: dict[tuple[str, str, str, str], list[RunRecord]] = {}
    for run in runs:
        groups.setdefault((run.task, run.recipe, run.model, run.split), []).append(run)
    rows = []
    for (task, recipe, model, split), group in groups.items():
        names = set(group[0].metrics)
        for run in group[1:]:
            names &= set(run.metrics)
        metrics = {}
        for name in sorted(names):
            values = np.array([run.metrics[name] for run in group], dtype=np.float64)
            std = float(values.std(ddof=1)) if values.size > 1 else 0.0
            metrics[name] = {"mean": float(values.mean()), "std": std}
        rows.append(
            {
                "task": task,
                "recipe": recipe,
                "model": model,
                "split": split,
                "seeds": len(group),
                "metrics": metrics,
            }
        )
    return sorted(
        rows, key=lambda row: (row["task"], row["recipe"], row["model"], row["split"])
    )


def render_markdown(
    rows: Sequence[dict], metrics: Sequence[str] = ("r2", "auroc_decisive", "mse")
) -> str:
    header = ["task", "recipe", "model", "split", "seeds", *metrics]
    lines = ["| " + " | ".join(header) + " |"]
    lines.append("|" + "---|" * len(header))
    for row in rows:
        cells = [
            row["task"],
            row["recipe"],
            row["model"],
            row["split"],
            str(row["seeds"]),
        ]
        for name in metrics:
            value = row["metrics"].get(name)
            if value is None:
                cells.append("—")
            else:
                cells.append(f"{value['mean']:.3f} ± {value['std']:.3f}")
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def _bucket(values: np.ndarray, low: int, high: int) -> np.ndarray:
    return (values >= low) & (values <= high)


def score_slices(
    outputs: dict[str, np.ndarray],
) -> dict[str, dict[str, float]]:
    """Metrics per ply bucket, |material| bucket, and source family."""
    predictions = np.asarray(outputs["predictions"], dtype=np.float64)
    targets = np.asarray(outputs["targets"], dtype=np.float64)
    plies = np.asarray(outputs["plies"], dtype=np.int64)
    materials = np.abs(np.asarray(outputs["materials"], dtype=np.int64))
    sources = np.asarray(outputs["sources"])
    slices: dict[str, dict[str, float]] = {}
    for low, high, name in PLY_BUCKETS:
        mask = _bucket(plies, low, high)
        if mask.any():
            slices[name] = score_metrics(predictions[mask], targets[mask])
    mask = plies > 60
    if mask.any():
        slices["ply:61+"] = score_metrics(predictions[mask], targets[mask])
    for low, high, name in MATERIAL_BUCKETS:
        mask = _bucket(materials, low, high)
        if mask.any():
            slices[name] = score_metrics(predictions[mask], targets[mask])
    for source in sorted(set(sources.tolist())):
        mask = sources == source
        if mask.any():
            slices[f"source:{source}"] = score_metrics(predictions[mask], targets[mask])
    return slices


def material_baseline(
    outputs: dict[str, np.ndarray], seed: int = 0
) -> dict[str, float]:
    """Affine material-only reference fitted on 80% and scored on 20%."""
    materials = np.asarray(outputs["materials"], dtype=np.float64)
    targets = np.asarray(outputs["targets"], dtype=np.float64)
    order = np.random.default_rng(seed).permutation(materials.size)
    split = int(0.8 * materials.size)
    train, test = order[:split], order[split:]
    design = np.stack([materials[train], np.ones(train.size)], axis=1)
    weights, *_ = np.linalg.lstsq(design, targets[train], rcond=None)
    predictions = materials[test] * weights[0] + weights[1]
    return score_metrics(predictions, targets[test])


def training_histories(root: str | Path) -> dict[tuple[str, str], list[list[dict]]]:
    root = Path(root)
    histories: dict[tuple[str, str], list[list[dict]]] = {}
    if not root.exists():
        return histories
    for path in sorted(root.glob("*/*/seed*/metrics.json")):
        recipe, model, _ = path.relative_to(root).parts[:3]
        histories.setdefault((recipe, model), []).append(
            _load_json(path).get("history") or []
        )
    return histories


def plot_training_curves(
    histories: dict[tuple[str, str], list[list[dict]]], out_path: str | Path
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(figsize=(6, 4))
    for (recipe, model), runs in sorted(histories.items()):
        if not runs:
            continue
        epochs = np.arange(1, max(len(run) for run in runs) + 1)
        values = np.full((len(runs), epochs.size), np.nan)
        for row, run in enumerate(runs):
            values[row, : len(run)] = [entry["val_loss"] for entry in run]
        axes.plot(
            epochs,
            np.nanmean(values, axis=0),
            marker="o",
            label=f"{model} ({recipe})",
        )
    axes.set_xlabel("epoch")
    axes.set_ylabel("validation MSE")
    axes.set_title("Position score probe: validation loss")
    axes.legend(fontsize=8)
    axes.grid(alpha=0.3)
    figure.tight_layout()
    figure.savefig(out_path, dpi=150)
    plt.close(figure)
    return


def _prediction_files(eval_root: str | Path) -> dict[tuple[str, str], list[Path]]:
    root = Path(eval_root)
    files: dict[tuple[str, str], list[Path]] = {}
    if not root.exists():
        return files
    for path in sorted(root.glob("*/*/seed*/predictions.npz")):
        recipe, model, _ = path.relative_to(root).parts[:3]
        files.setdefault((recipe, model), []).append(path)
    return files


def plot_calibration(eval_root: str | Path, out_path: str | Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(figsize=(5, 5))
    axes.plot([-1, 1], [-1, 1], linestyle="--", color="grey", label="ideal")
    for (recipe, model), paths in sorted(_prediction_files(eval_root).items()):
        predictions = np.concatenate([np.load(path)["predictions"] for path in paths])
        targets = np.concatenate([np.load(path)["targets"] for path in paths])
        bins = calibration_bins(predictions, targets, n_bins=10)
        mask = bins["counts"] > 0
        axes.plot(
            bins["prediction_mean"][mask],
            bins["target_mean"][mask],
            marker="o",
            label=f"{model} ({recipe})",
        )
    axes.set_xlabel("mean predicted score")
    axes.set_ylabel("mean realised score")
    axes.set_title("Held-out calibration")
    axes.legend(fontsize=8)
    axes.grid(alpha=0.3)
    figure.tight_layout()
    figure.savefig(out_path, dpi=150)
    plt.close(figure)
    return


def collect_eval_slices(eval_root: str | Path) -> dict[str, dict[str, float]]:
    """Pool predictions per (recipe, model) and slice the pooled set."""
    slices: dict[str, dict[str, float]] = {}
    for (recipe, model), paths in sorted(_prediction_files(eval_root).items()):
        outputs = {
            key: np.concatenate([np.load(path)[key] for path in paths])
            for key in ("predictions", "targets", "plies", "materials", "sources")
        }
        slices[f"{model} ({recipe})"] = score_slices(outputs)
    return slices


def collect_material_baselines(
    eval_root: str | Path,
) -> dict[str, dict[str, float]]:
    baselines: dict[str, dict[str, float]] = {}
    for (recipe, model), paths in sorted(_prediction_files(eval_root).items()):
        outputs = {
            key: np.concatenate([np.load(path)[key] for path in paths])
            for key in ("predictions", "targets", "plies", "materials", "sources")
        }
        baselines[f"{model} ({recipe})"] = material_baseline(outputs)
    return baselines


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize downstream probe runs")
    parser.add_argument("--training-root", default="data/models/downstream/score")
    parser.add_argument("--eval-root", default="data/eval/downstream/score")
    parser.add_argument("--sacrifice-root", default="data/models/downstream/sacrifice")
    parser.add_argument("--out-dir", default="data/eval/downstream/summary")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    runs = collect_score_runs(args.training_root, args.eval_root)
    runs.extend(collect_sacrifice_runs(args.sacrifice_root))
    rows = aggregate(runs)

    score_metrics_names = ("r2", "auroc_decisive", "mae")
    sacrifice_metrics_names = ("pr_auc", "auroc", "f1")
    score_rows = [row for row in rows if row["task"] == "score"]
    sacrifice_rows = [row for row in rows if row["task"] == "sacrifice"]

    markdown = "# Downstream probe summary\n\n"
    markdown += "## Position score regression\n\n"
    markdown += render_markdown(score_rows, score_metrics_names)
    markdown += "\n## Sacrifice classification\n\n"
    markdown += render_markdown(sacrifice_rows, sacrifice_metrics_names)
    (out_dir / "summary.md").write_text(markdown)
    (out_dir / "summary.json").write_text(
        json.dumps(
            {
                "rows": rows,
                "slices": collect_eval_slices(args.eval_root),
                "material_baseline": collect_material_baselines(args.eval_root),
            },
            indent=2,
        )
        + "\n"
    )

    plot_calibration(args.eval_root, out_dir / "calibration.png")
    plot_training_curves(
        training_histories(args.training_root), out_dir / "training_curves.png"
    )
    print(markdown)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
