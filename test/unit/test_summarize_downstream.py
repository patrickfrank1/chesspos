from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from bdd import given, then, when
from scripts.eval.summarize_downstream import (
    RunRecord,
    aggregate,
    collect_sacrifice_runs,
    collect_score_runs,
    material_baseline,
    plot_calibration,
    plot_training_curves,
    render_markdown,
    score_slices,
    training_histories,
)


def write_score_metrics(
    path: Path,
    history: list[dict],
    feature: str = "native",
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "feature": feature,
                "unfreeze_last_block": False,
                "window": 1,
                "include_state_features": False,
                "loss": "mse",
                "amp": "bf16",
                "checkpoint": "encoder.pt",
                "train_games": 100,
                "val_games": 10,
                "history": history,
            }
        )
        + "\n"
    )
    return


def write_eval_metrics(path: Path, metrics: dict[str, float]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "checkpoint": "probe.pt",
                "feature": "native",
                "window": 1,
                "include_state_features": False,
                "source": "patrickfrank1/chess-positions",
                "split": "test",
                "games": 10,
                "positions": 100,
                "metrics": metrics,
            }
        )
        + "\n"
    )
    return


def write_predictions(path: Path, predictions: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    targets = np.linspace(-1.0, 1.0, predictions.size).astype(np.float32)
    np.savez_compressed(
        path,
        predictions=predictions.astype(np.float32),
        targets=targets,
        plies=np.ones(predictions.size, dtype=np.int16),
        materials=np.zeros(predictions.size, dtype=np.int16),
        game_ids=np.array([f"g{i}" for i in range(predictions.size)]),
        sources=np.array(["lichess_elite_2019-01"] * predictions.size),
    )
    return


def write_sacrifice_checkpoint(path: Path, metrics: dict[str, float]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "encoder_config": {},
            "encoder_state": {},
            "head_config": {},
            "head_state": {},
            "feature": "native",
            "planner_state": {},
            "epoch": 1,
            "metrics": metrics,
        },
        path,
    )
    return


class TestCollectScoreRuns:
    def test_reads_the_best_validation_metrics_from_training_artifacts(self, tmp_path):
        with given("two seeds with two history entries each"):
            for seed, value in ((17, 0.30), (23, 0.40)):
                write_score_metrics(
                    tmp_path
                    / "train"
                    / "native"
                    / "run_20m_lazy"
                    / f"seed{seed}"
                    / "metrics.json",
                    history=[
                        {"epoch": 1, "train_loss": 0.9, "val_loss": 0.8, "metrics": {}},
                        {
                            "epoch": 2,
                            "train_loss": 0.8,
                            "val_loss": value,
                            "metrics": {"r2": value},
                        },
                    ],
                )
            write_eval_metrics(
                tmp_path
                / "eval"
                / "native"
                / "run_20m_lazy"
                / "seed17"
                / "metrics.json",
                {"r2": 0.5},
            )

        with when("the runs are collected"):
            runs = collect_score_runs(tmp_path / "train", tmp_path / "eval")

        with then("validation and test rows use the right metrics"):
            validation = [run for run in runs if run.split == "val"]
            test = [run for run in runs if run.split == "test"]
            assert len(validation) == 2
            assert len(test) == 1
            assert validation[0].metrics["r2"] in (0.3, 0.4)
            assert test[0].metrics["r2"] == 0.5
            assert {run.seed for run in validation} == {17, 23}
            assert all(run.task == "score" for run in runs)
        return


class TestCollectSacrificeRuns:
    def test_reads_metrics_from_the_best_checkpoint(self, tmp_path):
        with given("sacrifice checkpoints for two models"):
            write_sacrifice_checkpoint(
                tmp_path / "native" / "run_20m_lazy" / "seed17" / "best.pt",
                {"pr_auc": 0.21, "auroc": 0.64, "f1": 0.31},
            )
            write_sacrifice_checkpoint(
                tmp_path / "native" / "run_6m_isolated" / "seed17" / "best.pt",
                {"pr_auc": 0.18, "auroc": 0.61, "f1": 0.28},
            )

        with when("the runs are collected"):
            runs = collect_sacrifice_runs(tmp_path)

        with then("one val row per checkpoint is returned"):
            assert len(runs) == 2
            assert all(run.task == "sacrifice" and run.split == "val" for run in runs)
            assert {run.model for run in runs} == {
                "run_20m_lazy",
                "run_6m_isolated",
            }
            assert runs[0].metrics["pr_auc"] in (0.21, 0.18)
        return

    def test_missing_roots_are_ignored(self, tmp_path):
        with given("an empty root"):
            root = tmp_path / "missing"

        with when("the runs are collected"):
            runs = collect_sacrifice_runs(root)

        with then("the result is empty"):
            assert runs == []
        return


class TestAggregate:
    def test_groups_by_task_recipe_model_and_split(self):
        with given("three runs with known metric values"):
            runs = [
                RunRecord("score", "m", "native", 17, "val", {"r2": 0.2}),
                RunRecord("score", "m", "native", 23, "val", {"r2": 0.4}),
                RunRecord("score", "m", "full", 17, "test", {"r2": 0.6}),
            ]

        with when("the runs are aggregated"):
            rows = aggregate(runs)

        with then("mean and standard deviation are computed per group"):
            native = [row for row in rows if row["recipe"] == "native"][0]
            assert native["seeds"] == 2
            assert native["metrics"]["r2"]["mean"] == pytest.approx(0.3)
            assert native["metrics"]["r2"]["std"] == pytest.approx(0.1 * np.sqrt(2))
            full = [row for row in rows if row["recipe"] == "full"][0]
            assert full["metrics"]["r2"]["std"] == 0.0
        return

    def test_single_seed_has_zero_std(self):
        with given("one run"):
            runs = [RunRecord("score", "m", "native", 17, "val", {"r2": 0.2})]

        with when("the runs are aggregated"):
            rows = aggregate(runs)

        with then("the standard deviation is zero"):
            assert rows[0]["metrics"]["r2"]["std"] == 0.0
        return


class TestRenderMarkdown:
    def test_renders_one_row_per_group_with_mean_and_std(self):
        with given("an aggregated score run"):
            rows = aggregate(
                [
                    RunRecord(
                        "score", "m", "native", 17, "test", {"r2": 0.2, "mse": 0.8}
                    ),
                    RunRecord(
                        "score", "m", "native", 23, "test", {"r2": 0.4, "mse": 0.6}
                    ),
                ]
            )

        with when("the table is rendered"):
            markdown = render_markdown(rows, metrics=("r2", "mse"))

        with then("the row names both metrics with mean and std"):
            assert "| score | native | m | test | 2 |" in markdown
            assert "0.300" in markdown
            assert "0.700" in markdown
        return


class TestPlots:
    def test_calibration_plot_is_written(self, tmp_path):
        with given("prediction files for one model"):
            write_predictions(
                tmp_path / "native" / "m" / "seed17" / "predictions.npz",
                np.linspace(-1.0, 1.0, 200),
            )
            out = tmp_path / "calibration.png"

        with when("the calibration plot is rendered"):
            plot_calibration(tmp_path, out)

        with then("a non-empty PNG exists"):
            assert out.exists()
            assert out.stat().st_size > 1000
        return

    def test_training_curve_plot_is_written(self, tmp_path):
        with given("training histories for two models"):
            write_score_metrics(
                tmp_path / "train" / "native" / "a" / "seed17" / "metrics.json",
                history=[
                    {"epoch": 1, "train_loss": 0.9, "val_loss": 0.85, "metrics": {}},
                    {"epoch": 2, "train_loss": 0.8, "val_loss": 0.75, "metrics": {}},
                ],
            )
            write_score_metrics(
                tmp_path / "train" / "native" / "b" / "seed17" / "metrics.json",
                history=[
                    {"epoch": 1, "train_loss": 0.95, "val_loss": 0.9, "metrics": {}},
                    {"epoch": 2, "train_loss": 0.85, "val_loss": 0.8, "metrics": {}},
                ],
            )
            out = tmp_path / "curves.png"

        with when("the curve plot is rendered"):
            plot_training_curves(training_histories(tmp_path / "train"), out)

        with then("a non-empty PNG exists"):
            assert out.exists()
            assert out.stat().st_size > 1000
        return


class TestScoreSlices:
    def test_groups_predictions_by_ply_material_and_source(self):
        with given("predictions spanning ply, material, and source buckets"):
            outputs = {
                "predictions": np.array([1.0, -1.0, 0.5, -0.5], dtype=np.float32),
                "targets": np.array([1.0, -1.0, 0.0, -1.0], dtype=np.float32),
                "plies": np.array([5, 20, 45, 80], dtype=np.int16),
                "materials": np.array([0, 2, 5, 9], dtype=np.int16),
                "sources": np.array(["elite", "standard", "elite", "standard"]),
            }

        with when("the slices are computed"):
            slices = score_slices(outputs)

        with then("each bucket reports its own metrics and count"):
            assert slices["ply:1-10"]["n"] == 1.0
            assert slices["ply:11-30"]["n"] == 1.0
            assert slices["ply:31-60"]["n"] == 1.0
            assert slices["ply:61+"]["n"] == 1.0
            assert slices["material:0"]["n"] == 1.0
            assert slices["material:1-3"]["n"] == 1.0
            assert slices["material:4-6"]["n"] == 1.0
            assert slices["material:7+"]["n"] == 1.0
            assert slices["source:elite"]["n"] == 2.0
            assert slices["source:standard"]["n"] == 2.0
        return


class TestMaterialBaseline:
    def test_noiseless_material_recovers_the_outcome(self):
        with given("scores that are a deterministic function of material"):
            materials = np.arange(-9, 10, dtype=np.int16)
            targets = np.clip(materials / 9.0, -1.0, 1.0).astype(np.float32)
            outputs = {
                "predictions": np.zeros(materials.size, dtype=np.float32),
                "targets": targets,
                "plies": np.ones(materials.size, dtype=np.int16),
                "materials": materials,
                "sources": np.array(["s"] * materials.size),
            }

        with when("the material baseline is fitted"):
            baseline = material_baseline(outputs)

        with then("the baseline ranks every decisive game correctly"):
            assert baseline["auroc_decisive"] == pytest.approx(1.0)
            assert baseline["r2"] > 0.99
        return
