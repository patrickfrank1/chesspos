from __future__ import annotations

import numpy as np
import pytest

from bdd import given, then, when
from src.training.score_metrics import (
    calibration_bins,
    expected_calibration_error,
    score_metrics,
)


class TestScoreMetrics:
    def test_perfect_predictions_have_no_error(self):
        with given("predictions that exactly match the realised scores"):
            predictions = np.array([1.0, -1.0, 0.0, 1.0], dtype=np.float64)
            targets = predictions.copy()

        with when("the metrics are computed"):
            metrics = score_metrics(predictions, targets)

        with then("the error is zero and the ranking is perfect"):
            assert metrics["mse"] == 0.0
            assert metrics["mae"] == 0.0
            assert metrics["r2"] == 1.0
            assert metrics["pearson"] == 1.0
            assert metrics["spearman"] == 1.0
            assert metrics["auroc_decisive"] == 1.0
            assert metrics["n"] == 4
        return

    def test_constant_zero_prediction_has_no_explained_variance(self):
        with given("equal numbers of white and black wins and a constant zero model"):
            predictions = np.zeros(4, dtype=np.float64)
            targets = np.array([1.0, 1.0, -1.0, -1.0], dtype=np.float64)

        with when("the metrics are computed"):
            metrics = score_metrics(predictions, targets)

        with then("MSE is the target variance, R² is zero, and AUC is chance"):
            assert metrics["mse"] == 1.0
            assert metrics["mae"] == 1.0
            assert metrics["r2"] == 0.0
            assert metrics["pearson"] == 0.0
            assert metrics["spearman"] == 0.0
            assert metrics["auroc_decisive"] == 0.5
        return

    def test_auc_ignores_drawn_games(self):
        with given("predictions where only the decisive games matter"):
            predictions = np.array([1.0, 0.5, -1.0, 0.9], dtype=np.float64)
            targets = np.array([1.0, 0.0, -1.0, -1.0], dtype=np.float64)

        with when("the metrics are computed"):
            metrics = score_metrics(predictions, targets)

        with then("both black wins are ranked below the white win"):
            assert metrics["auroc_decisive"] == 1.0
        return

    def test_auc_is_chance_without_both_decisive_classes(self):
        with given("only drawn games"):
            predictions = np.array([0.1, 0.2], dtype=np.float64)
            targets = np.array([0.0, 0.0], dtype=np.float64)

        with when("the metrics are computed"):
            metrics = score_metrics(predictions, targets)

        with then("the decisive AUC falls back to 0.5"):
            assert metrics["auroc_decisive"] == 0.5
        return

    def test_rejects_mismatched_or_empty_inputs(self):
        with given("mismatched and empty arrays"):
            cases = [
                (np.zeros(3), np.zeros(2)),
                (np.zeros(0), np.zeros(0)),
            ]

        with when("the metrics are computed"):
            errors = []
            for predictions, targets in cases:
                try:
                    score_metrics(predictions, targets)
                except ValueError:
                    errors.append(True)

        with then("every invalid input raises ValueError"):
            assert errors == [True, True]
        return


class TestCalibration:
    def test_bins_average_predictions_and_targets(self):
        with given("two predictions per bin at the extremes"):
            predictions = np.array([-1.0, -0.8, 0.8, 1.0], dtype=np.float64)
            targets = np.array([-1.0, -1.0, 1.0, 1.0], dtype=np.float64)

        with when("two calibration bins are computed"):
            bins = calibration_bins(predictions, targets, n_bins=2)

        with then("each bin reports its mean prediction and realised score"):
            assert bins["edges"].tolist() == [-1.0, 0.0, 1.0]
            assert bins["counts"].tolist() == [2, 2]
            np.testing.assert_allclose(bins["prediction_mean"], [-0.9, 0.9])
            np.testing.assert_allclose(bins["target_mean"], [-1.0, 1.0])
        return

    def test_empty_bins_are_marked(self):
        with given("predictions that only land in the middle bins"):
            predictions = np.array([-0.1, 0.1], dtype=np.float64)
            targets = np.array([0.0, 0.0], dtype=np.float64)

        with when("ten bins are computed"):
            bins = calibration_bins(predictions, targets, n_bins=10)

        with then("bins without samples have zero count and NaN means"):
            assert bins["counts"].sum() == 2
            assert np.isnan(bins["prediction_mean"][0])
        return

    def test_expected_calibration_error_is_zero_when_perfect(self):
        with given("perfectly calibrated extreme predictions"):
            predictions = np.array([-1.0, 1.0], dtype=np.float64)
            targets = predictions.copy()

        with when("the calibration error is computed"):
            error = expected_calibration_error(predictions, targets, n_bins=10)

        with then("the error is zero"):
            assert error == pytest.approx(0.0, abs=1e-12)
        return

    def test_expected_calibration_error_counts_absolute_gap(self):
        with given("predictions off by 0.1 in both extremes"):
            predictions = np.array([-0.9, 0.9], dtype=np.float64)
            targets = np.array([-1.0, 1.0], dtype=np.float64)

        with when("the calibration error is computed with two bins"):
            error = expected_calibration_error(predictions, targets, n_bins=2)

        with then("the error equals the average absolute gap"):
            assert error == pytest.approx(0.1)
        return

    def test_rejects_an_invalid_bin_count(self):
        with given("zero bins"):
            predictions = np.array([0.0], dtype=np.float64)
            targets = np.array([0.0], dtype=np.float64)

        with when("the bins are computed"):
            error = None
            try:
                calibration_bins(predictions, targets, n_bins=0)
            except ValueError as exc:
                error = exc

        with then("a ValueError is raised"):
            assert isinstance(error, ValueError)
        return
