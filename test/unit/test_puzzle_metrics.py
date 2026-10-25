from __future__ import annotations

import pytest

from bdd import given, then, when
from src.training.puzzle_metrics import binary_metrics


class TestBinaryMetrics:
    def test_known_scores_produce_expected_ranking_and_threshold_metrics(self):
        with given("four examples with two positives and overlapping scores"):
            labels = [0, 0, 1, 1]
            scores = [0.1, 0.4, 0.35, 0.8]

        with when("the metrics are computed"):
            metrics = binary_metrics(scores, labels)

        with then("AUROC, average precision and best F1 match the hand calculation"):
            assert metrics["auroc"] == pytest.approx(0.75)
            assert metrics["pr_auc"] == pytest.approx(0.8333333, rel=1e-5)
            assert metrics["f1"] == pytest.approx(0.8)
            assert metrics["precision"] == pytest.approx(2 / 3)
            assert metrics["recall"] == pytest.approx(1.0)
            assert metrics["accuracy"] == pytest.approx(0.75)
            assert metrics["threshold"] == pytest.approx(0.35)
        return

    def test_perfect_separation_scores_one(self):
        with given("perfectly separated scores"):
            labels = [0, 1]
            scores = [0.1, 0.9]

        with when("the metrics are computed"):
            metrics = binary_metrics(scores, labels)

        with then("all ranking and threshold metrics are perfect"):
            assert metrics["auroc"] == pytest.approx(1.0)
            assert metrics["pr_auc"] == pytest.approx(1.0)
            assert metrics["f1"] == pytest.approx(1.0)
        return

    def test_tied_scores_yield_chance_auroc(self):
        with given("two examples with identical scores"):
            labels = [0, 1]
            scores = [0.5, 0.5]

        with when("the metrics are computed"):
            metrics = binary_metrics(scores, labels)

        with then("ranking is at chance and the best threshold predicts both"):
            assert metrics["auroc"] == pytest.approx(0.5)
            assert metrics["pr_auc"] == pytest.approx(0.5)
            assert metrics["f1"] == pytest.approx(2 / 3)
        return
