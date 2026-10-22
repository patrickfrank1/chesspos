from __future__ import annotations

import math

import pytest
import torch

from bdd import given, then, when
from src.training.losses import (
    calibrate_focal_loss,
    focal_loss_with_logits,
    resolve_focal_loss,
)


class TestFocalLoss:
    def test_gamma_zero_equals_half_the_binary_cross_entropy(self):
        with given("arbitrary logits, targets and alpha=0.5"):
            logits = torch.tensor([-3.0, -0.5, 0.0, 0.7, 2.5])
            targets = torch.tensor([0.0, 0.0, 1.0, 1.0, 1.0])

        with when("focal loss with gamma=0 is evaluated"):
            loss = focal_loss_with_logits(logits, targets, gamma=0.0, alpha=0.5)
            bce = torch.nn.functional.binary_cross_entropy_with_logits(logits, targets)

        with then("it equals half of the balanced binary cross entropy"):
            assert loss.item() == pytest.approx(0.5 * bce.item(), rel=1e-6)
        return

    def test_matches_the_hand_computed_value_for_gamma_two(self):
        with given("one positive and one negative logit"):
            logits = torch.tensor([0.0, 2.0])
            targets = torch.tensor([1.0, 0.0])

        with when("focal loss with gamma=2 and alpha=0.5 is evaluated"):
            loss = focal_loss_with_logits(logits, targets, gamma=2.0, alpha=0.5)

        with then("it matches the hand-computed mean focal loss"):
            assert loss.item() == pytest.approx(0.4558412436403485, rel=1e-6)
        return

    def test_alpha_scales_each_class_independently(self):
        with given("one positive and one negative logit at zero"):
            logits = torch.zeros(2)
            targets = torch.tensor([1.0, 0.0])

        with when("focal loss is evaluated per sample with alpha=0.75"):
            losses = focal_loss_with_logits(
                logits, targets, gamma=2.0, alpha=0.75, reduction="none"
            )

        with then("each class is scaled by its alpha weight"):
            per_sample = 0.25 * math.log(2.0)
            assert losses[0].item() == pytest.approx(0.75 * per_sample, rel=1e-6)
            assert losses[1].item() == pytest.approx(0.25 * per_sample, rel=1e-6)
        return

    def test_sum_and_mean_reductions_are_consistent(self):
        with given("a batch of logits and targets"):
            logits = torch.tensor([-2.0, -0.5, 0.3, 1.1])
            targets = torch.tensor([0.0, 0.0, 1.0, 1.0])

        with when("the loss is computed with every reduction"):
            per_sample = focal_loss_with_logits(
                logits, targets, gamma=2.0, alpha=0.5, reduction="none"
            )
            summed = focal_loss_with_logits(
                logits, targets, gamma=2.0, alpha=0.5, reduction="sum"
            )
            mean = focal_loss_with_logits(
                logits, targets, gamma=2.0, alpha=0.5, reduction="mean"
            )

        with then("sum and mean match the per-sample losses"):
            assert summed.item() == pytest.approx(per_sample.sum().item(), rel=1e-6)
            assert mean.item() == pytest.approx(per_sample.mean().item(), rel=1e-6)
        return

    def test_is_finite_for_saturated_logits(self):
        with given("logits far inside the tails of the sigmoid"):
            logits = torch.tensor([-100.0, 100.0])
            targets = torch.tensor([1.0, 0.0])

        with when("focal loss is evaluated"):
            loss = focal_loss_with_logits(logits, targets, gamma=2.0, alpha=0.5)

        with then("the loss is finite"):
            assert torch.isfinite(loss)
        return

    def test_rejects_an_unknown_reduction(self):
        with given("an unsupported reduction name"):
            logits = torch.zeros(1)
            targets = torch.zeros(1)

        with when("focal loss is evaluated"):
            with pytest.raises(ValueError):
                focal_loss_with_logits(logits, targets, reduction="bogus")

        with then("the invalid reduction is rejected"):
            pass
        return


class TestCalibrateFocalLoss:
    def test_balanced_labels_disable_focal_modulation(self):
        with given("an evenly split label distribution"):
            class_counts = {0: 500, 1: 500}

        with when("the focal loss is calibrated"):
            config = calibrate_focal_loss(class_counts)

        with then("alpha is neutral and gamma is zero"):
            assert config.alpha == pytest.approx(0.5)
            assert config.gamma == pytest.approx(0.0)
        return

    def test_alpha_balances_expected_class_contributions(self):
        with given("one positive for three negatives"):
            class_counts = {0: 600, 1: 200}

        with when("the focal loss is calibrated"):
            config = calibrate_focal_loss(class_counts)

        with then("alpha equals the negative prevalence"):
            assert config.alpha == pytest.approx(0.75)
            assert config.gamma == pytest.approx(math.log2(3.0))
        return

    def test_gamma_tracks_imbalance_and_saturates_at_two(self):
        with given("a nine to one imbalance"):
            class_counts = {0: 900, 1: 100}

        with when("the focal loss is calibrated"):
            config = calibrate_focal_loss(class_counts)

        with then("gamma is capped at two"):
            assert config.alpha == pytest.approx(0.9)
            assert config.gamma == pytest.approx(2.0)
        return

    def test_calibrates_the_sacrifice_prevalence(self):
        with given("the sacrifice label distribution from the Lichess dump"):
            class_counts = {0: 5_693_085, 1: 464_256}

        with when("the focal loss is calibrated"):
            config = calibrate_focal_loss(class_counts)

        with then("alpha upweights positives and gamma is saturated"):
            assert config.alpha == pytest.approx(0.9246, abs=1e-4)
            assert config.gamma == pytest.approx(2.0)
        return

    def test_gamma_and_alpha_grow_as_positives_become_rarer(self):
        with given("a sequence of increasingly imbalanced distributions"):
            prevalences = [0.5, 0.4, 0.3, 0.2, 0.1, 0.05]

        with when("each distribution is calibrated"):
            configs = [
                calibrate_focal_loss({0: round((1 - p) * 1000), 1: round(p * 1000)})
                for p in prevalences
            ]

        with then("alpha and gamma are non-decreasing"):
            assert [c.alpha for c in configs] == sorted(c.alpha for c in configs)
            assert [c.gamma for c in configs] == sorted(c.gamma for c in configs)
            assert configs[-1].gamma == pytest.approx(2.0)
        return

    @pytest.mark.parametrize(
        "class_counts",
        [
            {},
            {0: 0, 1: 0},
            {0: 10, 1: 0},
            {0: 0, 1: 10},
            {0: -1, 1: 5},
        ],
    )
    def test_rejects_distributions_without_both_classes(self, class_counts):
        with given("a label distribution that is empty or single-class"):
            counts = class_counts

        with when("the focal loss is calibrated"):
            with pytest.raises(ValueError):
                calibrate_focal_loss(counts)

        with then("the invalid distribution is rejected"):
            pass
        return


class TestResolveFocalLoss:
    def test_bce_needs_no_focal_configuration(self):
        with given("binary cross entropy as the selected loss"):
            class_counts = {0: 9, 1: 1}

        with when("the focal loss configuration is resolved"):
            config = resolve_focal_loss(
                loss="bce", class_counts=class_counts, gamma=None, alpha=None
            )

        with then("no focal configuration is produced"):
            assert config is None
        return

    def test_missing_parameters_are_calibrated_from_the_labels(self):
        with given("focal loss selected without explicit parameters"):
            class_counts = {0: 9, 1: 1}

        with when("the focal loss configuration is resolved"):
            config = resolve_focal_loss(
                loss="focal", class_counts=class_counts, gamma=None, alpha=None
            )

        with then("the calibrated parameters are used"):
            assert config is not None
            assert config.alpha == pytest.approx(0.9)
            assert config.gamma == pytest.approx(2.0)
        return

    def test_explicit_parameters_override_calibration(self):
        with given("focal loss with explicit gamma and alpha"):
            class_counts = {0: 9, 1: 1}

        with when("the focal loss configuration is resolved"):
            config = resolve_focal_loss(
                loss="focal", class_counts=class_counts, gamma=1.5, alpha=0.25
            )

        with then("the explicit values win"):
            assert config is not None
            assert config.gamma == pytest.approx(1.5)
            assert config.alpha == pytest.approx(0.25)
        return

    def test_partial_overrides_keep_the_calibrated_value(self):
        with given("focal loss with an explicit alpha only"):
            class_counts = {0: 9, 1: 1}

        with when("the focal loss configuration is resolved"):
            config = resolve_focal_loss(
                loss="focal", class_counts=class_counts, gamma=None, alpha=0.25
            )

        with then("gamma stays calibrated and alpha is explicit"):
            assert config is not None
            assert config.gamma == pytest.approx(2.0)
            assert config.alpha == pytest.approx(0.25)
        return
