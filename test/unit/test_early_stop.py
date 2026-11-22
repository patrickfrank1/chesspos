from __future__ import annotations

import pytest

from bdd import given, then, when
from src.training.early_stop import auroc_standard_error, is_at_chance


class TestAurocStandardError:
    def test_perfect_separation_has_a_small_error(self):
        with given("a perfect AUC on a large validation set"):
            auc, n_pos, n_neg = 1.0, 2000, 2000

        with when("the standard error is computed"):
            error = auroc_standard_error(auc, n_pos, n_neg)

        with then("the error vanishes"):
            assert error == pytest.approx(0.0, abs=1e-9)
        return

    def test_chance_auc_on_a_large_set_has_a_small_error(self):
        with given("AUC 0.5 with 2000 positives and 2000 negatives"):
            error = auroc_standard_error(0.5, 2000, 2000)

        with when("the error is inspected"):
            pass

        with then("it is well below the 0.01 margin"):
            assert 0.0 < error < 0.02
        return

    def test_small_samples_have_a_larger_error(self):
        with given("the same AUC on a tiny and a large validation set"):
            small = auroc_standard_error(0.6, 10, 10)
            large = auroc_standard_error(0.6, 5000, 5000)

        with when("the errors are compared"):
            pass

        with then("the small set is less certain"):
            assert small > large
        return

    def test_degenerate_class_counts_return_zero(self):
        with given("no positive or no negative examples"):
            none_positive = auroc_standard_error(0.5, 0, 100)
            none_negative = auroc_standard_error(0.5, 100, 0)

        with when("the counts are inspected"):
            pass

        with then("the error is zero and the chance test handles it"):
            assert none_positive == 0.0
            assert none_negative == 0.0
        return


class TestIsAtChance:
    def test_strong_auc_on_a_large_set_is_not_at_chance(self):
        with given("AUC 0.60 with 5000 positives and 5000 negatives"):
            at_chance = is_at_chance(0.60, 5000, 5000)

        with when("the chance test runs"):
            pass

        with then("the result is clearly above chance"):
            assert at_chance is False
        return

    def test_auc_at_chance_on_a_large_set_is_detected(self):
        with given("AUC 0.505 with 5000 positives and 5000 negatives"):
            at_chance = is_at_chance(0.505, 5000, 5000)

        with when("the chance test runs"):
            pass

        with then("the run is flagged as no better than random"):
            assert at_chance is True
        return

    def test_a_modest_auc_on_a_tiny_set_is_not_significant(self):
        with given("AUC 0.60 but only ten examples per class"):
            at_chance = is_at_chance(0.60, 10, 10)

        with when("the chance test runs"):
            pass

        with then("the evidence is insufficient, so it counts as chance"):
            assert at_chance is True
        return

    def test_degenerate_counts_are_treated_as_chance(self):
        with given("no negative examples"):
            at_chance = is_at_chance(0.5, 100, 0)

        with when("the chance test runs"):
            pass

        with then("the conservative answer is chance"):
            assert at_chance is True
        return

    def test_rejects_an_auc_outside_the_unit_interval(self):
        with given("an invalid AUC"):
            error = None

        with when("the chance test runs"):
            try:
                is_at_chance(1.2, 10, 10)
            except ValueError as exc:
                error = exc

        with then("a ValueError is raised"):
            assert isinstance(error, ValueError)
        return
