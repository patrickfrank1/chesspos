from __future__ import annotations

import numpy as np

from bdd import given, then, when
from src.dataset.puzzle_variants import (
    EpochPlan,
    PuzzleRecord,
    PuzzleVariantDataset,
    RollingVariantPlanner,
    build_epoch_plan,
    collate_variants,
    iter_variants,
    pack_segments,
    split_for_puzzle,
    variant_count,
)
from src.dataset.token_stream import CLS, PAD, SEP

SEGMENT_LEN = 3


def make_record(puzzle_id: str, n_positions: int, label: int) -> PuzzleRecord:
    segments = [
        np.full(SEGMENT_LEN, 10 + index, dtype=np.int16) for index in range(n_positions)
    ]
    return PuzzleRecord(puzzle_id=puzzle_id, segments=segments, label=label)


def make_records() -> list[PuzzleRecord]:
    return [
        make_record("p0", 3, 1),
        make_record("p1", 2, 1),
        make_record("p2", 4, 1),
        make_record("n0", 1, 0),
        make_record("n1", 3, 0),
        make_record("n2", 2, 0),
        make_record("n3", 1, 0),
    ]


class TestVariantSpace:
    def test_variant_count_and_iteration_cover_every_contiguous_subsequence(self):
        with given("a puzzle with three positions"):
            n_positions = 3

        with when("the variants are enumerated"):
            variants = list(iter_variants(n_positions))

        with then("all contiguous subsequences appear exactly once"):
            assert variant_count(n_positions) == 6
            assert len(variants) == 6
            assert len(set(variants)) == 6
            assert (0, 0) in variants
            assert (0, 2) in variants
            assert (2, 2) in variants
            assert all(0 <= start <= end < n_positions for start, end in variants)
        return


class TestEpochPlan:
    def test_plan_is_balanced_unique_and_in_bounds(self):
        with given("seven records with four negatives and three positives"):
            records = make_records()

        with when("a 50/50 epoch plan is built"):
            plan = build_epoch_plan(records, positive_fraction=0.5, seed=7)

        with then("the plan uses each negative once and unique positive variants"):
            assert len(plan.label) == 8
            assert int(plan.label.sum()) == 4

            triples = set(
                zip(
                    plan.puzzle_index.tolist(),
                    plan.start.tolist(),
                    plan.end.tolist(),
                )
            )
            assert len(triples) == len(plan.label)

            for index, start, end, label in zip(
                plan.puzzle_index,
                plan.start,
                plan.end,
                plan.label,
            ):
                record = records[int(index)]
                assert record.label == int(label)
                assert 0 <= start <= end < len(record.segments)

            negative_indices = {
                index for index, record in enumerate(records) if record.label == 0
            }
            used_negatives = set(plan.puzzle_index[plan.label == 0].tolist())
            assert used_negatives == negative_indices
        return

    def test_plan_is_deterministic_for_a_seed(self):
        with given("the same records and epoch seed"):
            records = make_records()

        with when("the plan is built twice"):
            first = build_epoch_plan(records, positive_fraction=0.5, seed=11)
            second = build_epoch_plan(records, positive_fraction=0.5, seed=11)

        with then("the configurations match exactly"):
            np.testing.assert_array_equal(first.puzzle_index, second.puzzle_index)
            np.testing.assert_array_equal(first.start, second.start)
            np.testing.assert_array_equal(first.end, second.end)
            np.testing.assert_array_equal(first.label, second.label)
        return

    def test_plan_uses_a_fresh_sample_for_a_different_seed(self):
        with given("the same records and two different epoch seeds"):
            records = make_records()

        with when("the plans are built"):
            first = build_epoch_plan(records, positive_fraction=0.5, seed=1)
            second = build_epoch_plan(records, positive_fraction=0.5, seed=2)

        with then("the sampled configurations differ"):
            first_triples = set(zip(first.puzzle_index, first.start, first.end))
            second_triples = set(zip(second.puzzle_index, second.start, second.end))
            assert first_triples != second_triples
        return

    def test_invalid_positive_fraction_is_rejected(self):
        with given("an invalid positive fraction"):
            records = make_records()

        with when("the plan is built"):
            try:
                build_epoch_plan(records, positive_fraction=0.0, seed=1)
            except ValueError:
                raised = True
            else:
                raised = False

        with then("the invalid fraction is rejected"):
            assert raised
        return


class TestRollingVariantPlanner:
    def test_positive_variants_do_not_repeat_until_the_space_is_exhausted(self):
        with given("two positives with 3 and 6 variants and two negatives"):
            records = [
                make_record("p0", 2, 1),
                make_record("p1", 3, 1),
                make_record("n0", 2, 0),
                make_record("n1", 2, 0),
            ]
            planner = RollingVariantPlanner(records, positive_fraction=0.5, seed=5)
            seen: set[tuple[int, int, int]] = set()

        with when("four epochs are drawn (8 of 9 positive variants)"):
            for _ in range(4):
                plan = planner.next_epoch()
                for index, start, end, label in zip(
                    plan.puzzle_index, plan.start, plan.end, plan.label
                ):
                    if label == 1:
                        key = (int(index), int(start), int(end))
                        assert key not in seen
                        seen.add(key)

        with then("no positive configuration repeats before exhaustion"):
            assert len(seen) == 8
        return

    def test_positive_variants_repeat_only_after_exhaustion(self):
        with given("the same planner and nine total positive variants"):
            records = [
                make_record("p0", 2, 1),
                make_record("p1", 3, 1),
                make_record("n0", 2, 0),
                make_record("n1", 2, 0),
            ]
            planner = RollingVariantPlanner(records, positive_fraction=0.5, seed=5)
            seen: set[tuple[int, int, int]] = set()

        with when("five epochs are drawn (10 positive draws)"):
            for _ in range(5):
                plan = planner.next_epoch()
                for index, start, end, label in zip(
                    plan.puzzle_index, plan.start, plan.end, plan.label
                ):
                    if label == 1:
                        seen.add((int(index), int(start), int(end)))

        with then("every variant is used and the extra draw is a repeat"):
            assert len(seen) == 9
        return

    def test_negative_variants_roll_before_repeating(self):
        with given("one positive and two negatives with three variants each"):
            records = [
                make_record("p0", 2, 1),
                make_record("n0", 2, 0),
                make_record("n1", 2, 0),
            ]
            planner = RollingVariantPlanner(records, positive_fraction=0.5, seed=5)
            seen: set[tuple[int, int, int]] = set()

        with when("three epochs are drawn"):
            for _ in range(3):
                plan = planner.next_epoch()
                for index, start, end, label in zip(
                    plan.puzzle_index, plan.start, plan.end, plan.label
                ):
                    if label == 0:
                        key = (int(index), int(start), int(end))
                        assert key not in seen
                        seen.add(key)

        with then("all six negative configurations are used exactly once"):
            assert len(seen) == 6
        return

    def test_state_dict_roundtrip_continues_the_same_cycle(self):
        with given("a planner advanced by one epoch"):
            records = [
                make_record("p0", 2, 1),
                make_record("p1", 3, 1),
                make_record("n0", 2, 0),
                make_record("n1", 2, 0),
            ]
            first = RollingVariantPlanner(records, positive_fraction=0.5, seed=5)
            first.next_epoch()
            state = first.state_dict()

        with when("the state is restored into a fresh planner"):
            second = RollingVariantPlanner(records, positive_fraction=0.5, seed=5)
            second.load_state_dict(state)
            expected = first.next_epoch()
            actual = second.next_epoch()

        with then("both planners produce the same next epoch"):
            np.testing.assert_array_equal(actual.puzzle_index, expected.puzzle_index)
            np.testing.assert_array_equal(actual.start, expected.start)
            np.testing.assert_array_equal(actual.end, expected.end)
            np.testing.assert_array_equal(actual.label, expected.label)
        return


class TestPackSegments:
    def test_pack_adds_cls_and_sep_and_computes_segment_ids(self):
        with given("two position segments"):
            segments = [
                np.array([10, 11, 12], dtype=np.int16),
                np.array([13, 14, 15], dtype=np.int16),
            ]

        with when("the segments are packed"):
            tokens, segment_ids = pack_segments(segments)

        with then("the stream is CLS-prefixed and SEP-separated"):
            np.testing.assert_array_equal(
                tokens,
                np.array([CLS, 10, 11, 12, SEP, 13, 14, 15, SEP], dtype=np.int16),
            )
            np.testing.assert_array_equal(
                segment_ids,
                np.array([0, 0, 0, 0, 1, 1, 1, 1, 2], dtype=np.int64),
            )
        return


class TestPuzzleVariantDataset:
    def test_item_is_a_contiguous_subsequence_with_the_sequence_label(self):
        with given("a positive record and a plan selecting its last two positions"):
            record = make_record("puzzle", 3, 1)
            plan = EpochPlan(
                puzzle_index=np.array([0], dtype=np.int32),
                start=np.array([1], dtype=np.int16),
                end=np.array([2], dtype=np.int16),
                label=np.array([1], dtype=np.int8),
            )
            dataset = PuzzleVariantDataset([record], plan)

        with when("the item is read"):
            item = dataset[0]

        with then("the tokens contain only the selected subsequence"):
            np.testing.assert_array_equal(
                item["tokens"],
                np.array([CLS, 11, 11, 11, SEP, 12, 12, 12, SEP], dtype=np.int16),
            )
            np.testing.assert_array_equal(
                item["segment_ids"],
                np.array([0, 0, 0, 0, 1, 1, 1, 1, 2], dtype=np.int64),
            )
            assert item["label"] == 1
            assert item["puzzle_id"] == "puzzle"
        return


class TestCollateVariants:
    def test_collate_pads_tokens_and_preserves_labels(self):
        with given("two samples with different sequence lengths"):
            batch = [
                {
                    "tokens": np.array([CLS, 1, 1, 1, SEP], dtype=np.int16),
                    "segment_ids": np.array([0, 0, 0, 0, 1], dtype=np.int64),
                    "label": 1,
                    "puzzle_id": "a",
                },
                {
                    "tokens": np.array(
                        [CLS, 2, 2, 2, SEP, 3, 3, 3, SEP], dtype=np.int16
                    ),
                    "segment_ids": np.array(
                        [0, 0, 0, 0, 1, 1, 1, 1, 2], dtype=np.int64
                    ),
                    "label": 0,
                    "puzzle_id": "b",
                },
            ]

        with when("the samples are collated"):
            collated = collate_variants(batch)

        with then("tokens are padded and labels stay aligned"):
            assert collated["tokens"].shape == (2, 9)
            assert collated["padding_mask"].tolist() == [
                [True, True, True, True, True, False, False, False, False],
                [True, True, True, True, True, True, True, True, True],
            ]
            assert collated["tokens"][0, 5].item() == PAD
            assert collated["labels"].tolist() == [1.0, 0.0]
            assert collated["puzzle_ids"] == ["a", "b"]
        return


class TestSplit:
    def test_split_is_deterministic(self):
        with given("the same puzzle id"):
            puzzle_id = "abc123"

        with when("the split is computed twice"):
            first = split_for_puzzle(puzzle_id, val_ratio=0.1)
            second = split_for_puzzle(puzzle_id, val_ratio=0.1)

        with then("the split is stable"):
            assert first == second
        return

    def test_split_ratio_is_close_to_the_requested_fraction(self):
        with given("many puzzle ids and a 20% validation ratio"):
            ids = [f"puzzle-{index}" for index in range(20_000)]

        with when("the splits are computed"):
            splits = [split_for_puzzle(puzzle_id, val_ratio=0.2) for puzzle_id in ids]

        with then("roughly 20% land in validation"):
            ratio = splits.count("val") / len(splits)
            assert 0.18 < ratio < 0.22
        return
