import numpy as np

from bdd import given, then, when
from src.evaluation.observability.ranking import (
    RankingCase,
    build_temporal_ranking_cases,
    build_theme_ranking_cases,
    chance_metrics,
    evaluate_ranking,
    surrogate_ranking_examples,
    surrogate_ranking_metrics,
    temporal_ranking_examples,
    temporal_ranking_metrics,
    theme_ranking_examples,
    theme_ranking_metrics,
)
from src.evaluation.observability.samples import PositionSample
from src.evaluation.observability.surrogates import SURROGATE_TASKS


def sample(
    sample_id,
    *,
    source="puzzle",
    puzzle_id=None,
    game_id=None,
    ply=None,
    themes=(),
    surrogates=None,
):
    return PositionSample(
        sample_id=sample_id,
        tokens=tuple([4] * 67),
        fen="8/8/8/8/8/8/8/8 w - - 0 1",
        source=source,
        puzzle_id=puzzle_id,
        game_id=game_id,
        ply=ply,
        themes=tuple(themes),
        surrogates=dict(surrogates or {"material": 0}),
    )


def test_evaluate_ranking_perfect_and_worst_ranks():
    with given("one query with a nearest positive and a farthest positive"):
        embeddings = np.array(
            [
                [1.0, 0.0],
                [1.0, 0.01],
                [0.0, 1.0],
                [-1.0, 0.0],
                [0.0, -1.0],
            ],
            dtype=np.float64,
        )
        perfect = RankingCase(query=0, candidates=(1, 2, 3, 4), positive_position=0)
        worst = RankingCase(query=0, candidates=(1, 2, 3, 4), positive_position=2)

    with when("both rankings are evaluated"):
        good = evaluate_ranking(embeddings, [perfect])
        bad = evaluate_ranking(embeddings, [worst])

    with then("the perfect case is top-1 and the worst is last"):
        assert good["top1"] == 1.0
        assert good["mrr"] == 1.0
        assert bad["top1"] == 0.0
        assert bad["mrr"] == 0.25
        assert bad["mean_rank"] == 4.0
    return


def test_chance_metrics_for_five_candidates():
    with given("a five-way ranking task"):
        expected = chance_metrics(5)

    with then("the chance top-1 is one fifth and MRR is the harmonic mean"):
        assert expected["chance_top1"] == 0.2
        assert abs(expected["chance_mrr"] - 0.4566666667) < 1e-6
        assert expected["chance_mean_rank"] == 3.0
    return


def test_theme_ranking_cases_have_one_positive_and_distinct_negatives():
    with given("puzzle samples with fork and non-fork themes"):
        samples = [
            sample(0, puzzle_id="p0", themes=("fork",)),
            sample(1, puzzle_id="p1", themes=("fork",)),
            sample(2, puzzle_id="p2", themes=("fork",)),
            sample(3, puzzle_id="p3", themes=("pin",)),
            sample(4, puzzle_id="p4", themes=("pin",)),
            sample(5, puzzle_id="p5", themes=("pin",)),
        ]
        surrogate_z = np.zeros((len(samples), 1))

    with when("three-way fork ranking cases are built"):
        cases = build_theme_ranking_cases(
            samples,
            "fork",
            k=3,
            max_queries=10,
            matched=False,
            surrogate_z=surrogate_z,
            seed=0,
        )

    with then("each case has a fork positive and distinct candidates"):
        assert cases
        for case in cases:
            assert len(case.candidates) == 3
            assert len(set(case.candidates)) == 3
            positive = case.candidates[case.positive_position]
            assert "fork" in samples[positive].themes
            assert positive != case.candidates[(case.positive_position + 1) % 3]
    return


def test_temporal_ranking_uses_a_same_game_positive():
    with given("test positions from two games"):
        samples = [
            sample(0, source="test", game_id="g1", ply=0),
            sample(1, source="test", game_id="g1", ply=1),
            sample(2, source="test", game_id="g1", ply=2),
            sample(3, source="test", game_id="g2", ply=0),
            sample(4, source="test", game_id="g2", ply=1),
        ]
        surrogate_z = np.zeros((len(samples), 1))

    with when("three-way temporal cases are built"):
        cases = build_temporal_ranking_cases(
            samples,
            k=3,
            max_queries=10,
            matched=False,
            surrogate_z=surrogate_z,
            seed=0,
        )

    with then("the positive comes from the same game"):
        assert cases
        for case in cases:
            positive = case.candidates[case.positive_position]
            assert samples[positive].game_id in {"g1", "g2"}
            games = {samples[index].game_id for index in case.candidates}
            assert len(games) == 2
    return


def test_theme_ranking_metrics_reward_separable_embeddings():
    with given("fork and pin embeddings in separate clusters"):
        samples = [
            sample(i, puzzle_id=f"p{i}", themes=("fork",)) for i in range(10)
        ] + [sample(10 + i, puzzle_id=f"q{i}", themes=("pin",)) for i in range(10)]
        rng = np.random.default_rng(0)
        embeddings = np.vstack(
            [
                rng.normal(loc=[1.0, 0.0], scale=0.01, size=(10, 2)),
                rng.normal(loc=[0.0, 1.0], scale=0.01, size=(10, 2)),
            ]
        )

    with when("theme ranking metrics are computed"):
        metrics = theme_ranking_metrics(
            embeddings,
            samples,
            ("fork", "pin"),
            k=5,
            max_queries=10,
            seed=0,
        )

    with then("top-1 is perfect for both random and matched negatives"):
        assert metrics["mean_random"]["top1"] == 1.0
        assert metrics["mean_matched"]["top1"] == 1.0
        assert metrics["mean_matched"]["mrr"] == 1.0
    return


def test_temporal_ranking_metrics_reward_temporal_embeddings():
    with given("same-game test positions embedded close together"):
        samples = [
            sample(0, source="test", game_id="g1", ply=0),
            sample(1, source="test", game_id="g1", ply=1),
            sample(2, source="test", game_id="g1", ply=2),
            sample(3, source="test", game_id="g2", ply=0),
            sample(4, source="test", game_id="g2", ply=1),
            sample(5, source="test", game_id="g2", ply=2),
        ]
        embeddings = np.array(
            [
                [1.0, 0.0],
                [1.0, 0.01],
                [1.0, 0.02],
                [0.0, 1.0],
                [0.01, 1.0],
                [0.02, 1.0],
            ]
        )

    with when("temporal ranking metrics are computed"):
        metrics = temporal_ranking_metrics(
            embeddings, samples, k=3, max_queries=6, seed=0
        )

    with then("top-1 is perfect for random and matched negatives"):
        assert metrics["random"]["top1"] == 1.0
        assert metrics["matched"]["top1"] == 1.0
    return


def test_theme_ranking_examples_mark_one_positive():
    with given("separable fork and pin samples"):
        samples = [sample(i, puzzle_id=f"p{i}", themes=("fork",)) for i in range(6)] + [
            sample(10 + i, puzzle_id=f"q{i}", themes=("pin",)) for i in range(6)
        ]
        embeddings = np.vstack(
            [
                np.tile([1.0, 0.0], (6, 1)),
                np.tile([0.0, 1.0], (6, 1)),
            ]
        )

    with when("theme ranking examples are built"):
        examples = theme_ranking_examples(
            embeddings,
            samples,
            ("fork", "pin"),
            k=5,
            max_queries=10,
            seed=0,
            matched=False,
            max_per_bucket=2,
        )

    with then("each example has one positive candidate and a bucket label"):
        assert len(examples) == 12
        assert {example["bucket"] for example in examples} == {
            "best",
            "random",
            "worst",
        }
        for example in examples:
            assert len(example["candidates"]) == 5
            assert sum(c["is_positive"] for c in example["candidates"]) == 1
    return


def test_temporal_ranking_examples_mark_one_positive():
    with given("two games with temporal structure"):
        samples = [
            sample(0, source="test", game_id="g1", ply=0),
            sample(1, source="test", game_id="g1", ply=1),
            sample(2, source="test", game_id="g1", ply=2),
            sample(3, source="test", game_id="g2", ply=0),
            sample(4, source="test", game_id="g2", ply=1),
            sample(5, source="test", game_id="g2", ply=2),
        ]
        embeddings = np.array(
            [
                [1.0, 0.0],
                [1.0, 0.01],
                [1.0, 0.02],
                [0.0, 1.0],
                [0.01, 1.0],
                [0.02, 1.0],
            ]
        )

    with when("temporal ranking examples are built"):
        examples = temporal_ranking_examples(
            embeddings,
            samples,
            k=3,
            max_queries=10,
            seed=0,
            matched=False,
            max_per_bucket=2,
        )

    with then("the positive candidate is marked and bucketed"):
        assert examples
        assert {example["bucket"] for example in examples} == {
            "best",
            "random",
            "worst",
        }
        for example in examples:
            assert len(example["candidates"]) == 3
            assert sum(c["is_positive"] for c in example["candidates"]) == 1
    return


def test_surrogate_ranking_metrics_reward_separable_classes():
    with given("samples separated by material class"):
        samples = [
            sample(
                i,
                source="test",
                game_id=f"g{i}",
                surrogates={"material": 5, "phase": 2, "open_files": 0},
            )
            for i in range(8)
        ] + [
            sample(
                10 + i,
                source="test",
                game_id=f"h{i}",
                surrogates={"material": -5, "phase": 2, "open_files": 0},
            )
            for i in range(8)
        ]
        rng = np.random.default_rng(0)
        embeddings = np.vstack(
            [
                rng.normal(loc=[1.0, 0.0], scale=0.01, size=(8, 2)),
                rng.normal(loc=[0.0, 1.0], scale=0.01, size=(8, 2)),
            ]
        )
        task = next(task for task in SURROGATE_TASKS if task.name == "material")

    with when("surrogate ranking metrics are computed"):
        metrics = surrogate_ranking_metrics(
            embeddings, samples, (task,), k=5, max_queries_per_class=8, seed=0
        )

    with then("the classes are perfectly ranked against random negatives"):
        assert metrics["mean_random"]["top1"] == 1.0
    return


def test_surrogate_ranking_examples_are_grouped_by_class():
    with given("samples from two material classes"):
        samples = [
            sample(
                i,
                source="test",
                game_id=f"g{i}",
                surrogates={"material": 5, "phase": 1},
            )
            for i in range(6)
        ] + [
            sample(
                10 + i,
                source="test",
                game_id=f"h{i}",
                surrogates={"material": -5, "phase": 1},
            )
            for i in range(6)
        ]
        rng = np.random.default_rng(0)
        embeddings = np.vstack(
            [
                rng.normal(loc=[1.0, 0.0], scale=0.01, size=(6, 2)),
                rng.normal(loc=[0.0, 1.0], scale=0.01, size=(6, 2)),
            ]
        )
        task = next(task for task in SURROGATE_TASKS if task.name == "material")

    with when("surrogate ranking examples are built"):
        examples = surrogate_ranking_examples(
            embeddings,
            samples,
            (task,),
            k=5,
            max_queries_per_class=5,
            seed=0,
            modes=("random",),
            max_per_bucket=2,
        )

    with then("each class has examples with one positive candidate"):
        assert "material" in examples
        assert set(examples["material"]) == {"white_advantage", "black_advantage"}
        for classes in examples["material"].values():
            buckets = classes["random"]
            assert set(buckets) == {"best", "random", "worst"}
            for rows in buckets.values():
                assert all(
                    sum(candidate["is_positive"] for candidate in row["candidates"])
                    == 1
                    for row in rows
                )
    return


def test_temporal_ranking_skips_pairs_beyond_max_ply_distance():
    with given("a game with an isolated late position"):
        samples = [
            sample(0, source="test", game_id="g1", ply=0),
            sample(1, source="test", game_id="g1", ply=1),
            sample(2, source="test", game_id="g1", ply=40),
            sample(3, source="test", game_id="g2", ply=0),
            sample(4, source="test", game_id="g2", ply=1),
        ]
        surrogate_z = np.zeros((len(samples), 1))

    with when("temporal cases are built with a one-ply radius"):
        cases = build_temporal_ranking_cases(
            samples,
            k=3,
            max_queries=10,
            matched=False,
            surrogate_z=surrogate_z,
            seed=0,
            max_ply_distance=1,
        )

    with then("every positive is a one-ply neighbour"):
        assert cases
        for case in cases:
            query = samples[case.query]
            positive = samples[case.candidates[case.positive_position]]
            assert abs(positive.ply - query.ply) <= 1
            assert query.ply != 40
    return


def test_temporal_ranking_samples_any_neighbour_within_window():
    with given("many games with four contiguous positions"):
        samples = [
            sample(
                game * 4 + offset,
                source="test",
                game_id=f"g{game}",
                ply=offset,
            )
            for game in range(20)
            for offset in range(4)
        ]
        surrogate_z = np.zeros((len(samples), 1))

    with when("temporal cases are built with a three-ply window"):
        cases = build_temporal_ranking_cases(
            samples,
            k=5,
            max_queries=200,
            matched=False,
            surrogate_z=surrogate_z,
            seed=0,
            max_ply_distance=3,
        )

    with then("positive distances vary across the window"):
        distances = {
            abs(
                samples[case.candidates[case.positive_position]].ply
                - samples[case.query].ply
            )
            for case in cases
        }
        assert distances == {1, 2, 3}
    return
