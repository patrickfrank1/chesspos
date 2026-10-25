from __future__ import annotations

import chess
import numpy as np
import pytest
import torch

from bdd import given, then, when
from src.dataset.token_stream import TokenStreamEncoder
from src.evaluation.retrieval import (
    BenchPosition,
    RetrievalBenchmark,
    alignment_uniformity,
    bootstrap_ci,
    effective_rank,
    evaluate_near,
    evaluate_retrieval,
    evaluate_strategic,
    ranking_metrics,
    retrieval_scores,
    strategic_pool,
    strategic_theme_metrics,
)
from src.modeling.transformer import MaskedStreamTransformer, TransformerConfig


def tiny_encoder() -> MaskedStreamTransformer:
    return MaskedStreamTransformer(
        TransformerConfig(
            vocab_size=52,
            d_model=1,
            n_heads=1,
            n_layers=1,
            d_ff=2,
            dropout=0.0,
            max_seq_len=256,
            max_segments=8,
        )
    ).eval()


def board_tokens() -> tuple[int, ...]:
    return tuple(int(value) for value in TokenStreamEncoder().encode(chess.Board()))


def make_benchmark() -> RetrievalBenchmark:
    tokens = board_tokens()
    positions = (
        BenchPosition(0, "a", 0, "", tokens, ("rookEndgame",)),
        BenchPosition(1, "b", 0, "", tokens, ("rookEndgame",)),
        BenchPosition(2, "c", 0, "", tokens, ("pawnEndgame",)),
    )
    return RetrievalBenchmark(
        positions=positions,
        near_queries=np.array([0]),
        near_relevance={0: {1: 0}},
        strategic_queries={"rookEndgame": np.array([0])},
        manifest={},
    )


def make_strategic_benchmark() -> RetrievalBenchmark:
    tokens = board_tokens()
    positions: list[BenchPosition] = []
    index = 0
    for puzzle in range(4):
        for ply in range(3):
            positions.append(
                BenchPosition(
                    index,
                    f"rook-{puzzle}",
                    ply,
                    "",
                    tokens,
                    ("rookEndgame",),
                )
            )
            index += 1
    for puzzle in range(20):
        for ply in range(2):
            positions.append(
                BenchPosition(index, f"pawn-{puzzle}", ply, "", tokens, ())
            )
            index += 1
    return RetrievalBenchmark(
        positions=tuple(positions),
        near_queries=np.array([], dtype=np.int64),
        near_relevance={},
        strategic_queries={"rookEndgame": np.array([0, 3, 6, 9], dtype=np.int64)},
        manifest={
            "seed": 5,
            "strategic_positives": 2,
            "strategic_negatives": 5,
        },
    )


class TestRankingMetrics:
    def test_recall_counts_queries_with_a_hit(self):
        with given("two queries with different relevant gallery items"):
            scores = np.array([[0.9, 0.8, 0.1], [0.5, 0.4, 0.3]])
            relevance = [{1: 0}, {0: 0, 2: 2}]

        with when("ranking metrics are computed at k=1 and k=2"):
            metrics = ranking_metrics(scores, relevance, ks=(1, 2))

        with then("each query contributes a hit only inside k"):
            assert metrics["recall@1"] == pytest.approx(0.5)
            assert metrics["recall@2"] == pytest.approx(1.0)
            assert metrics["coverage"] == pytest.approx(1.0)
        return

    def test_query_without_relevant_candidates_is_skipped(self):
        with given("one query with positives and one without"):
            scores = np.array([[0.9, 0.1], [0.8, 0.2]])
            relevance = [{0: 0}, {}]

        with when("ranking metrics are computed"):
            metrics = ranking_metrics(scores, relevance, ks=(1,))

        with then("coverage drops and the empty query is ignored"):
            assert metrics["recall@1"] == pytest.approx(1.0)
            assert metrics["coverage"] == pytest.approx(0.5)
        return

    def test_map_rewards_relevant_items_ranked_higher(self):
        with given("one query with a relevant item at rank one and rank three"):
            scores = np.array([[0.9, 0.8, 0.7]])
            relevance = [{0: 0, 2: 2}]

        with when("average precision is computed"):
            metrics = ranking_metrics(scores, relevance, ks=(3,))

        with then("the precision at each hit is averaged"):
            assert metrics["map"] == pytest.approx((1.0 + 2.0 / 3.0) / 2.0)
        return


class TestScoreMasking:
    def test_same_puzzle_candidates_are_masked(self):
        with given("a query and two gallery items from the same and another puzzle"):
            embeddings = np.array([[1.0, 0.0], [0.99, 0.0], [0.0, 1.0]])
            query_ids = np.array([0])
            gallery_ids = np.arange(3)

        with when("retrieval scores are computed"):
            scores = retrieval_scores(
                embeddings,
                query_ids,
                gallery_ids,
                puzzle_ids=["a", "a", "b"],
            )

        with then("same-puzzle candidates cannot be retrieved"):
            assert np.isneginf(scores[0, 0])
            assert np.isneginf(scores[0, 1])
            assert np.isfinite(scores[0, 2])
        return


class TestGeometryMetrics:
    def test_effective_rank_is_one_for_collapsed_embeddings(self):
        with given("identical embeddings"):
            embeddings = np.tile(np.array([[1.0, 2.0, 3.0]]), (16, 1))

        with when("the effective rank is computed"):
            rank = effective_rank(embeddings)

        with then("only one dimension is used"):
            assert rank == pytest.approx(1.0)
        return

    def test_effective_rank_is_high_for_isotropic_embeddings(self):
        with given("an isotropic random embedding matrix"):
            rng = np.random.default_rng(0)
            embeddings = rng.normal(size=(512, 32))

        with when("the effective rank is computed"):
            rank = effective_rank(embeddings)

        with then("many dimensions are used"):
            assert rank > 10
        return

    def test_alignment_is_zero_for_identical_positive_pairs(self):
        with given("positive pairs with identical embeddings"):
            embeddings = np.array([[1.0, 0.0], [0.0, 1.0]])
            positive_pairs = [(0, 0), (1, 1)]

        with when("alignment is computed"):
            metrics = alignment_uniformity(embeddings, positive_pairs, seed=0)

        with then("identical pairs have zero alignment error"):
            assert metrics["alignment"] == pytest.approx(0.0)
        return

    def test_uniformity_rewards_spread_embeddings(self):
        with given("random unit embeddings and collapsed embeddings"):
            rng = np.random.default_rng(0)
            spread = rng.normal(size=(128, 16))
            spread /= np.linalg.norm(spread, axis=1, keepdims=True)
            collapsed = np.tile(spread[0], (128, 1))

        with when("uniformity is computed for both"):
            spread_metrics = alignment_uniformity(spread, [(0, 1)], seed=0)
            collapsed_metrics = alignment_uniformity(collapsed, [(0, 1)], seed=0)

        with then("spread embeddings have lower uniformity loss"):
            assert spread_metrics["uniformity"] < collapsed_metrics["uniformity"]
            assert collapsed_metrics["uniformity"] == pytest.approx(0.0)
        return

    def test_bootstrap_ci_brackets_the_point_estimate(self):
        with given("a small sample of per-query hits"):
            values = np.array([1.0, 0.0, 1.0, 0.0, 1.0])

        with when("a bootstrap confidence interval is computed"):
            low, high = bootstrap_ci(values, seed=0, n_boot=500)

        with then("the interval brackets the sample mean"):
            assert low <= values.mean() <= high
        return


class TestEvaluateNear:
    def test_uses_cross_puzzle_scores_and_relevance(self):
        with given("a benchmark with one query and one exact positive"):
            benchmark = make_benchmark()
            embeddings = np.array([[1.0, 0.0], [0.9, 0.1], [0.0, 1.0]])

        with when("near retrieval is evaluated"):
            metrics = evaluate_near(benchmark, embeddings, ks=(1, 5))

        with then("the positive is retrieved at rank one"):
            assert metrics["recall@1"] == pytest.approx(1.0)
            assert metrics["map"] == pytest.approx(1.0)
            assert metrics["coverage"] == pytest.approx(1.0)
            assert metrics["recall@1_ci_low"] <= metrics["recall@1"]
            assert metrics["recall@1_ci_high"] >= metrics["recall@1"]
        return


class TestStrategicPooling:
    def test_pool_is_balanced_cross_puzzle_and_deterministic(self):
        with given("a theme with more eligible candidates than the pool allows"):
            benchmark = make_strategic_benchmark()

        with when("the same strategic pool is sampled twice"):
            galleries, relevance = strategic_pool(benchmark, "rookEndgame", 0)
            again_galleries, again_relevance = strategic_pool(
                benchmark, "rookEndgame", 0
            )

        with then("the pool has fixed balanced composition and is reproducible"):
            assert len(galleries) == 7
            assert len(relevance) == 2
            assert np.array_equal(galleries, again_galleries)
            assert relevance == again_relevance
            positives = {int(galleries[local]) for local in relevance}
            negatives = {int(gallery) for gallery in galleries} - positives
            assert len(positives) == 2
            assert len(negatives) == 5
            for index in positives:
                assert "rookEndgame" in benchmark.positions[index].themes
                assert benchmark.positions[index].puzzle_id != "rook-0"
            for index in negatives:
                assert "rookEndgame" not in benchmark.positions[index].themes
                assert benchmark.positions[index].puzzle_id != "rook-0"
        return

    def test_pool_caps_positives_at_available_candidates(self):
        with given("a query with nine eligible positives"):
            benchmark = make_strategic_benchmark()

        with when("the pool requests ten positives and five negatives"):
            galleries, relevance = strategic_pool(
                benchmark, "rookEndgame", 0, positives=10, negatives=5
            )

        with then("all available positives are used alongside five negatives"):
            assert len(relevance) == 9
            assert len(galleries) == 14
        return

    def test_pools_do_not_depend_on_query_order(self):
        with given("the same queries stored in reverse order"):
            benchmark = make_strategic_benchmark()
            reordered = RetrievalBenchmark(
                positions=benchmark.positions,
                near_queries=benchmark.near_queries,
                near_relevance=benchmark.near_relevance,
                strategic_queries={
                    "rookEndgame": benchmark.strategic_queries["rookEndgame"][::-1]
                },
                manifest=benchmark.manifest,
            )
            embeddings = np.random.default_rng(0).standard_normal(
                (len(benchmark.positions), 8)
            )

        with when("per-theme metrics are computed for both orders"):
            first = strategic_theme_metrics(benchmark, embeddings)
            second = strategic_theme_metrics(reordered, embeddings)

        with then("the metrics agree"):
            assert first["rookEndgame"]["map"] == pytest.approx(
                second["rookEndgame"]["map"]
            )
            assert first["rookEndgame"]["chance_map"] == pytest.approx(
                second["rookEndgame"]["chance_map"]
            )
        return


class TestStrategicChanceAdjustedMetrics:
    def test_reports_chance_map_and_lift(self):
        with given("a balanced benchmark and random embeddings"):
            benchmark = make_strategic_benchmark()
            embeddings = np.random.default_rng(0).standard_normal(
                (len(benchmark.positions), 8)
            )

        with when("strategic metrics are computed"):
            metrics = evaluate_strategic(benchmark, embeddings, ks=(1, 5))

        with then("chance and chance-adjusted MAP are reported"):
            assert metrics["n_queries"] == 4
            assert metrics["chance_map"] == pytest.approx(2 / 7)
            assert metrics["map_lift"] == pytest.approx(
                metrics["map"] - metrics["chance_map"]
            )
        return

    def test_handles_queries_with_different_pool_sizes(self):
        with given("a theme whose puzzles contribute different positive counts"):
            tokens = board_tokens()
            benchmark = RetrievalBenchmark(
                positions=(
                    BenchPosition(0, "a", 0, "", tokens, ("rookEndgame",)),
                    BenchPosition(1, "a", 1, "", tokens, ("rookEndgame",)),
                    BenchPosition(2, "a", 2, "", tokens, ("rookEndgame",)),
                    BenchPosition(3, "b", 0, "", tokens, ("rookEndgame",)),
                    BenchPosition(4, "n0", 0, "", tokens, ()),
                    BenchPosition(5, "n1", 0, "", tokens, ()),
                ),
                near_queries=np.array([], dtype=np.int64),
                near_relevance={},
                strategic_queries={"rookEndgame": np.array([0, 3])},
                manifest={
                    "seed": 1,
                    "strategic_positives": 10,
                    "strategic_negatives": 2,
                },
            )
            embeddings = np.random.default_rng(0).standard_normal((6, 8))

        with when("per-theme metrics are computed"):
            metrics = strategic_theme_metrics(benchmark, embeddings)["rookEndgame"]

        with then("queries with different pool sizes are aggregated"):
            assert metrics["n_queries"] == 2
            assert metrics["chance_map"] == pytest.approx((1 / 3 + 3 / 5) / 2)
        return

    def test_per_theme_metrics_include_map_and_lift(self):
        with given("a balanced benchmark and random embeddings"):
            benchmark = make_strategic_benchmark()
            embeddings = np.random.default_rng(0).standard_normal(
                (len(benchmark.positions), 8)
            )

        with when("per-theme metrics are computed"):
            per_theme = strategic_theme_metrics(benchmark, embeddings)

        with then("the theme reports MAP, chance and its lift"):
            assert set(per_theme) == {"rookEndgame"}
            metrics = per_theme["rookEndgame"]
            assert metrics["n_queries"] == 4
            assert metrics["chance_map"] == pytest.approx(2 / 7)
            assert metrics["map_lift"] == pytest.approx(
                metrics["map"] - metrics["chance_map"]
            )
        return


class TestEvaluateRetrieval:
    def test_reports_near_strategic_and_geometry_metrics(self):
        with given("a tiny encoder and a benchmark with both tasks"):
            benchmark = make_benchmark()
            encoder = tiny_encoder()

        with when("the retrieval suite is evaluated"):
            metrics = evaluate_retrieval(
                encoder, benchmark, torch.device("cpu"), batch_size=2
            )

        with then("all metric families are reported"):
            assert "near_recall@1" in metrics
            assert "near_map" in metrics
            assert "strategic_recall@1" in metrics
            assert "strategic_map" in metrics
            assert "strategic_chance_map" in metrics
            assert "strategic_map_lift" in metrics
            assert "strategic_theme_rookEndgame_map" in metrics
            assert "strategic_theme_rookEndgame_chance_map" in metrics
            assert "strategic_theme_rookEndgame_map_lift" in metrics
            assert "strategic_theme_rookEndgame_n_queries" in metrics
            assert "rank_effective" in metrics
            assert "alignment" in metrics
            assert "uniformity" in metrics
            assert 0.0 <= metrics["near_recall@1"] <= 1.0
            assert metrics["rank_effective"] >= 1.0
        return
