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
    ranking_metrics,
    retrieval_scores,
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
            assert "rank_effective" in metrics
            assert "alignment" in metrics
            assert "uniformity" in metrics
            assert 0.0 <= metrics["near_recall@1"] <= 1.0
            assert metrics["rank_effective"] >= 1.0
        return
