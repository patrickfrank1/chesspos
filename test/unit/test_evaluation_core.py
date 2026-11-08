from __future__ import annotations

import chess
import numpy as np
import pytest
import torch

from bdd import given, then, when
from src.dataset.token_stream import TokenStreamEncoder
from src.evaluation.core.embeddings import embed_tokens
from src.evaluation.core.ranking import (
    chance_metrics,
    cosine_scores,
    normalize,
    rank_of_first_relevant,
    ranking_metrics,
    recall_hits,
)
from src.modeling.transformer import MaskedStreamTransformer, TransformerConfig


def tiny_model() -> MaskedStreamTransformer:
    return MaskedStreamTransformer(
        TransformerConfig(
            d_model=8,
            n_heads=2,
            n_layers=1,
            d_ff=16,
            dropout=0.0,
            max_seq_len=128,
            max_segments=2,
        )
    )


def board_segments() -> list[np.ndarray]:
    encoder = TokenStreamEncoder()
    return [encoder.encode(chess.Board())]


def test_normalize_returns_unit_rows_and_keeps_zero_vectors_finite():
    with given("a matrix with a normal row and a zero row"):
        embeddings = np.array([[3.0, 4.0], [0.0, 0.0]], dtype=np.float64)

    with when("the matrix is normalized"):
        values = normalize(embeddings)

    with then("non-zero rows are unit length and zero rows stay zero"):
        assert np.linalg.norm(values[0]) == pytest.approx(1.0)
        assert np.all(values[1] == 0.0)
        assert np.isfinite(values).all()
    return


def test_cosine_scores_compare_queries_against_a_gallery():
    with given("three unnormalized embeddings and explicit query/gallery ids"):
        embeddings = np.array([[2.0, 0.0], [0.0, 3.0], [1.0, 1.0]], dtype=np.float64)

    with when("cosine scores are computed"):
        scores = cosine_scores(
            embeddings,
            query_ids=np.array([0]),
            gallery_ids=np.array([1, 2]),
        )

    with then("the scores are dot products of normalized rows"):
        assert scores.shape == (1, 2)
        assert scores[0, 0] == pytest.approx(0.0)
        assert scores[0, 1] == pytest.approx(1.0 / np.sqrt(2.0))
    return


def test_rank_of_first_relevant_averages_tied_scores():
    with given("a query whose relevant item ties with another candidate"):
        scores = np.array([0.5, 0.5, 0.1], dtype=np.float64)

    with when("the rank of the relevant item is computed"):
        rank = rank_of_first_relevant(scores, {0: 0})

    with then("the tied rank is the average of the occupied ranks"):
        assert rank == pytest.approx(1.5)
    return


def test_rank_of_first_relevant_uses_any_relevant_item_in_graded_relevance():
    with given("two relevant items with different grades and scores"):
        scores = np.array([0.9, 0.8, 0.1], dtype=np.float64)

    with when("the rank is computed for graded relevance"):
        rank = rank_of_first_relevant(scores, {0: 2, 1: 0})

    with then("the highest-scored relevant item determines the rank"):
        assert rank == pytest.approx(1.0)
    return


def test_ranking_metrics_reports_binary_ranking_and_retrieval_metrics():
    with given("two binary queries with the positive first and third"):
        scores = np.array([[0.9, 0.1, 0.0], [0.1, 0.8, 0.2]], dtype=np.float64)
        relevance = [{0: 0}, {0: 0}]

    with when("ranking metrics are computed at k=1 and k=3"):
        metrics = ranking_metrics(scores, relevance, ks=(1, 3))

    with then("all fixed-size and retrieval metrics are aggregated together"):
        assert metrics["coverage"] == pytest.approx(1.0)
        assert metrics["n_queries"] == 2
        assert metrics["top1"] == pytest.approx(0.5)
        assert metrics["mrr"] == pytest.approx((1.0 + 1.0 / 3.0) / 2.0)
        assert metrics["mean_rank"] == pytest.approx(2.0)
        assert metrics["recall@1"] == pytest.approx(0.5)
        assert metrics["recall@3"] == pytest.approx(1.0)
        assert metrics["ndcg@1"] == pytest.approx(0.5)
        assert metrics["ndcg@3"] == pytest.approx(0.75)
        assert metrics["map"] == pytest.approx((1.0 + 1.0 / 3.0) / 2.0)
    return


def test_ranking_metrics_keeps_graded_relevance_metrics():
    with given("one query with a relevant item at rank one and rank three"):
        scores = np.array([[0.9, 0.8, 0.7]], dtype=np.float64)
        relevance = [{0: 0, 2: 2}]

    with when("ranking metrics are computed"):
        metrics = ranking_metrics(scores, relevance, ks=(3,))

    with then("the graded average precision and rank metrics are preserved"):
        assert metrics["map"] == pytest.approx((1.0 + 2.0 / 3.0) / 2.0)
        assert metrics["top1"] == pytest.approx(1.0)
        assert metrics["mrr"] == pytest.approx(1.0)
        assert metrics["mean_rank"] == pytest.approx(1.0)
    return


def test_ranking_metrics_skips_queries_without_relevance():
    with given("one covered query and one query without relevant candidates"):
        scores = np.array([[0.9, 0.1], [0.8, 0.2]], dtype=np.float64)
        relevance = [{0: 0}, {}]

    with when("ranking metrics are computed"):
        metrics = ranking_metrics(scores, relevance, ks=(1,))

    with then("only the covered query contributes to the metrics"):
        assert metrics["coverage"] == pytest.approx(0.5)
        assert metrics["n_queries"] == 1
        assert metrics["top1"] == pytest.approx(1.0)
        assert metrics["mrr"] == pytest.approx(1.0)
        assert metrics["mean_rank"] == pytest.approx(1.0)
        assert metrics["recall@1"] == pytest.approx(1.0)
    return


def test_ranking_metrics_zeroes_when_no_query_has_relevance():
    with given("two queries without any relevant candidates"):
        scores = np.zeros((2, 3), dtype=np.float64)
        relevance = [{}, {}]

    with when("ranking metrics are computed"):
        metrics = ranking_metrics(scores, relevance, ks=(1, 3))

    with then("all aggregate metrics are zero and coverage is empty"):
        assert metrics["coverage"] == pytest.approx(0.0)
        assert metrics["n_queries"] == 0
        assert metrics["top1"] == pytest.approx(0.0)
        assert metrics["mrr"] == pytest.approx(0.0)
        assert metrics["mean_rank"] == pytest.approx(0.0)
        assert metrics["recall@1"] == pytest.approx(0.0)
        assert metrics["recall@3"] == pytest.approx(0.0)
        assert metrics["ndcg@1"] == pytest.approx(0.0)
        assert metrics["ndcg@3"] == pytest.approx(0.0)
        assert metrics["map"] == pytest.approx(0.0)
    return


def test_recall_hits_returns_one_value_per_covered_query():
    with given("two queries with positives at ranks one and three"):
        scores = np.array([[0.9, 0.1, 0.0], [0.1, 0.8, 0.2]], dtype=np.float64)
        relevance = [{0: 0}, {0: 0}]

    with when("recall hits are computed at k=1 and k=3"):
        at_one = recall_hits(scores, relevance, 1)
        at_three = recall_hits(scores, relevance, 3)

    with then("only queries with a hit inside k are counted"):
        assert at_one.tolist() == [1.0, 0.0]
        assert at_three.tolist() == [1.0, 1.0]
    return


def test_chance_metrics_match_a_uniform_random_ranking():
    with given("a five-way ranking task"):
        expected = chance_metrics(5)

    with then("the chance top-1, MRR, and mean rank are the random baselines"):
        assert expected["chance_top1"] == pytest.approx(0.2)
        assert expected["chance_mrr"] == pytest.approx(0.4566666667)
        assert expected["chance_mean_rank"] == pytest.approx(3.0)
    return


def test_embed_tokens_returns_normalized_cpu_vectors_and_restores_mode():
    with given("a tiny model in training mode and one board segment"):
        model = tiny_model()
        model.train()
        segments = board_segments()

    with when("the segments are embedded"):
        vectors = embed_tokens(model, segments, device="cpu", batch_size=1)

    with then("vectors are normalized, float32, on cpu, and training resumes"):
        assert vectors.shape == (1, 8)
        assert vectors.dtype == torch.float32
        assert vectors.device.type == "cpu"
        assert torch.linalg.norm(vectors, dim=-1)[0].item() == pytest.approx(
            1.0, abs=1e-5
        )
        assert model.training is True
    return


def test_embed_tokens_uses_autocast_when_bf16_is_enabled():
    with given("a tiny model whose embed call records autocast state"):
        model = tiny_model()
        model.eval()
        segments = board_segments()
        calls: list[bool] = []
        original_embed = model.embed

        def wrapped_embed(*args, **kwargs):
            calls.append(torch.is_autocast_enabled("cpu"))
            return original_embed(*args, **kwargs)

        model.embed = wrapped_embed

    with when("the segments are embedded with bf16 enabled"):
        embed_tokens(model, segments, device="cpu", bf16=True)

    with then("the embedding forward runs under autocast"):
        assert calls == [True]
    return
