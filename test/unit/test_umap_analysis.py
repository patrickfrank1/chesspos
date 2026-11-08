from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from bdd import given, then, when
from scripts.analysis.umap_embedding_structure import (
    adjacent_stats,
    binary_theme_purity,
    jsonable,
    label_purity,
    same_game_fraction,
    umap_step_stats,
)


def test_label_purity_measures_neighbour_agreement_against_prevalence():
    with given("two labels with equal prevalence and perfectly grouped neighbours"):
        labels = np.array(["a", "a", "b", "b"])
        indices = np.array([[1], [0], [3], [2]])

    with when("k-nearest-neighbour purity is computed"):
        result = label_purity(indices, labels)

    with then("purity is perfect and the baseline is the class prevalence"):
        assert result["purity"] == pytest.approx(1.0)
        assert result["random_baseline"] == pytest.approx(0.5)
        assert result["lift"] == pytest.approx(0.5)
        assert result["class_counts"] == {"a": 2, "b": 2}
    return


def test_binary_theme_purity_only_averages_positive_queries():
    with given("two positive queries whose neighbours are half positive"):
        positive = np.array([True, True, False, False])
        indices = np.array([[1, 2], [0, 3], [0, 1], [0, 1]])

    with when("binary theme purity is computed"):
        result = binary_theme_purity(indices, positive)

    with then("only the two positive queries contribute to purity"):
        assert result["n"] == 2
        assert result["purity"] == pytest.approx(0.5)
        assert result["prevalence"] == pytest.approx(0.5)
        assert result["lift"] == pytest.approx(0.0)
    return


def test_same_game_fraction_compares_to_the_game_size_baseline():
    with given("two games of two positions each and neighbours that mix games"):
        game_ids = np.array(["a", "a", "b", "b"])
        indices = np.array([[1, 2], [0, 3], [3, 0], [2, 1]])

    with when("the same-game neighbour fraction is computed"):
        result = same_game_fraction(indices, game_ids)

    with then("the observed fraction and the game-aware baseline are correct"):
        assert result["same_game_fraction"] == pytest.approx(0.5)
        assert result["random_baseline"] == pytest.approx(1.0 / 3.0)
        assert result["lift"] == pytest.approx(1.0 / 6.0)
    return


def test_adjacent_stats_hits_adjacent_plies_in_the_neighbour_list():
    with given("an eight-position game with adjacent plies and a line embedding"):
        embeddings = np.stack(
            [np.linspace(1.0, 8.0, 8), np.linspace(0.0, 0.7, 8)], axis=1
        )
        embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
        frame = pd.DataFrame(
            {
                "game_id": ["g"] * 8,
                "ply": list(range(8)),
            }
        )
        indices = np.array(
            [[1, 2], [0, 2], [1, 3], [2, 4], [3, 5], [4, 6], [5, 7], [6, 5]]
        )

    with when("adjacent-ply statistics are computed"):
        result = adjacent_stats(embeddings, frame, indices, seed=17, k=2)

    with then("all adjacent positives are found in the neighbour lists"):
        assert result["n_pairs"] == 7
        assert result["knn_adjacent_hit_rate"] == pytest.approx(1.0)
        assert result["adjacent_median"] < result["random_median"]
    return


def test_umap_step_stats_rewards_short_adjacent_steps():
    with given("a one-dimensional UMAP embedding for an eight-position game"):
        coords = np.stack([np.arange(8.0), np.zeros(8)], axis=1)
        frame = pd.DataFrame(
            {
                "game_id": ["g"] * 8,
                "ply": list(range(8)),
            }
        )

    with when("UMAP step statistics are computed"):
        result = umap_step_stats(coords, frame, seed=17)

    with then("adjacent steps are shorter than random steps"):
        assert result["adjacent_median"] == pytest.approx(1.0)
        assert 0.0 < result["ratio"] < 1.0
    return


def test_jsonable_replaces_non_finite_floats_with_none():
    with given("a result containing a NaN metric"):
        payload = {"metric": float("nan"), "nested": [float("inf")]}

    with when("the payload is made JSON-safe"):
        result = jsonable(payload)

    with then("non-finite values become null"):
        assert result == {"metric": None, "nested": [None]}
    return
