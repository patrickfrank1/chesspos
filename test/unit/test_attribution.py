from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import chess
import numpy as np
import pytest
import torch
from fastapi.testclient import TestClient

from bdd import given, then, when
from src.app.chess_eval.app import create_app
from src.app.chess_eval.attribution import (
    attribute_position,
    deletion_curve,
    kernel_shap,
    occlusion_deltas,
)
from src.app.chess_eval.backend import ModelHandle, ModelInfo, evaluate_handle
from src.dataset.token_stream import EMPTY, MASK, VOCAB_SIZE
from src.modeling.puzzle_classifier import PoolingClassifier, PoolingClassifierConfig
from src.modeling.score_head import ScoreHead, ScoreHeadConfig
from src.modeling.transformer import MaskedStreamTransformer, TransformerConfig
from src.run.classify_position import load_classifier
from src.run.eval_score_probe import load_score_probe
from src.app.chess_eval.puzzles import SolvedPuzzle
from src.run.evaluate_attribution import (
    attribution_report,
    piece_localization_report,
)
from src.run.explain_position import main as explain_main
from test.unit.test_puzzles import queen_sac_boards


def tiny_config() -> TransformerConfig:
    return TransformerConfig(
        vocab_size=VOCAB_SIZE,
        d_model=2,
        n_heads=1,
        n_layers=1,
        d_ff=2,
        dropout=0.0,
        max_seq_len=256,
        max_segments=8,
    )


def save_score_checkpoint(path: Path, window: int = 1) -> None:
    config = tiny_config()
    encoder = MaskedStreamTransformer(config)
    head = ScoreHead(
        ScoreHeadConfig(d_model=2, hidden_dim=2, pooling="mean", state_features=0)
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "encoder_config": asdict(config),
            "encoder_state": encoder.state_dict(),
            "head_config": asdict(head.config),
            "head_state": head.state_dict(),
            "feature": "full",
            "token_pooling": "mean",
            "unfreeze_last_block": False,
            "unfreeze_all": False,
            "window": window,
            "include_state_features": False,
            "metrics": {"r2": 0.1, "auroc_decisive": 0.6, "mae": 0.8},
        },
        path,
    )
    return


def save_puzzle_checkpoint(path: Path) -> None:
    config = tiny_config()
    encoder = MaskedStreamTransformer(config)
    head = PoolingClassifier(
        PoolingClassifierConfig(d_model=2, hidden_dim=2, pooling="mean")
    )
    # Keep the tiny model away from sigmoid saturation so deltas stay visible.
    torch.nn.init.normal_(head.classifier.weight, std=0.01)
    torch.nn.init.zeros_(head.classifier.bias)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "encoder_config": asdict(config),
            "encoder_state": encoder.state_dict(),
            "head_config": asdict(head.config),
            "head_state": head.state_dict(),
            "feature": "full",
            "metrics": {"pr_auc": 0.2, "auroc": 0.7, "f1": 0.25, "threshold": 0.5},
        },
        path,
    )
    return


def score_handle(path: Path, window: int = 1) -> ModelHandle:
    info = ModelInfo(
        model_id="tests/score",
        label="tiny score",
        task="score",
        path=path,
        recipe="full",
        model="run_20m_lazy",
        seed=17,
    )
    return ModelHandle(
        info=info,
        model=load_score_probe(path, torch.device("cpu")),
        metrics={},
        metadata={
            "feature": "full",
            "window": window,
            "include_state_features": False,
            "token_pooling": "mean",
        },
    )


def sacrifice_handle(path: Path) -> ModelHandle:
    info = ModelInfo(
        model_id="tests/sacrifice",
        label="tiny sacrifice",
        task="sacrifice",
        path=path,
        recipe="full",
        model="run_6m_isolated",
        seed=17,
    )
    return ModelHandle(
        info=info,
        model=load_classifier(str(path), torch.device("cpu")),
        metrics={"threshold": 0.5},
        metadata={"feature": "full", "window": 1},
    )


def counting_scorer(rows_seen: list[np.ndarray]):
    """Deterministic scorer: counts knight tokens (value 5) in each row."""

    def score(rows: np.ndarray) -> np.ndarray:
        rows = np.asarray(rows)
        rows_seen.append(rows.copy())
        return (rows == 5).sum(axis=1).astype(np.float64)

    return score


def summing_scorer(rows: np.ndarray) -> np.ndarray:
    return np.asarray(rows, dtype=np.float64).sum(axis=1)


class TestOcclusionDeltas:
    def test_delta_is_the_output_lost_when_a_square_is_removed(self):
        with given("a board row with three pieces and a knight-counting scorer"):
            base = np.array([[EMPTY, EMPTY, 5, EMPTY, 5, EMPTY, EMPTY, 5]])
            positions = np.array([[2, 4, 7]])
            calls: list[np.ndarray] = []
            score = counting_scorer(calls)

        with when("each of the three squares is occluded once"):
            base_value, deltas = occlusion_deltas(
                score, base, positions, replacement=MASK
            )

        with then("the base value and the per-square drops are exact"):
            assert base_value == 3.0
            assert deltas.shape == (1, 3)
            assert deltas.tolist() == [[1.0, 1.0, 1.0]]

        with then("only the occluded token changes and it becomes the mask token"):
            base_row = base[0]
            occluded_rows = calls[1]
            assert occluded_rows.shape == (3, base_row.size)
            for index, square_position in enumerate(positions[0]):
                row = occluded_rows[index]
                changed = np.flatnonzero(row != base_row)
                assert changed.tolist() == [int(square_position)]
                assert row[square_position] == MASK
        return

    def test_occluding_an_empty_square_is_a_no_op_for_the_empty_replacement(self):
        with given("a board row with one piece and a matching scorer"):
            base = np.array([[EMPTY, EMPTY, 5, EMPTY]])
            calls: list[np.ndarray] = []
            score = counting_scorer(calls)

        with when("one occupied and one empty square are removed"):
            _, deltas = occlusion_deltas(
                score, base, np.array([[2, 0]]), replacement=EMPTY
            )

        with then("only the occupied square loses output"):
            assert deltas.tolist() == [[1.0, 0.0]]
        return

    def test_batches_the_occlusions_instead_of_one_forward_per_square(self):
        with given("a board row with four squares and a batch size of one"):
            base = np.array([[5, 5, 5, 5]])
            calls: list[np.ndarray] = []
            score = counting_scorer(calls)

        with when("the squares are occluded with a batch size of one"):
            base_value, deltas = occlusion_deltas(
                score, base, np.array([[0, 1, 2, 3]]), replacement=MASK, batch_size=1
            )

        with then("the base and every variant are still evaluated exactly once"):
            assert base_value == 4.0
            assert deltas.tolist() == [[1.0, 1.0, 1.0, 1.0]]
            assert sum(len(call) for call in calls) == 5
        return


class TestKernelShap:
    def test_split_a_two_player_and_game_into_equal_shares(self):
        with given(
            "a game where the output only appears when both players are present"
        ):

            def value_fn(coalitions: np.ndarray) -> np.ndarray:
                coalitions = np.asarray(coalitions, dtype=bool)
                return (coalitions[:, 0] & coalitions[:, 1]).astype(np.float64)

        with when("the Shapley values are estimated"):
            phi = kernel_shap(value_fn, n_players=2, nsamples=64, seed=0)

        with then("each player gets half of the interaction"):
            assert phi.shape == (2,)
            assert phi.tolist() == pytest.approx([0.5, 0.5], abs=1e-6)
        return

    def test_recover_linear_player_weights(self):
        with given("a linear game with known player weights"):
            weights = np.array([0.2, -1.0, 3.0, 0.5, -0.25])

            def value_fn(coalitions: np.ndarray) -> np.ndarray:
                coalitions = np.asarray(coalitions, dtype=np.float64)
                return coalitions @ weights

        with when("the Shapley values are estimated"):
            phi = kernel_shap(value_fn, n_players=5, nsamples=128, seed=3)

        with then("the players recover their weights exactly"):
            assert phi.tolist() == pytest.approx(weights.tolist(), abs=1e-6)
        return

    def test_efficiency_holds_for_a_non_linear_game(self):
        with given("a three-way synergy game"):

            def value_fn(coalitions: np.ndarray) -> np.ndarray:
                coalitions = np.asarray(coalitions, dtype=bool)
                return coalitions.all(axis=1).astype(np.float64)

        with when("the Shapley values are estimated"):
            phi = kernel_shap(value_fn, n_players=3, nsamples=64, seed=7)
            full = float(value_fn(np.ones((1, 3), dtype=bool))[0])
            empty = float(value_fn(np.zeros((1, 3), dtype=bool))[0])

        with then("the values sum to the output difference and share the synergy"):
            assert phi.sum() == pytest.approx(full - empty, abs=1e-6)
            assert phi.tolist() == pytest.approx([1 / 3, 1 / 3, 1 / 3], abs=1e-5)
        return

    def test_is_reproducible_for_a_seed(self):
        with given("a game with one dominant player"):

            def value_fn(coalitions: np.ndarray) -> np.ndarray:
                coalitions = np.asarray(coalitions, dtype=np.float64)
                return coalitions[:, 0] * 2.0 + coalitions[:, 1]

        with when("the same seed is used twice"):
            first = kernel_shap(value_fn, n_players=4, nsamples=32, seed=11)
            second = kernel_shap(value_fn, n_players=4, nsamples=32, seed=11)

        with then("the two estimates are identical"):
            assert first.tolist() == second.tolist()
        return

    def test_rejects_a_zero_player_game(self):
        with given("an empty coalition game"):

            def value_fn(coalitions: np.ndarray) -> np.ndarray:
                return np.zeros(len(coalitions))

        with when("the Shapley values are requested"):
            with pytest.raises(ValueError):
                kernel_shap(value_fn, n_players=0)
        return


class TestDeletionCurve:
    def test_deleting_high_impact_squares_first_drops_the_output_faster(self):
        with given("a board whose squares carry distinct output values"):
            base = np.array([[1, 2, 3, 4, 0]])
            positions = np.array([0, 1, 2, 3])

        with when("the largest squares are deleted first and then the smallest"):
            fast = deletion_curve(
                summing_scorer, base, positions, order=[3, 2, 1, 0], replacement=0
            )
            slow = deletion_curve(
                summing_scorer, base, positions, order=[0, 1, 2, 3], replacement=0
            )

        with then("the fast curve is already below the slow curve after one step"):
            assert fast[0] == slow[0] == 10.0
            assert fast[1] == 6.0
            assert slow[1] == 9.0
            assert fast[-1] == slow[-1] == 0.0
        return


class TestAttributionOnModels:
    def test_occlusion_explains_a_sacrifice_probability(self, tmp_path):
        with given("a tiny sacrifice checkpoint and the starting board"):
            path = tmp_path / "sacrifice.pt"
            save_puzzle_checkpoint(path)
            handle = sacrifice_handle(path)
            boards = [chess.Board()]

        with when("square occlusion explains the probability"):
            result = attribute_position(
                handle,
                boards,
                device="cpu",
                method="occlusion",
                occlusion="mask",
            )

        with then("every square and piece group is reported"):
            assert result.task == "sacrifice"
            assert result.method == "occlusion"
            assert len(result.squares) == 64
            assert len(result.segment_squares) == 1
            assert len(result.segment_totals) == 1
            assert all(np.isfinite(result.squares))
            assert "white pawn" in result.piece_groups
            assert "black knight" in result.piece_groups
            assert len(result.piece_groups) == 12
            assert result.model_value == pytest.approx(
                evaluate_handle(handle, boards, "cpu"), abs=1e-6
            )
        return

    def test_shap_is_efficient_for_a_single_position_score_probe(self, tmp_path):
        with given("a tiny score checkpoint and the starting board"):
            path = tmp_path / "score.pt"
            save_score_checkpoint(path)
            handle = score_handle(path)
            boards = [chess.Board()]

        with when("KernelSHAP explains the score"):
            result = attribute_position(
                handle,
                boards,
                device="cpu",
                method="shap",
                samples=64,
                seed=0,
            )

        with then("the square values sum to the output above the all-masked board"):
            total = float(sum(result.squares))
            assert total == pytest.approx(
                result.model_value - result.baseline_value, abs=1e-4
            )
            assert result.samples == 64
        return

    def test_shap_attributes_every_position_in_a_score_window(self, tmp_path):
        with given("a score probe with a two-position window and two boards"):
            path = tmp_path / "score_window.pt"
            save_score_checkpoint(path, window=2)
            handle = score_handle(path, window=2)
            boards = [
                chess.Board(),
                chess.Board(
                    "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1"
                ),
            ]

        with when("KernelSHAP explains the score"):
            result = attribute_position(
                handle, boards, device="cpu", method="shap", samples=48, seed=1
            )

        with then("both positions are attributed and the totals stay efficient"):
            assert len(result.segment_squares) == 2
            assert len(result.segment_totals) == 2
            assert len(result.squares) == 64
            total = sum(sum(segment) for segment in result.segment_squares)
            assert total == pytest.approx(
                result.model_value - result.baseline_value, abs=1e-4
            )
        return

    def test_ranked_squares_are_ordered_by_absolute_impact(self, tmp_path):
        with given("a tiny score checkpoint and the starting board"):
            path = tmp_path / "score.pt"
            save_score_checkpoint(path)
            handle = score_handle(path)

        with when("the top three squares are requested"):
            result = attribute_position(
                handle, [chess.Board()], device="cpu", method="occlusion"
            )
            top = result.ranked(limit=3)

        with then(
            "three distinct squares with names are returned by descending impact"
        ):
            assert len(top) == 3
            impacts = [abs(entry["value"]) for entry in top]
            assert impacts == sorted(impacts, reverse=True)
            for entry in top:
                assert 0 <= chess.parse_square(entry["square"]) < 64
                assert "value" in entry
        return

    def test_rejects_an_unknown_method(self, tmp_path):
        with given("a tiny sacrifice checkpoint"):
            path = tmp_path / "sacrifice.pt"
            save_puzzle_checkpoint(path)
            handle = sacrifice_handle(path)

        with when("an unsupported method is requested"):
            with pytest.raises(ValueError):
                attribute_position(handle, [chess.Board()], device="cpu", method="lime")
        return


class TestAttributionApi:
    def test_attribution_endpoint_explains_a_score_position(self, tmp_path):
        with given("an app with a tiny score checkpoint"):
            save_score_checkpoint(tmp_path / "score" / "full" / "best.pt")
            app = create_app(model_roots=[tmp_path], device="cpu")
            client = TestClient(app)
            model_id = client.get("/api/models").json()["models"][0]["model_id"]

        with when("an occlusion attribution is requested"):
            response = client.post(
                "/api/attribution",
                json={
                    "start_fen": chess.STARTING_FEN,
                    "moves": [],
                    "model_id": model_id,
                    "method": "occlusion",
                    "occlusion": "mask",
                },
            )

        with then("the attribution payload is returned for the current position"):
            assert response.status_code == 200
            payload = response.json()
            assert payload["attribution"]["method"] == "occlusion"
            assert len(payload["attribution"]["squares"]) == 64
            assert payload["attribution"]["top"]
            assert payload["fen"] == chess.STARTING_FEN
        return

    def test_unknown_model_and_method_are_rejected(self, tmp_path):
        with given("an app with a tiny score checkpoint"):
            save_score_checkpoint(tmp_path / "score" / "full" / "best.pt")
            app = create_app(model_roots=[tmp_path], device="cpu")
            client = TestClient(app)
            model_id = client.get("/api/models").json()["models"][0]["model_id"]

        with when("an unknown model is explained"):
            unknown = client.post(
                "/api/attribution",
                json={"model_id": "missing", "method": "occlusion"},
            )
        with then("the request is a 404"):
            assert unknown.status_code == 404

        with when("an unknown method is requested for a known model"):
            bad_method = client.post(
                "/api/attribution",
                json={"model_id": model_id, "method": "lime"},
            )
        with then("the request is a 400"):
            assert bad_method.status_code == 400
        return


class TestCounterfactualValidity:
    def test_kings_are_never_occluded(self, tmp_path):
        with given("a tiny sacrifice checkpoint and the starting board"):
            path = tmp_path / "sacrifice.pt"
            save_puzzle_checkpoint(path)
            handle = sacrifice_handle(path)
            boards = [chess.Board()]

        with when("square occlusion explains the probability"):
            result = attribute_position(
                handle, boards, device="cpu", method="occlusion"
            )

        with then("the kings cannot be removed and are reported as zero"):
            assert result.squares[chess.E1] == 0.0
            assert result.squares[chess.E8] == 0.0
            others = [
                square
                for square, piece in boards[0].piece_map().items()
                if piece.piece_type != chess.KING
            ]
            assert any(result.squares[square] != 0.0 for square in others)
        return

    def test_shap_with_the_board_as_background_cancels_out(self, tmp_path):
        with given("a background identical to the explained board"):
            path = tmp_path / "sacrifice.pt"
            save_puzzle_checkpoint(path)
            handle = sacrifice_handle(path)
            boards = [chess.Board()]

        with when("KernelSHAP explains the score with that background"):
            result = attribute_position(
                handle,
                boards,
                device="cpu",
                method="shap",
                samples=32,
                background=boards,
            )

        with then("replacing every piece with itself leaves no attribution"):
            assert all(abs(value) < 1e-6 for value in result.squares)
            assert result.baseline_value == pytest.approx(result.model_value, abs=1e-6)
            assert result.background == 1
        return

    def test_shap_is_efficient_against_a_background(self, tmp_path):
        with given("a score probe and a real background position"):
            path = tmp_path / "score.pt"
            save_score_checkpoint(path)
            handle = score_handle(path)
            boards = [chess.Board()]
            background = [
                chess.Board(
                    "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1"
                )
            ]

        with when("KernelSHAP explains the score"):
            result = attribute_position(
                handle,
                boards,
                device="cpu",
                method="shap",
                samples=64,
                background=background,
            )

        with then("the values still sum to the output above the background baseline"):
            total = float(sum(result.squares))
            assert total == pytest.approx(
                result.model_value - result.baseline_value, abs=1e-4
            )
        return


class TestPieceLocalization:
    def test_report_ranks_the_sacrificed_piece_for_each_method(self, tmp_path):
        with given("a tiny sacrifice checkpoint and one known sacrifice"):
            path = tmp_path / "sacrifice.pt"
            save_puzzle_checkpoint(path)
            handle = sacrifice_handle(path)
            boards = tuple(queen_sac_boards())
            puzzle = SolvedPuzzle(
                puzzle_id="sac",
                boards=boards,
                sacrificed_square=chess.D1,
                rating=1500,
            )

        with when("the piece-localization report is built"):
            report = piece_localization_report(
                handle, [puzzle], device="cpu", samples=16, background=[boards[0]]
            )

        with then("every method reports top-k and mean rank statistics"):
            assert set(report["methods"]) == {"occlusion", "shap"}
            assert report["positions"] == 1
            for entry in report["methods"].values():
                assert 0.0 <= entry["top1"] <= 1.0
                assert 0.0 <= entry["top3"] <= 1.0
                assert entry["mean_rank"] >= 1.0
        return


class TestFaithfulnessReport:
    def test_report_compares_each_method_against_random_deletion(self, tmp_path):
        with given("a tiny score checkpoint and two positions"):
            path = tmp_path / "score.pt"
            save_score_checkpoint(path)
            handle = score_handle(path)
            fens = [
                chess.STARTING_FEN,
                "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1",
            ]

        with when("the attribution report is built"):
            report = attribution_report(
                handle, fens, device="cpu", samples=16, random_trials=1, seed=0
            )

        with then("both methods report a faithfulness lift over random deletion"):
            assert set(report["methods"]) == {"occlusion", "shap"}
            assert report["positions"] == 2
            for entry in report["methods"].values():
                assert "faithfulness" in entry
                assert "random" in entry
                assert "lift" in entry
        return


class TestExplainCli:
    def test_cli_prints_square_attribution_as_json(self, tmp_path, capsys):
        with given("a tiny score checkpoint"):
            path = tmp_path / "score.pt"
            save_score_checkpoint(path)

        with when("the CLI explains the starting position"):
            code = explain_main(
                [
                    "--checkpoint",
                    str(path),
                    "--fen",
                    chess.STARTING_FEN,
                    "--method",
                    "occlusion",
                    "--top",
                    "3",
                    "--device",
                    "cpu",
                ]
            )
            payload = json.loads(capsys.readouterr().out)

        with then("the task is detected and the top squares are printed"):
            assert code == 0
            assert payload["task"] == "score"
            assert len(payload["squares"]) == 64
            assert len(payload["top"]) == 3
            assert payload["method"] == "occlusion"
        return

    def test_cli_detects_the_sacrifice_checkpoint(self, tmp_path, capsys):
        with given("a tiny sacrifice checkpoint"):
            path = tmp_path / "sacrifice.pt"
            save_puzzle_checkpoint(path)

        with when("the CLI explains the starting position"):
            code = explain_main(
                [
                    "--checkpoint",
                    str(path),
                    "--fen",
                    chess.STARTING_FEN,
                    "--device",
                    "cpu",
                    "--top",
                    "2",
                ]
            )
            payload = json.loads(capsys.readouterr().out)

        with then("the sacrifice task is detected"):
            assert code == 0
            assert payload["task"] == "sacrifice"
            assert payload["method"] == "occlusion"
            assert len(payload["top"]) == 2
        return
