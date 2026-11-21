from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import chess
import torch
from fastapi.testclient import TestClient

from bdd import given, then, when
from src.dataset.token_stream import VOCAB_SIZE
from src.chess_eval.app import create_app
from src.chess_eval.backend import (
    apply_move,
    discover_models,
    evaluate_models,
    position_payload,
    replay,
)
from src.modeling.puzzle_classifier import PoolingClassifier, PoolingClassifierConfig
from src.modeling.score_head import ScoreHead, ScoreHeadConfig
from src.modeling.transformer import MaskedStreamTransformer, TransformerConfig


def tiny_config() -> TransformerConfig:
    return TransformerConfig(
        vocab_size=VOCAB_SIZE,
        d_model=1,
        n_heads=1,
        n_layers=1,
        d_ff=2,
        dropout=0.0,
        max_seq_len=256,
        max_segments=8,
    )


def save_score_checkpoint(path: Path) -> None:
    config = tiny_config()
    encoder = MaskedStreamTransformer(config)
    head = ScoreHead(
        ScoreHeadConfig(d_model=1, hidden_dim=2, pooling="mean", state_features=0)
    )
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
            "window": 1,
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
        PoolingClassifierConfig(d_model=1, hidden_dim=2, pooling="mean")
    )
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


class TestModelDiscovery:
    def test_finds_best_checkpoints_and_labels_them(self, tmp_path):
        with given("a score run and a sacrifice run on disk"):
            score = tmp_path / "score" / "unfreeze_all_mean" / "run_20m_lazy"
            score = score / "seed17"
            score.mkdir(parents=True)
            save_score_checkpoint(score / "best.pt")
            (score / "last.pt").write_bytes(b"not a checkpoint")
            sacrifice = (
                tmp_path
                / "sacrifice"
                / "unfreeze_all_attn"
                / "run_6m_isolated"
                / "seed17"
            )
            sacrifice.mkdir(parents=True)
            save_puzzle_checkpoint(sacrifice / "best.pt")

        with when("the model roots are scanned"):
            models = discover_models([tmp_path])

        with then("only the best checkpoint of each run is listed"):
            assert len(models) == 2
            assert {model.task for model in models} == {"score", "sacrifice"}
            score_model = next(model for model in models if model.task == "score")
            assert "20M" in score_model.label
            assert "full FT mean" in score_model.label
            assert score_model.recipe == "unfreeze_all_mean"
            assert score_model.seed == 17
            assert score_model.path.name == "best.pt"
            assert "6M" in next(
                model.label for model in models if model.task == "sacrifice"
            )
        return


class TestMoveReplay:
    def test_replays_legal_moves_and_reports_state(self):
        with given("the starting position"):
            start_fen = chess.STARTING_FEN

        with when("e2e4 is replayed"):
            boards, san_moves = replay(start_fen, ["e2e4"])
            payload = position_payload(start_fen, ["e2e4"], boards, san_moves)

        with then("the board advances and the legal moves are exposed"):
            assert payload["turn"] == "black"
            assert payload["moves"] == ["e2e4"]
            assert payload["san_moves"] == ["e4"]
            assert payload["fen"] != start_fen
            assert "e7e5" in payload["legal_moves"]
            assert payload["material_balance"] == 0
        return

    def test_rejects_an_illegal_move(self):
        with given("the starting position"):
            start_fen = chess.STARTING_FEN

        with when("an illegal move is applied"):
            error = None
            try:
                apply_move(start_fen, [], "e2", "e5", None)
            except ValueError as exc:
                error = exc

        with then("the move is rejected"):
            assert isinstance(error, ValueError)
        return


class TestModelEvaluation:
    def test_score_model_returns_a_white_pov_value(self, tmp_path):
        with given("a tiny score checkpoint and the starting board"):
            path = tmp_path / "score" / "full" / "run_20m_lazy" / "seed17"
            path.mkdir(parents=True)
            save_score_checkpoint(path / "best.pt")
            models = discover_models([tmp_path])
            boards, _ = replay(chess.STARTING_FEN, [])

        with when("the position is evaluated"):
            results = evaluate_models(models, boards, torch.device("cpu"))

        with then("the score is bounded and reports its metrics"):
            assert len(results) == 1
            result = results[0]
            assert result["task"] == "score"
            assert -1.0 <= result["value"] <= 1.0
            assert result["expected_points"] == (result["value"] + 1.0) / 2.0
            assert result["metrics"]["r2"] == 0.1
        return

    def test_puzzle_model_returns_a_probability(self, tmp_path):
        with given("a tiny sacrifice checkpoint and the starting board"):
            path = tmp_path / "sacrifice" / "full" / "run_6m_isolated" / "seed17"
            path.mkdir(parents=True)
            save_puzzle_checkpoint(path / "best.pt")
            models = discover_models([tmp_path])
            boards, _ = replay(chess.STARTING_FEN, [])

        with when("the position is evaluated"):
            results = evaluate_models(models, boards, torch.device("cpu"))

        with then("the output is a probability"):
            assert len(results) == 1
            result = results[0]
            assert result["task"] == "sacrifice"
            assert 0.0 <= result["value"] <= 1.0
            assert result["percent"] == result["value"] * 100.0
            assert result["above_threshold"] == (result["value"] >= 0.5)
        return


class TestHttpApi:
    def test_position_and_move_endpoints(self, tmp_path):
        with given("an app with a tiny score checkpoint"):
            path = tmp_path / "score" / "full" / "run_20m_lazy" / "seed17"
            path.mkdir(parents=True)
            save_score_checkpoint(path / "best.pt")
            app = create_app(model_roots=[tmp_path], device="cpu")
            client = TestClient(app)

        with when("the model list and starting position are requested"):
            models = client.get("/api/models").json()["models"]
            model_id = models[0]["model_id"]
            response = client.post(
                "/api/position",
                json={
                    "start_fen": chess.STARTING_FEN,
                    "moves": [],
                    "model_ids": [model_id],
                },
            )

        with then("the position and evaluation are returned"):
            assert response.status_code == 200
            payload = response.json()
            assert payload["position"]["turn"] == "white"
            assert payload["evaluations"][0]["task"] == "score"

        with when("a legal move is submitted"):
            response = client.post(
                "/api/move",
                json={
                    "start_fen": chess.STARTING_FEN,
                    "moves": [],
                    "source": "e2",
                    "target": "e4",
                    "promotion": None,
                    "model_ids": [model_id],
                },
            )

        with then("the returned position includes the move"):
            assert response.status_code == 200
            assert response.json()["position"]["moves"] == ["e2e4"]
        return

    def test_cross_origin_isolation_headers_are_set(self, tmp_path):
        with given("an app serving the board page"):
            app = create_app(model_roots=[tmp_path], device="cpu")
            client = TestClient(app)

        with when("the root page is requested"):
            response = client.get("/")

        with then("the headers required by threaded WASM are present"):
            assert response.headers["cross-origin-opener-policy"] == "same-origin"
            assert response.headers["cross-origin-embedder-policy"] == "require-corp"
            assert response.headers["cross-origin-resource-policy"] == "same-origin"
        return

    def test_stockfish_assets_are_served(self, tmp_path):
        with given("an app with the vendored Stockfish build"):
            app = create_app(model_roots=[tmp_path], device="cpu")
            client = TestClient(app)

        with when("the engine script and WASM are requested"):
            script = client.get("/static/stockfish/sf_19.js")
            wasm = client.get("/static/stockfish/sf_19.wasm")

        with then("both assets are available with the expected media types"):
            assert script.status_code == 200
            assert script.headers["content-type"].startswith(
                ("text/javascript", "application/javascript")
            )
            assert wasm.status_code == 200
            assert wasm.headers["content-type"] == "application/wasm"
        return

    def test_stockfish_nnue_is_served_from_the_cache(self, tmp_path):
        with given("a cached NNUE file"):
            nnue_dir = tmp_path / "nnue"
            nnue_dir.mkdir()
            (nnue_dir / "nn-1a298aa575a0.nnue").write_bytes(b"fake-nnue")
            app = create_app(
                model_roots=[tmp_path / "models"],
                device="cpu",
                nnue_cache_dir=nnue_dir,
            )
            client = TestClient(app)

        with when("the NNUE endpoint is requested"):
            response = client.get("/api/stockfish/nnue")
            info = client.get("/api/stockfish/info").json()

        with then("the cached bytes and engine metadata are returned"):
            assert response.status_code == 200
            assert response.content == b"fake-nnue"
            assert info["engine"] == "Stockfish 19"
            assert info["cached"] is True
            assert info["nnue"] == "nn-1a298aa575a0.nnue"
        return

    def test_index_page_is_served(self, tmp_path):
        with given("an app with no checkpoints"):
            app = create_app(model_roots=[tmp_path], device="cpu")
            client = TestClient(app)

        with when("the root page is requested"):
            response = client.get("/")

        with then("the board page is returned"):
            assert response.status_code == 200
            assert "position evaluator" in response.text
        return

    def test_illegal_move_returns_bad_request(self, tmp_path):
        with given("an app with no models"):
            app = create_app(model_roots=[tmp_path], device="cpu")
            client = TestClient(app)

        with when("an illegal move is submitted"):
            response = client.post(
                "/api/move",
                json={
                    "start_fen": chess.STARTING_FEN,
                    "moves": [],
                    "source": "e2",
                    "target": "e5",
                    "promotion": None,
                    "model_ids": [],
                },
            )

        with then("the API reports a client error"):
            assert response.status_code == 400
        return
