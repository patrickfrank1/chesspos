from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import chess
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch

from bdd import given, then, when
from src.dataset.game_outcomes import (
    GameOutcome,
    ScoreDataset,
    ScorePlan,
)
from src.dataset.puzzle_variants import pack_segments
from src.dataset.token_stream import VOCAB_SIZE, TokenStreamEncoder
from src.modeling.score_head import (
    FrozenEncoderScore,
    ScoreHead,
    ScoreHeadConfig,
)
from src.modeling.transformer import MaskedStreamTransformer, TransformerConfig
from src.run.eval_score_probe import load_score_probe
from src.run.eval_score_probe import main as eval_main
from src.run.train_score_probe import (
    build_optimizer,
    build_score_loader,
    evaluate,
    main,
    parse_args,
    save_checkpoint,
    train_epoch,
)


def tiny_config() -> TransformerConfig:
    return TransformerConfig(
        vocab_size=VOCAB_SIZE,
        d_model=1,
        n_heads=1,
        n_layers=2,
        d_ff=2,
        dropout=0.0,
        max_seq_len=512,
        max_segments=8,
    )


def save_encoder_checkpoint(path: Path) -> None:
    config = tiny_config()
    encoder = MaskedStreamTransformer(config)
    torch.save(
        {"model_config": asdict(config), "model_state": encoder.state_dict()},
        path,
    )
    return


def make_records(n_games: int, n_positions: int = 4) -> list[GameOutcome]:
    results = ["1-0", "0-1", "1/2-1/2"]
    return [
        GameOutcome(
            game_id=f"g{index}",
            segments=[
                np.full(67, 10 + index, dtype=np.int16) for index in range(n_positions)
            ],
            result=results[index % len(results)],
            white_elo=2000,
            black_elo=2000,
        )
        for index in range(n_games)
    ]


def make_plan(n_games: int, n_positions: int = 4) -> ScorePlan:
    game_index = np.repeat(np.arange(n_games, dtype=np.int32), n_positions)
    ply = np.tile(np.arange(1, n_positions + 1, dtype=np.int16), n_games)
    return ScorePlan(game_index=game_index, ply=ply)


def make_score_parquet(path: Path, n_games: int) -> None:
    encoder = TokenStreamEncoder()
    board = chess.Board()
    segments = []
    for uci in ("e2e4", "e7e5", "g1f3"):
        board.push_uci(uci)
        segments.append(encoder.encode(board))
    packed = pack_segments(segments)[0].tolist()
    results = ["1-0", "0-1", "1/2-1/2"]
    table = pa.table(
        {
            "packed": pa.array([packed] * n_games, pa.list_(pa.int16())),
            "n_positions": pa.array([3] * n_games, pa.int32()),
            "game_id": pa.array([f"g{index}" for index in range(n_games)], pa.string()),
            "white_elo": pa.array([2000] * n_games, pa.int32()),
            "black_elo": pa.array([1980] * n_games, pa.int32()),
            "result": pa.array([results[index % 3] for index in range(n_games)]),
        }
    )
    pq.write_table(table, path)
    return


def make_model(feature: str = "native", state_features: int = 0) -> FrozenEncoderScore:
    encoder = MaskedStreamTransformer(tiny_config()).eval()
    head = ScoreHead(ScoreHeadConfig(d_model=1, state_features=state_features))
    return FrozenEncoderScore(encoder, head, feature=feature)


class TestParseArgs:
    def test_defaults_target_the_frozen_native_recipe(self):
        with given("only the required checkpoint argument"):
            argv: list[str] = ["--checkpoint", "encoder.pt"]

        with when("the arguments are parsed"):
            args = parse_args(argv)

        with then("the native embed and MSE loss are the defaults"):
            assert args.feature == "native"
            assert args.loss == "mse"
            assert args.unfreeze_last_block is False
            assert args.pooling == "mean"
            assert args.positions_per_game == 16
            assert args.max_ply == 150
        return

    def test_exposes_the_alternative_feature_and_unfreeze_flags(self):
        with given("arguments for the alternative experiment arm"):
            argv = [
                "--checkpoint",
                "encoder.pt",
                "--feature",
                "full",
                "--unfreeze-last-block",
                "--loss",
                "huber",
                "--include-state-features",
            ]

        with when("the arguments are parsed"):
            args = parse_args(argv)

        with then("the alternative settings are accepted"):
            assert args.feature == "full"
            assert args.unfreeze_last_block is True
            assert args.loss == "huber"
            assert args.include_state_features is True
        return

    def test_defaults_to_a_separate_small_encoder_lr(self):
        with given("the default arguments"):
            args = parse_args(["--checkpoint", "encoder.pt"])

        with when("the encoder learning rate is inspected"):
            encoder_lr = args.encoder_lr

        with then("the encoder LR is much smaller than the head LR"):
            assert encoder_lr == 1e-5
            assert args.lr == 1e-3
        return

    def test_defaults_to_no_autocast_and_accepts_bf16(self):
        with given("default and bf16 argument lists"):
            default = parse_args(["--checkpoint", "encoder.pt"])
            mixed = parse_args(["--checkpoint", "encoder.pt", "--amp", "bf16"])

        with when("both are parsed"):
            default_amp = default.amp
            mixed_amp = mixed.amp

        with then("autocast is opt-in"):
            assert default_amp == "none"
            assert mixed_amp == "bf16"
        return


class TestTrainEpoch:
    def test_updates_only_the_head_and_reports_a_loss(self):
        with given("a tiny frozen model and a plan of eight samples"):
            records = make_records(2)
            loader = build_score_loader(
                ScoreDataset(records, make_plan(2)),
                batch_size=2,
                num_workers=0,
                device=torch.device("cpu"),
            )
            model = make_model()
            optimizer = torch.optim.AdamW(model.head.parameters(), lr=1e-2)
            before = model.head.classifier.weight.detach().clone()

        with when("one epoch is trained"):
            loss = train_epoch(
                model,
                loader,
                optimizer,
                torch.device("cpu"),
                epoch=1,
                log_every=1,
            )

        with then("the head changes and the loss is finite"):
            assert np.isfinite(loss)
            assert not torch.allclose(before, model.head.classifier.weight)
            assert all(p.grad is None for p in model.encoder.parameters())
        return

    def test_logs_every_n_steps(self, capsys):
        with given("a loader with two batches and log_every=1"):
            records = make_records(2)
            loader = build_score_loader(
                ScoreDataset(records, make_plan(2, n_positions=2)),
                batch_size=2,
                num_workers=0,
                device=torch.device("cpu"),
            )
            model = make_model()
            optimizer = torch.optim.AdamW(model.head.parameters(), lr=1e-3)

        with when("one epoch is trained"):
            train_epoch(
                model,
                loader,
                optimizer,
                torch.device("cpu"),
                epoch=3,
                log_every=1,
            )

        with then("a log line is printed per step"):
            output = capsys.readouterr().out
            assert "epoch=3 step=1/2" in output
            assert "epoch=3 step=2/2" in output
            assert "loss=" in output
        return


class TestEvaluate:
    def test_returns_score_metrics_and_mse_loss(self):
        with given("a frozen model and a validation loader"):
            records = make_records(2)
            loader = build_score_loader(
                ScoreDataset(records, make_plan(2)),
                batch_size=2,
                num_workers=0,
                device=torch.device("cpu"),
            )
            model = make_model()

        with when("the model is evaluated"):
            metrics, loss = evaluate(model, loader, torch.device("cpu"))

        with then("the metric keys and a finite loss are returned"):
            assert {"mse", "mae", "r2", "pearson", "spearman"} <= set(metrics)
            assert metrics["n"] == 8
            assert np.isfinite(loss)
            assert loss == pytest.approx(metrics["mse"])
        return


class TestCheckpoint:
    def test_round_trips_encoder_and_head_state(self, tmp_path):
        with given("a trained tiny model"):
            model = make_model(feature="full")
            path = tmp_path / "best.pt"

        with when("the checkpoint is saved"):
            save_checkpoint(
                path,
                model,
                epoch=2,
                feature="full",
                unfreeze_last_block=False,
                metrics={"mse": 0.25},
                window=3,
                include_state_features=True,
            )

        with then("the checkpoint holds both states and the recipe"):
            checkpoint = torch.load(path, map_location="cpu", weights_only=False)
            assert checkpoint["feature"] == "full"
            assert checkpoint["epoch"] == 2
            assert checkpoint["metrics"] == {"mse": 0.25}
            assert checkpoint["window"] == 3
            assert checkpoint["include_state_features"] is True
            assert "encoder_config" in checkpoint
            assert "head_config" in checkpoint
            model.encoder.load_state_dict(checkpoint["encoder_state"])
            model.head.load_state_dict(checkpoint["head_state"])
        return


class TestMain:
    def test_runs_end_to_end_and_writes_artifacts(self, tmp_path):
        with given("a shard of games and a tiny encoder checkpoint"):
            make_score_parquet(tmp_path / "games.parquet", n_games=20)
            encoder_path = tmp_path / "encoder.pt"
            save_encoder_checkpoint(encoder_path)
            output_dir = tmp_path / "run"

        with when("the training CLI runs for one epoch"):
            exit_code = main(
                [
                    "--data-dir",
                    str(tmp_path),
                    "--checkpoint",
                    str(encoder_path),
                    "--output-dir",
                    str(output_dir),
                    "--epochs",
                    "1",
                    "--batch-size",
                    "4",
                    "--num-workers",
                    "0",
                    "--device",
                    "cpu",
                    "--positions-per-game",
                    "2",
                    "--val-ratio",
                    "0.5",
                    "--log-every",
                    "0",
                ]
            )

        with then("metrics, predictions, and a checkpoint are written"):
            assert exit_code == 0
            assert (output_dir / "metrics.json").exists()
            assert (output_dir / "predictions.npz").exists()
            assert (output_dir / "best.pt").exists()
        return


class TestEvalScoreProbe:
    def test_loads_encoder_head_and_recipe_from_a_probe_checkpoint(self, tmp_path):
        with given("a saved probe checkpoint using the full recipe"):
            model = make_model(feature="full", state_features=3)
            path = tmp_path / "probe.pt"
            save_checkpoint(
                path,
                model,
                epoch=1,
                feature="full",
                unfreeze_last_block=False,
                metrics={"mse": 0.2},
                window=2,
                include_state_features=True,
            )

        with when("the probe is loaded for evaluation"):
            loaded = load_score_probe(path, torch.device("cpu"))

        with then("the recipe and weights are restored"):
            assert loaded.feature == "full"
            assert loaded.unfreeze_last_block is False
            assert loaded.head.config.state_features == 3
            torch.testing.assert_close(
                loaded.head.classifier.weight, model.head.classifier.weight
            )
        return

    def test_writes_metrics_and_predictions_for_a_game_shard(self, tmp_path):
        with given("a games shard and a probe checkpoint"):
            pd = tmp_path
            make_score_parquet(pd / "games.parquet", n_games=8)
            model = make_model()
            probe_path = pd / "probe.pt"
            save_checkpoint(
                probe_path,
                model,
                epoch=1,
                feature="native",
                unfreeze_last_block=False,
                metrics={"mse": 0.2},
                window=1,
                include_state_features=False,
            )
            output_dir = pd / "eval"

        with when("the evaluation CLI runs"):
            exit_code = eval_main(
                [
                    "--checkpoint",
                    str(probe_path),
                    "--data-dir",
                    str(pd),
                    "--output-dir",
                    str(output_dir),
                    "--batch-size",
                    "4",
                    "--num-workers",
                    "0",
                    "--device",
                    "cpu",
                    "--amp",
                    "none",
                ]
            )

        with then("metrics and per-position predictions are written"):
            assert exit_code == 0
            metrics = json.loads((output_dir / "metrics.json").read_text())
            assert {"mse", "r2", "n"} <= set(metrics["metrics"])
            assert metrics["games"] == 8
            predictions = np.load(output_dir / "predictions.npz")
            assert predictions["predictions"].shape == predictions["targets"].shape
            assert predictions["sources"].shape[0] == predictions["targets"].shape[0]
        return

    def test_rejects_a_missing_checkpoint(self, tmp_path):
        with given("a path that does not exist"):
            path = tmp_path / "missing.pt"

        with when("the probe is loaded"):
            error = None
            try:
                load_score_probe(path, torch.device("cpu"))
            except FileNotFoundError as exc:
                error = exc

        with then("a FileNotFoundError is raised"):
            assert isinstance(error, FileNotFoundError)
        return


class TestOptimizer:
    def test_frozen_model_has_a_single_head_group(self):
        with given("a frozen probe with no trainable encoder parameters"):
            model = make_model()

        with when("the optimizer is built"):
            optimizer = build_optimizer(
                model, lr=1e-3, encoder_lr=1e-5, weight_decay=0.01
            )

        with then("only the head parameters are optimized at the head LR"):
            assert len(optimizer.param_groups) == 1
            assert optimizer.param_groups[0]["lr"] == 1e-3
            group_params = {id(p) for p in optimizer.param_groups[0]["params"]}
            assert group_params == {
                id(p) for p in model.head.parameters() if p.requires_grad
            }
        return

    def test_unfrozen_encoder_gets_its_own_group(self):
        with given("a probe with the last block unfrozen"):
            encoder = MaskedStreamTransformer(tiny_config())
            head = ScoreHead(ScoreHeadConfig(d_model=1))
            model = FrozenEncoderScore(
                encoder, head, feature="full", unfreeze_last_block=True
            )

        with when("the optimizer is built"):
            optimizer = build_optimizer(
                model, lr=1e-3, encoder_lr=1e-5, weight_decay=0.0
            )

        with then("the encoder uses the small LR and the head the large one"):
            assert len(optimizer.param_groups) == 2
            lrs = sorted(group["lr"] for group in optimizer.param_groups)
            assert lrs == [1e-5, 1e-3]
        return
