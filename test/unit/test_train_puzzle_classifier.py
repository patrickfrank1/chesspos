from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from unittest.mock import patch

import chess
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch

from bdd import given, then, when
from src.dataset.puzzle_variants import (
    EpochPlan,
    PuzzleRecord,
    PuzzleVariantDataset,
    collate_variants,
    load_puzzle_records,
    split_for_puzzle,
)
from src.dataset.token_stream import VOCAB_SIZE, TokenStreamEncoder, pack_stream
from src.modeling.puzzle_classifier import (
    MeanPoolingClassifier,
    MeanPoolingClassifierConfig,
)
from src.modeling.transformer import MaskedStreamTransformer, TransformerConfig
from src.run.classify_position import classify_fen, load_classifier
from src.run.train_puzzle_classifier import (
    build_validation_plan,
    train_epoch,
)
from src.modeling.puzzle_classifier import FrozenEncoderClassifier


def make_training_parquet(path: Path) -> None:
    encoder = TokenStreamEncoder()
    puzzle_ids = []
    packed = []
    labels = []
    for index in range(200):
        board = chess.Board()
        segments = []
        for uci in ("e2e4", "e7e5", "g1f3"):
            board.push_uci(uci)
            segments.append(encoder.encode(board))
        puzzle_ids.append(f"id-{index}")
        packed.append(pack_stream(segments).tolist())
        labels.append(index % 2)
    table = pa.table(
        {
            "puzzle_id": pa.array(puzzle_ids, type=pa.string()),
            "packed": pa.array(packed, type=pa.list_(pa.int16())),
            "sacrifice": pa.array(labels, type=pa.int8()),
        }
    )
    pq.write_table(table, path)
    return


def tiny_config() -> TransformerConfig:
    return TransformerConfig(
        vocab_size=VOCAB_SIZE,
        d_model=1,
        n_heads=1,
        n_layers=1,
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


def save_classifier_checkpoint(path: Path) -> None:
    config = tiny_config()
    encoder = MaskedStreamTransformer(config)
    head = MeanPoolingClassifier(MeanPoolingClassifierConfig(d_model=1))
    torch.save(
        {
            "encoder_config": asdict(config),
            "encoder_state": encoder.state_dict(),
            "head_config": asdict(head.config),
            "head_state": head.state_dict(),
        },
        path,
    )
    return


def make_record(puzzle_id: str, n_positions: int, label: int) -> PuzzleRecord:
    segments = [np.full(3, 10 + index, dtype=np.int16) for index in range(n_positions)]
    return PuzzleRecord(puzzle_id=puzzle_id, segments=segments, label=label)


class TestValidationPlan:
    def test_uses_every_single_position_with_the_sequence_label(self):
        with given("records with three and two positions"):
            records = [make_record("a", 3, 1), make_record("b", 2, 0)]

        with when("the validation plan is built"):
            plan = build_validation_plan(records)

        with then("each position appears once with its puzzle label"):
            assert len(plan) == 5
            assert plan.start.tolist() == [0, 1, 2, 0, 1]
            assert plan.end.tolist() == [0, 1, 2, 0, 1]
            assert plan.puzzle_index.tolist() == [0, 0, 0, 1, 1]
            assert plan.label.tolist() == [1, 1, 1, 0, 0]
        return


class TestLoadRecords:
    def test_loads_packed_sequences_and_theme_labels(self, tmp_path):
        with given("a parquet shard with packed sequences and a sacrifice column"):
            path = tmp_path / "part.parquet"
            make_training_parquet(path)

        with when("the records are loaded"):
            records = load_puzzle_records(tmp_path, "sacrifice")

        with then("each row becomes one labeled puzzle record"):
            assert len(records) == 200
            assert records[0].puzzle_id == "id-0"
            assert records[0].label == 0
            assert len(records[0].segments) == 3
        return

    def test_limit_stops_after_the_requested_number_of_records(self, tmp_path):
        with given("a parquet shard with 200 records"):
            path = tmp_path / "part.parquet"
            make_training_parquet(path)

        with when("the loader is limited to 7 records"):
            records = load_puzzle_records(tmp_path, "sacrifice", limit=7)

        with then("only the requested number is returned"):
            assert len(records) == 7
        return


class TestStepLogging:
    def test_train_epoch_logs_every_n_steps(self, capsys):
        with given("a tiny frozen-encoder model and a plan of eight samples"):
            encoder = MaskedStreamTransformer(tiny_config()).eval()
            head = MeanPoolingClassifier(MeanPoolingClassifierConfig(d_model=1))
            model = FrozenEncoderClassifier(encoder, head)
            records = [make_record(f"p-{i}", 2, i % 2) for i in range(8)]
            plan = EpochPlan(
                puzzle_index=np.arange(8, dtype=np.int32),
                start=np.zeros(8, dtype=np.int16),
                end=np.zeros(8, dtype=np.int16),
                label=np.array([i % 2 for i in range(8)], dtype=np.int8),
            )
            loader = torch.utils.data.DataLoader(
                PuzzleVariantDataset(records, plan),
                batch_size=2,
                collate_fn=collate_variants,
            )
            optimizer = torch.optim.AdamW(model.head.parameters(), lr=1e-3)

        with when("one epoch trains with log_every=2 on epoch 3"):
            train_epoch(
                model,
                loader,
                optimizer,
                torch.device("cpu"),
                pos_weight=None,
                epoch=3,
                log_every=2,
            )

        with then("a log line is printed every two steps with running loss"):
            output = capsys.readouterr().out
            assert output.count("epoch=3 step=") == 2
            assert "step=2/4" in output
            assert "step=4/4" in output
            assert "loss=" in output
        return

    def test_train_epoch_does_not_log_when_log_every_is_zero(self, capsys):
        with given("a tiny frozen-encoder model and a plan of four samples"):
            encoder = MaskedStreamTransformer(tiny_config()).eval()
            head = MeanPoolingClassifier(MeanPoolingClassifierConfig(d_model=1))
            model = FrozenEncoderClassifier(encoder, head)
            records = [make_record(f"p-{i}", 2, i % 2) for i in range(4)]
            plan = EpochPlan(
                puzzle_index=np.arange(4, dtype=np.int32),
                start=np.zeros(4, dtype=np.int16),
                end=np.zeros(4, dtype=np.int16),
                label=np.array([0, 1, 0, 1], dtype=np.int8),
            )
            loader = torch.utils.data.DataLoader(
                PuzzleVariantDataset(records, plan),
                batch_size=2,
                collate_fn=collate_variants,
            )
            optimizer = torch.optim.AdamW(model.head.parameters(), lr=1e-3)

        with when("one epoch trains with log_every=0"):
            train_epoch(
                model,
                loader,
                optimizer,
                torch.device("cpu"),
                pos_weight=None,
                epoch=1,
                log_every=0,
            )

        with then("no step log lines are printed"):
            assert capsys.readouterr().out == ""
        return


class TestTrainingSmoke:
    def test_training_writes_a_loadable_classifier_checkpoint(self, tmp_path):
        with given("a tiny encoder, a small labeled shard and both data splits"):
            save_encoder_checkpoint(tmp_path / "encoder.pt")
            make_training_parquet(tmp_path / "part.parquet")
            ids = [f"id-{index}" for index in range(200)]
            assert any(split_for_puzzle(i, 0.2) == "train" for i in ids)
            assert any(split_for_puzzle(i, 0.2) == "val" for i in ids)
            argv = [
                "train_puzzle_classifier.py",
                "--data-dir",
                str(tmp_path),
                "--theme",
                "sacrifice",
                "--checkpoint",
                str(tmp_path / "encoder.pt"),
                "--output-dir",
                str(tmp_path / "out"),
                "--epochs",
                "1",
                "--batch-size",
                "8",
                "--num-workers",
                "0",
                "--device",
                "cpu",
                "--val-ratio",
                "0.2",
            ]

        with when("the training entry point runs"):
            import src.run.train_puzzle_classifier as cli

            with patch("sys.argv", argv):
                exit_code = cli.main()

        with then("best.pt exists and loads as an inference classifier"):
            assert exit_code == 0
            checkpoint = tmp_path / "out" / "best.pt"
            assert checkpoint.exists()
            model = load_classifier(str(checkpoint), torch.device("cpu"))
            probability = classify_fen(model, chess.STARTING_FEN, torch.device("cpu"))
            assert 0.0 <= probability <= 1.0
        return
