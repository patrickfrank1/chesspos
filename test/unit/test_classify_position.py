from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import chess
import torch

from bdd import given, then, when
from src.dataset.token_stream import VOCAB_SIZE
from src.modeling.puzzle_classifier import (
    MeanPoolingClassifier,
    MeanPoolingClassifierConfig,
)
from src.modeling.transformer import MaskedStreamTransformer, TransformerConfig
from src.run.classify_position import (
    classify_fen,
    classify_pgn,
    load_classifier,
    score_positions,
)


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


def write_pgn(path: Path) -> None:
    path.write_text(
        '[Event "Test"]\n[Site "Local"]\n[Result "*"]\n\n1. e4 e5 2. Nf3 Nc6 3. Bb5 *\n'
    )
    return


class TestSinglePositionInference:
    def test_fen_inference_returns_a_probability(self, tmp_path):
        with given("a tiny classifier checkpoint and a FEN"):
            checkpoint = tmp_path / "classifier.pt"
            save_classifier_checkpoint(checkpoint)
            model = load_classifier(str(checkpoint), torch.device("cpu"))

        with when("the FEN is scored"):
            probability = classify_fen(model, chess.STARTING_FEN, torch.device("cpu"))

        with then("the output is a probability"):
            assert 0.0 <= probability <= 1.0
        return

    def test_window_mode_scores_every_position(self, tmp_path):
        with given("a tiny classifier and a short game"):
            checkpoint = tmp_path / "classifier.pt"
            save_classifier_checkpoint(checkpoint)
            model = load_classifier(str(checkpoint), torch.device("cpu"))
            board = chess.Board()
            boards = [board.copy()]
            for uci in ("e2e4", "e7e5", "g1f3"):
                board.push_uci(uci)
                boards.append(board.copy())

        with when("the positions are scored with a history window"):
            scores = score_positions(model, boards, torch.device("cpu"), window=2)

        with then("one probability per position is returned"):
            assert len(scores) == len(boards)
            assert all(0.0 <= score <= 1.0 for score in scores)
        return

    def test_pgn_mode_scores_the_start_and_every_move(self, tmp_path):
        with given("a tiny classifier and a three-move PGN"):
            checkpoint = tmp_path / "classifier.pt"
            save_classifier_checkpoint(checkpoint)
            model = load_classifier(str(checkpoint), torch.device("cpu"))
            pgn_path = tmp_path / "game.pgn"
            write_pgn(pgn_path)

        with when("the game is scored"):
            positions = classify_pgn(model, pgn_path, torch.device("cpu"), window=1)

        with then("the start position and all five plies are present"):
            assert len(positions) == 6
            assert positions[0]["ply"] == 0
            assert positions[0]["move"] is None
            assert positions[1]["move"] == "e2e4"
            assert positions[-1]["move"] == "f1b5"
            assert all(0.0 <= row["probability"] <= 1.0 for row in positions)
        return
