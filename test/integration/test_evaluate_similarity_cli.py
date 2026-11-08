import json
from dataclasses import asdict

import chess
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch

from bdd import given, then, when
from src.dataset.token_stream import TokenStreamEncoder, pack_stream
from src.evaluation.observability.samples import load_puzzle_samples
from src.modeling.transformer import MaskedStreamTransformer, TransformerConfig
from src.run.evaluate_similarity import EvaluationConfig, run_evaluation

TINY_CONFIG = TransformerConfig(
    d_model=16,
    n_heads=4,
    n_layers=1,
    d_ff=32,
    dropout=0.0,
    max_seq_len=128,
    max_segments=8,
)


def write_checkpoint(path, seed=0, step=0, val_loss=1.0):
    torch.manual_seed(seed)
    model = MaskedStreamTransformer(TINY_CONFIG)
    torch.save(
        {
            "step": step,
            "model_state": model.state_dict(),
            "model_config": asdict(TINY_CONFIG),
            "val_loss": val_loss,
            "val_acc": 0.1,
        },
        path,
    )
    return


def write_test_shard(path):
    encoder = TokenStreamEncoder()
    board = chess.Board()
    segments = [encoder.encode(board)]
    board.push_san("e4")
    segments.append(encoder.encode(board))
    board.push_san("e5")
    segments.append(encoder.encode(board))
    table = pa.table(
        {
            "packed": pa.array(
                [
                    pack_stream(segments).tolist(),
                    pack_stream(segments).tolist(),
                    pack_stream(segments).tolist(),
                ],
                pa.list_(pa.int16()),
            ),
            "n_positions": pa.array([3, 3, 3], pa.int32()),
            "game_id": pa.array(["g1", "g2", "g3"], pa.string()),
        }
    )
    pq.write_table(table, path)
    return


def write_puzzle_shard(path):
    encoder = TokenStreamEncoder()
    board = chess.Board()
    segments = [encoder.encode(board)]
    board.push_san("e4")
    segments.append(encoder.encode(board))
    rows = [
        (f"p{index}", 1 if index % 2 == 0 else 0, 0 if index % 2 == 0 else 1)
        for index in range(12)
    ]
    table = pa.table(
        {
            "puzzle_id": pa.array([row[0] for row in rows], pa.string()),
            "packed": pa.array(
                [pack_stream(segments).tolist() for _ in rows], pa.list_(pa.int16())
            ),
            "rating": pa.array(
                [1200 + index for index in range(len(rows))], pa.int32()
            ),
            "fork": pa.array([row[1] for row in rows], pa.int8()),
            "pin": pa.array([row[2] for row in rows], pa.int8()),
        }
    )
    pq.write_table(table, path)
    return


def test_run_evaluation_creates_artifacts(tmp_path):
    with given("tiny checkpoints, test shards, and puzzle shards"):
        checkpoint = tmp_path / "tiny.pt"
        write_checkpoint(checkpoint, seed=0, step=10, val_loss=1.0)
        test_dir = tmp_path / "test"
        test_dir.mkdir()
        write_test_shard(test_dir / "batch_0000.parquet")
        puzzle_dir = tmp_path / "puzzles"
        puzzle_dir.mkdir()
        write_puzzle_shard(puzzle_dir / "batch_0000.parquet")
        out_dir = tmp_path / "out"
        config = EvaluationConfig(
            checkpoint=str(checkpoint),
            out_dir=str(out_dir),
            test_dir=str(test_dir),
            puzzles=str(puzzle_dir),
            themes=("fork", "pin"),
            n_test=6,
            n_puzzles=12,
            puzzle_position="first",
            per_game_cap=2,
            rank_k=5,
            rank_queries=20,
            rank_examples=2,
            rank_example_modes=("matched",),
            device="cpu",
            batch_size=8,
            seed=0,
        )

    with when("the evaluation runs end to end"):
        run_evaluation(config)

    with then("all artifacts and a valid report are written"):
        assert (out_dir / "report.html").exists()
        assert (out_dir / "metrics.json").exists()
        assert (out_dir / "samples.parquet").exists()
        assert (out_dir / "embeddings.npy").exists()
        assert (out_dir / "ranking_examples.json").exists()
        assert not (out_dir / "coords.parquet").exists()
        assert not (out_dir / "explanations.json").exists()
        assert not (out_dir / "comparison.json").exists()
        payload = json.loads((out_dir / "metrics.json").read_text())
        assert isinstance(payload, dict)
        assert payload["n_samples"] == 18
        assert any(key.startswith("motif_rank5_top1") for key in payload)
        assert any(key.startswith("test_temporal_rank5") for key in payload)
        assert any(key.startswith("surrogate_rank5") for key in payload)
        examples = json.loads((out_dir / "ranking_examples.json").read_text())
        assert set(examples) == {"test", "motifs", "surrogates"}
        test_buckets = examples["test"]["matched"]
        assert set(test_buckets) == {"best", "random", "worst"}
        test_examples = [row for rows in test_buckets.values() for row in rows]
        assert test_examples
        motif_examples = [
            row
            for theme_buckets in examples["motifs"]["matched"].values()
            for rows in theme_buckets.values()
            for row in rows
        ]
        assert motif_examples
        for example in test_examples + motif_examples:
            assert len(example["candidates"]) == 5
            assert (
                sum(candidate["is_positive"] for candidate in example["candidates"])
                == 1
            )
        html = (out_dir / "report.html").read_text()
        assert "Position similarity ranking" in html
        assert "Test: in-game vs out-of-game" in html
        assert "Motif labels" in html
        assert "Surrogate classes" in html
        assert "<details>" in html
    return


def test_evaluation_samples_keep_theme_labels(tmp_path):
    with given("a puzzle shard with a fork label"):
        encoder = TokenStreamEncoder()
        board = chess.Board()
        segment = encoder.encode(board)
        table = pa.table(
            {
                "puzzle_id": pa.array(["p1"], pa.string()),
                "packed": pa.array(
                    [pack_stream([segment]).tolist()], pa.list_(pa.int16())
                ),
                "rating": pa.array([1200], pa.int32()),
                "fork": pa.array([1], pa.int8()),
                "pin": pa.array([0], pa.int8()),
            }
        )
        pq.write_table(table, tmp_path / "puzzles.parquet")

    with when("the puzzle samples are loaded"):
        samples = load_puzzle_samples(
            tmp_path, n=1, themes=("fork",), position="first", seed=0
        )

    with then("the theme is attached to the sample"):
        assert len(samples) == 1
        assert samples[0].themes == ("fork",)
    return


pytestmark = pytest.mark.integration
