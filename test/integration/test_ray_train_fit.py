import random
from pathlib import Path

import chess
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import ray

from bdd import given, then, when
from src.dataset.token_stream import TokenStreamEncoder, pack_stream
from src.run.train_ray import RayTrainingConfig, build_trainer

REPO_ROOT = Path(__file__).resolve().parents[2]


def write_split(directory: Path, n_games: int) -> None:
    """Write one parquet shard of packed token streams for n random games."""
    directory.mkdir(parents=True)
    encoder = TokenStreamEncoder()
    packed = []
    for seed in range(n_games):
        rng = random.Random(seed)
        board = chess.Board()
        segments = [encoder.encode(board)]
        while len(segments) < 8:
            moves = list(board.legal_moves)
            if not moves:
                break
            board.push(moves[rng.randrange(len(moves))])
            if board.is_game_over():
                break
            segments.append(encoder.encode(board))
        packed.append(pack_stream(segments).tolist())
    table = pa.table({"packed": pa.array(packed, type=pa.list_(pa.int64()))})
    pq.write_table(table, directory / "batch_0000.parquet")
    return


class TestRayTrainFit:
    def test_end_to_end_fit_two_steps(self, tmp_path, monkeypatch):
        with given("a tiny local dataset split into train and test parquet shards"):
            train_dir = tmp_path / "train"
            val_dir = tmp_path / "test"
            write_split(train_dir, n_games=4)
            write_split(val_dir, n_games=2)
            config = RayTrainingConfig(
                train_dir=str(train_dir),
                val_dir=str(val_dir),
                steps=2,
                batch_size=2,
                d_model=16,
                n_layers=1,
                n_heads=2,
                d_ff=32,
                max_seq_len=64,
                max_segments=8,
                dataloader_workers=0,
                log_every=1,
                eval_every=100,
                save_every=1,
                full_eval=False,
                mlflow_uri="",
                seed=3,
                use_gpu=False,
                num_training_workers=1,
                storage_path=str(tmp_path / "results"),
            )

        with when("the TorchTrainer runs two training steps on one CPU worker"):
            # Keep Ray workers on the test venv (which has torch) instead of
            # letting the uv-run hook build a fresh base-dependency venv.
            # The constant is read at import time, so patch it directly.
            monkeypatch.setattr(
                ray._private.ray_constants, "RAY_ENABLE_UV_RUN_RUNTIME_ENV", False
            )
            ray.init(
                ignore_reinit_error=True,
                runtime_env={
                    "working_dir": str(REPO_ROOT),
                    "excludes": [".venv", ".git", "data", "ray_results"],
                },
            )
            try:
                trainer = build_trainer(config)
                result = trainer.fit()
            finally:
                ray.shutdown()

        with then("the run reports the final step and a checkpoint"):
            assert result.metrics["step"] == 2
            assert result.checkpoint is not None
        return


pytestmark = pytest.mark.integration
