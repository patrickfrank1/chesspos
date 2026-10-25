from types import SimpleNamespace

import chess
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from bdd import given, then, when
from src.dataset.token_stream import TokenStreamEncoder, pack_stream
from src.training import ray_train
from src.training.ray_train import RayTrainingConfig, train_loop_per_worker


def _write_split(directory, games: int = 4, plies: int = 30) -> None:
    encoder = TokenStreamEncoder()
    rows = []
    for seed in range(games):
        rng = np.random.default_rng(seed)
        board = chess.Board()
        segments = []
        for _ in range(plies):
            moves = list(board.legal_moves)
            board.push(moves[int(rng.integers(len(moves)))])
            segments.append(encoder.encode(board))
        rows.append(pack_stream(segments).tolist())
    table = pa.table({"packed": pa.array(rows, type=pa.list_(pa.int64()))})
    directory.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, directory / "batch_0000.parquet")
    return


def _fake_train(reports: list[dict]) -> SimpleNamespace:
    class Context:
        def get_world_rank(self) -> int:
            return 0

        def get_world_size(self) -> int:
            return 1

    def report(metrics, checkpoint=None):
        reports.append(metrics)

    return SimpleNamespace(
        get_context=lambda: Context(),
        get_checkpoint=lambda: None,
        report=report,
        torch=SimpleNamespace(
            prepare_model=lambda model: model,
            prepare_data_loader=lambda loader: loader,
        ),
    )


def test_reported_train_loss_reflects_the_interval_at_a_save_step(tmp_path):
    with given("a tiny local train/val split and a fake single-worker Ray context"):
        train_dir = tmp_path / "train"
        val_dir = tmp_path / "val"
        _write_split(train_dir)
        _write_split(val_dir)
        reports: list[dict] = []
        fake_train = _fake_train(reports)
        original_train = ray_train.train
        ray_train.train = fake_train
        config = RayTrainingConfig(
            train_dir=str(train_dir),
            val_dir=str(val_dir),
            steps=4,
            batch_size=1,
            d_model=16,
            n_layers=1,
            n_heads=2,
            d_ff=32,
            dropout=0.0,
            max_seq_len=256,
            max_segments=8,
            mlm_weight=1.0,
            inv_weight=0.0,
            var_weight=0.0,
            cov_weight=0.0,
            nce_weight=0.0,
            dataloader_workers=0,
            log_every=2,
            eval_every=1_000,
            eval_batches=1,
            token_rank_tokens=64,
            save_every=4,
            full_eval=False,
            freq_weighting=False,
            bf16=False,
            use_gpu=False,
            mlflow_uri="",
            checkpoint_dir=str(tmp_path / "checkpoints"),
            seed=0,
        )

    try:
        with when("the worker loop reaches the final save step (also a log step)"):
            train_loop_per_worker(config.to_dict())
    finally:
        ray_train.train = original_train

    with then("the reported train loss is the latest interval average, not zero"):
        assert reports, "the loop did not report any metrics"
        assert reports[-1]["step"] == 4
        assert reports[-1]["train_loss"] > 0.0
    return
