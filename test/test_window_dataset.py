import chess
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch

from src.dataset.token_stream import CLS, SEP, TokenStreamEncoder, pack_stream
from src.modeling.masking import IGNORE_INDEX
from src.training.window_dataset import (
    WindowDataset,
    WindowDatasetConfig,
    collate_windows,
)


def random_game(
    seed: int, min_plies: int = 20, max_plies: int = 80
) -> list[np.ndarray]:
    rng = np.random.default_rng(seed)
    board = chess.Board()
    encoder = TokenStreamEncoder()
    segments = []
    n_plies = int(rng.integers(min_plies, max_plies))
    for _ in range(n_plies):
        moves = list(board.legal_moves)
        board.push(moves[int(rng.integers(len(moves)))])
        if board.is_game_over():
            break
        segments.append(encoder.encode(board))
    return segments


@pytest.fixture(scope="module")
def data_dir(tmp_path_factory):
    directory = tmp_path_factory.mktemp("processed")
    table = pa.table(
        {
            "packed": pa.array(
                [pack_stream(random_game(seed)).tolist() for seed in range(20)],
                type=pa.list_(pa.int64()),
            )
        }
    )
    pq.write_table(table, directory / "batch_0000.parquet")
    return directory


def config(data_dir, **kwargs):
    defaults = dict(
        data_dir=str(data_dir),
        max_seq_len=256,
        max_segments=32,
        p_single=0.15,
        p_few=0.25,
        few_min=2,
        few_max=6,
    )
    defaults.update(kwargs)
    return WindowDatasetConfig(**defaults)


class TestWindowDataset:
    def test_loads_all_games(self, data_dir):
        dataset = WindowDataset(config(data_dir))
        assert len(dataset) == 20

    def test_window_structure(self, data_dir):
        dataset = WindowDataset(config(data_dir, seed=7))
        for index in range(len(dataset)):
            sample = dataset[index]
            tokens = sample["tokens"]
            assert tokens[0] == CLS
            assert tokens[-1] == SEP
            assert len(tokens) <= 256
            n_segments = int((tokens == SEP).sum())
            assert 1 <= n_segments <= 32
            assert np.all(tokens >= 0)

    def test_single_mode(self, data_dir):
        dataset = WindowDataset(config(data_dir, seed=1, p_single=1.0, p_few=0.0))
        for index in range(len(dataset)):
            tokens = dataset[index]["tokens"]
            assert int((tokens == SEP).sum()) == 1
            assert len(tokens) <= 45

    def test_few_mode(self, data_dir):
        dataset = WindowDataset(
            config(data_dir, seed=2, p_single=0.0, p_few=1.0, few_min=2, few_max=6)
        )
        counts = []
        for index in range(len(dataset)):
            tokens = dataset[index]["tokens"]
            counts.append(int((tokens == SEP).sum()))
        assert min(counts) >= 2
        assert max(counts) <= 6

    def test_full_mode_respects_budget(self, data_dir):
        dataset = WindowDataset(
            config(data_dir, seed=3, p_single=0.0, p_few=0.0, max_seq_len=200)
        )
        for index in range(len(dataset)):
            tokens = dataset[index]["tokens"]
            assert len(tokens) <= 200

    def test_long_game_window_starts_randomly(self, data_dir, tmp_path):
        directory = tmp_path / "long"
        directory.mkdir()
        table = pa.table(
            {
                "packed": pa.array(
                    [
                        pack_stream(
                            random_game(seed, min_plies=120, max_plies=150)
                        ).tolist()
                        for seed in range(900, 920)
                    ],
                    type=pa.list_(pa.int64()),
                )
            }
        )
        pq.write_table(table, directory / "batch_0000.parquet")
        dataset = WindowDataset(
            config(directory, seed=5, p_single=0.0, p_few=0.0, max_seq_len=128)
        )
        starts = set()
        for index in range(len(dataset)):
            sample = dataset[index]
            tokens = sample["tokens"]
            starts.add(int((tokens == SEP).sum()))
            assert len(tokens) <= 128
        assert len(starts) > 1

    def test_masking_applied(self, data_dir):
        dataset = WindowDataset(config(data_dir, seed=9))
        any_masked = False
        for index in range(len(dataset)):
            sample = dataset[index]
            assert sample["targets"].shape == sample["tokens"].shape
            any_masked |= bool((sample["targets"] != IGNORE_INDEX).any())
        assert any_masked

    def test_masking_disabled(self, data_dir):
        dataset = WindowDataset(config(data_dir, seed=9, apply_masking=False))
        for index in range(len(dataset)):
            assert np.all(dataset[index]["targets"] == IGNORE_INDEX)

    def test_seeded_reproducibility(self, data_dir):
        a = WindowDataset(config(data_dir, seed=11))
        b = WindowDataset(config(data_dir, seed=11))
        for index in range(5):
            sa, sb = a[index], b[index]
            assert np.array_equal(sa["tokens"], sb["tokens"])
            assert np.array_equal(sa["targets"], sb["targets"])


class TestCollate:
    def test_collate_shapes_and_dtypes(self, data_dir):
        dataset = WindowDataset(config(data_dir, seed=13))
        samples = [dataset[i] for i in range(4)]
        batch = collate_windows(samples)

        max_len = max(len(s["tokens"]) for s in samples)
        assert batch["tokens"].shape == (4, max_len)
        assert batch["tokens"].dtype == torch.int64
        assert batch["targets"].dtype == torch.int64
        assert batch["segment_ids"].dtype == torch.int64
        assert batch["padding_mask"].dtype == torch.bool
        assert int(batch["padding_mask"].sum()) == sum(
            len(s["tokens"]) for s in samples
        )

    def test_segment_ids_from_cumsum(self):
        stream = pack_stream(random_game(21)[:3])
        batch = collate_windows(
            [
                {
                    "tokens": stream,
                    "targets": np.full(len(stream), IGNORE_INDEX, dtype=np.int16),
                }
            ]
        )
        segment_ids = batch["segment_ids"][0].numpy()
        assert segment_ids[0] == 0
        expected = np.cumsum(stream == SEP)
        assert np.array_equal(segment_ids, expected)
        assert int(segment_ids.max()) == 3

    def test_pad_positions_excluded_from_mask(self):
        stream = pack_stream(random_game(22)[:2])
        batch = collate_windows(
            [
                {
                    "tokens": stream,
                    "targets": np.full(len(stream), IGNORE_INDEX, dtype=np.int16),
                },
                {
                    "tokens": stream[:20],
                    "targets": np.full(20, IGNORE_INDEX, dtype=np.int16),
                },
            ]
        )
        pad_row = ~batch["padding_mask"][1]
        assert pad_row.sum() == len(stream) - 20
        assert torch.all(batch["targets"][1][pad_row] == IGNORE_INDEX)
