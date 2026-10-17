import chess
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch

from bdd import given, then, when
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
        with given("a window dataset over a directory of 20 games"):
            dataset = WindowDataset(config(data_dir))

        with when("the dataset length is queried"):
            length = len(dataset)

        with then("all 20 games are reported"):
            assert length == 20

    def test_window_structure(self, data_dir):
        with given("a window dataset with default settings"):
            dataset = WindowDataset(config(data_dir, seed=7))

        with when("every window is sampled"):
            windows = [dataset[index] for index in range(len(dataset))]

        with then("each window is a bounded CLS/SEP-framed token stream"):
            for sample in windows:
                tokens = sample["tokens"]
                assert tokens[0] == CLS
                assert tokens[-1] == SEP
                assert len(tokens) <= 256
                n_segments = int((tokens == SEP).sum())
                assert 1 <= n_segments <= 32
                assert np.all(tokens >= 0)

    def test_single_mode(self, data_dir):
        with given("a dataset forced to single-segment windows"):
            dataset = WindowDataset(config(data_dir, seed=1, p_single=1.0, p_few=0.0))

        with when("every window is sampled"):
            windows = [dataset[index] for index in range(len(dataset))]

        with then("each window contains exactly one segment"):
            for sample in windows:
                tokens = sample["tokens"]
                assert int((tokens == SEP).sum()) == 1
                assert len(tokens) <= 45

    def test_few_mode(self, data_dir):
        with given("a dataset forced to few-segment windows of 2 to 6 segments"):
            dataset = WindowDataset(
                config(data_dir, seed=2, p_single=0.0, p_few=1.0, few_min=2, few_max=6)
            )

        with when("every window is sampled and its segment count recorded"):
            counts = [
                int((dataset[index]["tokens"] == SEP).sum())
                for index in range(len(dataset))
            ]

        with then("all counts stay within the few-segment bounds"):
            assert min(counts) >= 2
            assert max(counts) <= 6

    def test_full_mode_respects_budget(self, data_dir):
        with given("a dataset in full mode with a 200-token budget"):
            dataset = WindowDataset(
                config(data_dir, seed=3, p_single=0.0, p_few=0.0, max_seq_len=200)
            )

        with when("every window is sampled"):
            windows = [dataset[index] for index in range(len(dataset))]

        with then("no window exceeds the token budget"):
            for sample in windows:
                assert len(sample["tokens"]) <= 200

    def test_long_game_window_starts_randomly(self, data_dir, tmp_path):
        with given("a dataset of long games and a 128-token budget"):
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

        with when("every window is sampled and its end position recorded"):
            starts = set()
            for index in range(len(dataset)):
                sample = dataset[index]
                tokens = sample["tokens"]
                starts.add(int((tokens == SEP).sum()))
                assert len(tokens) <= 128

        with then("windows start at varying positions within the budget"):
            assert len(starts) > 1

    def test_masking_applied(self, data_dir):
        with given("a dataset with masking enabled"):
            dataset = WindowDataset(config(data_dir, seed=9))

        with when("every window is sampled"):
            windows = [dataset[index] for index in range(len(dataset))]

        with then("targets match token shapes and some tokens are masked"):
            any_masked = False
            for sample in windows:
                assert sample["targets"].shape == sample["tokens"].shape
                any_masked |= bool((sample["targets"] != IGNORE_INDEX).any())
            assert any_masked

    def test_masking_disabled(self, data_dir):
        with given("a dataset with masking disabled"):
            dataset = WindowDataset(config(data_dir, seed=9, apply_masking=False))

        with when("every window is sampled"):
            windows = [dataset[index] for index in range(len(dataset))]

        with then("all targets are the ignore index"):
            for sample in windows:
                assert np.all(sample["targets"] == IGNORE_INDEX)

    def test_seeded_reproducibility(self, data_dir):
        with given("two datasets built with the same seed"):
            a = WindowDataset(config(data_dir, seed=11))
            b = WindowDataset(config(data_dir, seed=11))

        with when("the same indices are sampled from both"):
            pairs = [(a[i], b[i]) for i in range(5)]

        with then("both datasets produce identical tokens and targets"):
            for sa, sb in pairs:
                assert np.array_equal(sa["tokens"], sb["tokens"])
                assert np.array_equal(sa["targets"], sb["targets"])


class TestCollate:
    def test_collate_shapes_and_dtypes(self, data_dir):
        with given("samples drawn from a window dataset"):
            dataset = WindowDataset(config(data_dir, seed=13))
            samples = [dataset[i] for i in range(4)]

        with when("the samples are collated"):
            batch = collate_windows(samples)

        with then("the batch has padded shapes and correct dtypes"):
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
        with given("a collated sample built from a packed stream"):
            stream = pack_stream(random_game(21)[:3])
            batch = collate_windows(
                [
                    {
                        "tokens": stream,
                        "targets": np.full(len(stream), IGNORE_INDEX, dtype=np.int16),
                    }
                ]
            )

        with when("the segment ids are read from the batch"):
            segment_ids = batch["segment_ids"][0].numpy()

        with then("they equal the running count of SEP tokens"):
            assert segment_ids[0] == 0
            expected = np.cumsum(stream == SEP)
            assert np.array_equal(segment_ids, expected)
            assert int(segment_ids.max()) == 3

    def test_pad_positions_excluded_from_mask(self):
        with given("a collated batch with one padded sample"):
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

        with when("the padding positions of the short row are inspected"):
            pad_row = ~batch["padding_mask"][1]

        with then("only pad positions are excluded and their targets are ignored"):
            assert pad_row.sum() == len(stream) - 20
            assert torch.all(batch["targets"][1][pad_row] == IGNORE_INDEX)
