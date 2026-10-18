import chess
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch

from bdd import given, then, when
from src.dataset.token_stream import CLS, SEP, TokenStreamEncoder, pack_stream
from src.modeling.masking import IGNORE_INDEX, MaskingConfig
from src.training.window_dataset import (
    WindowDataset,
    WindowDatasetConfig,
    _load_game_segments,
    build_mode_pattern,
    collate_windows,
    inverse_frequency_weights,
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
        return

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
        return

    def test_single_mode(self, data_dir):
        with given(
            "a dataset forced to single-segment windows with legacy mode drawing"
        ):
            dataset = WindowDataset(
                config(data_dir, seed=1, p_single=1.0, p_few=0.0, balanced_modes=False)
            )

        with when("every window is sampled"):
            windows = [dataset[index] for index in range(len(dataset))]

        with then("each window contains exactly one segment"):
            for sample in windows:
                tokens = sample["tokens"]
                assert int((tokens == SEP).sum()) == 1
                assert len(tokens) <= 45
        return

    def test_few_mode(self, data_dir):
        with given("a dataset forced to few-segment windows with legacy mode drawing"):
            dataset = WindowDataset(
                config(
                    data_dir,
                    seed=2,
                    p_single=0.0,
                    p_few=1.0,
                    few_min=2,
                    few_max=6,
                    balanced_modes=False,
                )
            )

        with when("every window is sampled and its segment count recorded"):
            counts = [
                int((dataset[index]["tokens"] == SEP).sum())
                for index in range(len(dataset))
            ]

        with then("all counts stay within the few-segment bounds"):
            assert min(counts) >= 2
            assert max(counts) <= 6
        return

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
        return

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
        return

    def test_masking_applied(self, data_dir):
        with given("a dataset with masking enabled"):
            dataset = WindowDataset(config(data_dir, seed=9))

        with when("every window is sampled"):
            windows = [dataset[index] for index in range(len(dataset))]

        with then("targets and weights match token shapes and some tokens are masked"):
            any_masked = False
            for sample in windows:
                assert sample["targets"].shape == sample["tokens"].shape
                assert sample["weights"].shape == sample["tokens"].shape
                assert sample["weights"].dtype == np.float32
                masked = sample["targets"] != IGNORE_INDEX
                any_masked |= bool(masked.any())
                assert np.all(sample["weights"][~masked] == 0.0)
                assert np.all(sample["weights"][masked] > 0.0)
            assert any_masked
        return

    def test_masking_disabled(self, data_dir):
        with given("a dataset with masking disabled"):
            dataset = WindowDataset(config(data_dir, seed=9, apply_masking=False))

        with when("every window is sampled"):
            windows = [dataset[index] for index in range(len(dataset))]

        with then("all targets are the ignore index"):
            for sample in windows:
                assert np.all(sample["targets"] == IGNORE_INDEX)
        return

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
                assert np.array_equal(sa["weights"], sb["weights"])
        return


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
            assert batch["weights"].dtype == torch.float32
            assert batch["weights"].shape == batch["tokens"].shape
            assert batch["segment_ids"].dtype == torch.int64
            assert batch["padding_mask"].dtype == torch.bool
            assert int(batch["padding_mask"].sum()) == sum(
                len(s["tokens"]) for s in samples
            )
        return

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
        return

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
        return


class TestNestedArchiveLayout:
    def write_shard(self, directory, archive: str, seeds: tuple[int, ...]):
        shard_dir = directory / archive
        shard_dir.mkdir(parents=True)
        table = pa.table(
            {
                "packed": pa.array(
                    [pack_stream(random_game(seed)).tolist() for seed in seeds],
                    type=pa.list_(pa.int64()),
                )
            }
        )
        pq.write_table(table, shard_dir / "batch_0000.parquet")
        (shard_dir / "batch_0000.meta.json").touch()
        return

    def test_window_dataset_loads_nested_shards(self, tmp_path):
        with given("a data directory with shards nested per archive and meta sidecars"):
            directory = tmp_path / "processed"
            self.write_shard(directory, "lichess_2014-01", (1, 2))
            self.write_shard(directory, "lichess_2021-01", (3, 4))

        with when("a window dataset is created over the directory"):
            dataset = WindowDataset(
                WindowDatasetConfig(
                    data_dir=str(directory), max_seq_len=256, max_segments=32
                )
            )

        with then("all four games across both archives are loaded"):
            assert len(dataset) == 4
        return

    def test_load_game_segments_from_nested_directory(self, tmp_path):
        with given("a data directory with shards nested per archive and meta sidecars"):
            directory = tmp_path / "processed"
            self.write_shard(directory, "lichess_2014-01", (1, 2))
            self.write_shard(directory, "lichess_2021-01", (3, 4))

        with when("the game segments are loaded from the directory"):
            games = _load_game_segments(directory)

        with then("all four games are returned and sidecars are ignored"):
            assert len(games) == 4
            assert all(len(game) > 0 for game in games)
        return


class TestModeAssignment:
    def test_mode_pattern_reflects_probabilities(self):
        with given("the default masking probabilities (0.45/0.30/0.25)"):
            masking = MaskingConfig()

        with when("the mode pattern is built"):
            pattern = build_mode_pattern(masking)

        with then("the period-20 pattern contains 9 random, 6 board, and 5 span"):
            assert len(pattern) == 20
            assert pattern.count("random") == 9
            assert pattern.count("board") == 6
            assert pattern.count("span") == 5
        return

    def test_mode_pattern_single_mode(self):
        with given("a one-hot board masking config as used by the eval suite"):
            masking = MaskingConfig(p_random=0.0, p_board=1.0, p_span=0.0)

        with when("the mode pattern is built"):
            pattern = build_mode_pattern(masking)

        with then("every slot is board mode"):
            assert set(pattern) == {"board"}
        return

    def test_force_mode_samples_long_windows(self, data_dir):
        with given("a dataset with force_mode='board' and a full-window budget"):
            dataset = WindowDataset(
                config(data_dir, seed=4, force_mode="board", max_seq_len=2048)
            )

        with when("every window is sampled and its masked segments are recorded"):
            results = []
            for sample in dataset_iter(dataset):
                targets = sample["targets"]
                sep_positions = np.flatnonzero(sample["tokens"] == SEP)
                flags = fully_masked_segments(targets, sep_positions)
                masked = targets != IGNORE_INDEX
                weight_ok = bool(np.all(sample["weights"][masked] == 2.0))
                contiguous = len(flags) == 4 and len(flags) == (
                    flags[-1] - flags[0] + 1
                )
                results.append(contiguous and weight_ok)

        with then("each window masks four contiguous segments with board weights"):
            assert all(results)
        return

    def test_force_mode_falls_back_to_random_on_short_games(self):
        with given("a dataset of 3-segment games with force_mode='board'"):
            games = [random_game(seed, min_plies=3, max_plies=4) for seed in range(4)]
            dataset = WindowDataset(
                WindowDatasetConfig(
                    games=games,
                    max_seq_len=2048,
                    max_segments=32,
                    force_mode="board",
                    seed=6,
                )
            )

        with when("every window is sampled"):
            windows = [dataset[index] for index in range(len(dataset))]

        with then("masking falls back to random mode with random-mode weights"):
            for sample in windows:
                masked = sample["targets"] != IGNORE_INDEX
                assert masked.any()
                assert np.all(sample["weights"][masked] == 1.0)
        return

    def test_balanced_modes_follow_pattern(self, data_dir):
        with given("a balanced-mode dataset over 20 games with a full-window budget"):
            dataset = WindowDataset(config(data_dir, seed=8, max_seq_len=2048))

        with when("the first 20 windows are sampled and their mode weights summed"):
            upweighted = 0
            for index in range(20):
                sample = dataset[index]
                masked = sample["targets"] != IGNORE_INDEX
                assert masked.any()
                upweighted += int(np.all(sample["weights"][masked] == 2.0))

        with then("11 of 20 windows carry board/span weights (6 board + 5 span)"):
            assert upweighted == 11
        return


class TestFrequencyWeights:
    def test_uniform_counts_give_uniform_weights(self):
        with given("a unigram count vector with equal counts everywhere"):
            counts = np.full(806, 10)

        with when("inverse-frequency weights are computed"):
            weights = inverse_frequency_weights(counts)

        with then("every token gets weight 1.0"):
            assert np.all(weights == 1.0)
        return

    def test_rare_tokens_are_upweighted_and_capped(self):
        with given("a count vector with one very frequent and one rare token"):
            counts = np.zeros(806)
            counts[38] = 1_000_000
            counts[100] = 10

        with when("inverse-frequency weights are computed"):
            weights = inverse_frequency_weights(counts, cap=50.0)

        with then("the frequent token gets 1.0 and the rare token hits the cap"):
            assert weights[38] == 1.0
            assert weights[100] == 50.0
        return

    def test_smoothing_avoids_infinite_weights(self):
        with given("a count vector with tokens never observed"):
            counts = np.zeros(806)
            counts[38] = 500

        with when("inverse-frequency weights are computed"):
            weights = inverse_frequency_weights(counts)

        with then("all weights are finite and positive"):
            assert np.all(np.isfinite(weights))
            assert np.all(weights > 0.0)
        return

    def test_dataset_applies_frequency_weights(self, data_dir):
        with given("a dataset with a unigram weight table favouring one token"):
            table = np.ones(806, dtype=np.float32)
            table[60] = 5.0
            dataset = WindowDataset(
                config(data_dir, seed=12, max_seq_len=2048, unigram_weights=table)
            )

        with when("windows are sampled and weights recomputed manually"):
            checked = 0
            for index in range(len(dataset)):
                sample = dataset[index]
                masked = sample["targets"] != IGNORE_INDEX
                mode_weight = (
                    sample["weights"][masked] / table[sample["targets"][masked]]
                )
                assert np.all(sample["weights"][masked] > 0.0)
                checked += int(np.all(np.isin(mode_weight, (1.0, 2.0))))
                if checked > 3:
                    break

        with then("weights equal mode weight times frequency weight"):
            assert checked > 3
        return


def dataset_iter(dataset):
    for index in range(len(dataset)):
        yield dataset[index]


def fully_masked_segments(targets, sep_positions):
    starts = [1] + [int(s) + 1 for s in sep_positions[:-1]]
    flags = []
    for i, start in enumerate(starts):
        end = int(sep_positions[i])
        flags.append(all(targets[p] != IGNORE_INDEX for p in range(start, end)))
    return np.asarray([idx for idx, flag in enumerate(flags) if flag])


class TestParallelShardLoading(TestNestedArchiveLayout):
    def test_parallel_load_matches_sequential_order(self, tmp_path):
        with given("a data directory with several shards across nested archives"):
            directory = tmp_path / "processed"
            self.write_shard(directory, "lichess_2014-01", (1, 2, 3))
            self.write_shard(directory, "lichess_2014-02", (4, 5))
            self.write_shard(directory, "lichess_2021-01", (6, 7, 8, 9))

        with when("the segments are loaded with a thread pool"):
            parallel = _load_game_segments(directory, max_workers=3)

        with then("the games match the sequential load, in shard order"):
            sequential = _load_game_segments(directory, max_workers=1)
            assert len(parallel) == 9
            assert all(
                np.array_equal(left, right)
                for game_p, game_s in zip(parallel, sequential)
                for left, right in zip(game_p, game_s)
            )
        return

    def test_parallel_load_single_file(self, tmp_path):
        with given("a single parquet shard"):
            directory = tmp_path / "processed"
            self.write_shard(directory, "lichess_2014-01", (1, 2))
            shard = sorted(directory.rglob("*.parquet"))[0]

        with when("the shard is loaded directly with a thread pool argument"):
            games = _load_game_segments(shard, max_workers=4)

        with then("both games are returned"):
            assert len(games) == 2
        return

    def test_parallel_load_empty_directory(self, tmp_path):
        with given("an empty data directory"):
            directory = tmp_path / "processed"
            directory.mkdir()

        with when("the segments are loaded with a thread pool"):
            games = _load_game_segments(directory, max_workers=4)

        with then("no games are returned"):
            assert games == []
        return
