import json

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch
from types import SimpleNamespace

import src.training.window_dataset as window_dataset
from bdd import given, then, when
from src.dataset.token_stream import CLS, EMPTY, SEP, VOCAB_SIZE
from src.training.window_dataset import (
    StreamingWindowDataset,
    WindowDatasetConfig,
    _load_game_segments,
    load_game_pool,
    worker_skip_windows,
    collate_windows,
    streaming_unigram_counts,
    unigram_counts,
)
from torch.utils.data import DataLoader


def write_stream_shard(directory, name, seeds, n_boards=3, board_len=8):
    """Write one parquet shard of packed CLS/SEP-delimited game streams."""
    streams = []
    for seed in seeds:
        rng = np.random.default_rng(seed)
        stream = []
        for _ in range(n_boards):
            stream.append(CLS)
            stream.extend(rng.integers(EMPTY, VOCAB_SIZE, size=board_len).tolist())
            stream.append(SEP)
        streams.append(stream)
    directory.mkdir(parents=True, exist_ok=True)
    pq.write_table(
        pa.table({"packed": pa.array(streams, type=pa.list_(pa.int64()))}),
        directory / name,
    )
    return directory / name


def make_config(directory, **overrides):
    defaults = dict(
        data_dir=str(directory),
        max_seq_len=4096,
        max_segments=128,
        p_single=0.0,
        p_few=0.0,
        seed=17,
        apply_masking=False,
    )
    defaults.update(overrides)
    return WindowDatasetConfig(**defaults)


def expected_packed_games(directory):
    """Packed full-game windows for every game in the directory (shard order)."""
    games = _load_game_segments(directory)
    packed = []
    for segments in games:
        tokens = [CLS]
        for segment in segments:
            tokens.extend(segment.tolist())
            tokens.append(SEP)
        packed.append(np.array(tokens, dtype=np.int16))
    return packed


def sample_signature(sample):
    return sample["tokens"].tobytes()


def packed_from_segments(segments):
    tokens = [CLS]
    for segment in segments:
        tokens.extend(segment.tolist())
        tokens.append(SEP)
    return np.asarray(tokens, dtype=np.int16).tobytes()


class TestStreamingBasics:
    def test_len_comes_from_metadata_without_loading(self, tmp_path):
        with given("a directory with two shards of three games each"):
            directory = tmp_path / "train"
            write_stream_shard(directory, "a.parquet", (1, 2, 3))
            write_stream_shard(directory, "b.parquet", (4, 5, 6))

        with when("the streaming dataset is created"):
            dataset = StreamingWindowDataset(make_config(directory))

        with then("the length equals the total game count from shard metadata"):
            assert len(dataset) == 6
        return

    def test_epoch_yields_every_game_exactly_once(self, tmp_path):
        with given("a directory with two shards and windows that cover whole games"):
            directory = tmp_path / "train"
            write_stream_shard(directory, "a.parquet", (1, 2, 3))
            write_stream_shard(directory, "b.parquet", (4, 5, 6))
            dataset = StreamingWindowDataset(make_config(directory))

        with when("one full epoch is consumed"):
            samples = []
            for sample in dataset:
                samples.append(sample_signature(sample))
                if len(samples) == len(dataset):
                    break

        with then("the samples are exactly the packed games, as a multiset"):
            expected = sorted(g.tobytes() for g in expected_packed_games(directory))
            assert sorted(samples) == expected
        return

    def test_epoch_order_changes_across_epochs(self, tmp_path):
        with given("a seeded streaming dataset over two shards"):
            directory = tmp_path / "train"
            write_stream_shard(directory, "a.parquet", (1, 2, 3))
            write_stream_shard(directory, "b.parquet", (4, 5, 6))
            dataset = StreamingWindowDataset(make_config(directory))

        with when("two epochs are consumed from the same iterator"):
            iterator = iter(dataset)
            first = [
                sample_signature(sample)
                for sample, _ in zip(iterator, range(len(dataset)))
            ]
            second = [
                sample_signature(sample)
                for sample, _ in zip(iterator, range(len(dataset)))
            ]

        with then("the sample order differs between the epochs"):
            assert first != second
        return

    def test_deterministic_given_seed(self, tmp_path):
        with given("a seeded streaming dataset"):
            directory = tmp_path / "train"
            write_stream_shard(directory, "a.parquet", (1, 2, 3))
            write_stream_shard(directory, "b.parquet", (4, 5, 6))

        with when("one epoch is consumed twice with fresh datasets"):

            def one_epoch():
                dataset = StreamingWindowDataset(make_config(directory))
                out = []
                for sample in dataset:
                    out.append(sample_signature(sample))
                    if len(out) == len(dataset):
                        return out
                return out

            first = one_epoch()
            second = one_epoch()

        with then("both epochs are identical"):
            assert first == second
        return

    def test_windows_respect_max_seq_len(self, tmp_path):
        with given("games longer than a small max_seq_len"):
            directory = tmp_path / "train"
            write_stream_shard(directory, "a.parquet", (1, 2), n_boards=50)
            dataset = StreamingWindowDataset(make_config(directory, max_seq_len=128))

        with when("a full epoch is consumed"):
            lengths = []
            for sample in dataset:
                lengths.append(len(sample["tokens"]))
                if len(lengths) == len(dataset):
                    break

        with then("no window exceeds the sequence budget"):
            assert all(length <= 128 for length in lengths)
        return


class TestStreamingWorkers:
    def test_worker_shards_are_disjoint_and_covering(self, monkeypatch, tmp_path):
        with given("four shards and a fake worker context of two workers"):
            directory = tmp_path / "train"
            for i in range(4):
                write_stream_shard(directory, f"s{i}.parquet", (i,))
            dataset = StreamingWindowDataset(make_config(directory))

            infos = [
                SimpleNamespace(id=0, num_workers=2),
                SimpleNamespace(id=1, num_workers=2),
            ]

        with when("each worker resolves its shard assignment"):
            assignments = []
            for info in infos:
                monkeypatch.setattr(torch.utils.data, "get_worker_info", lambda: info)
                assignments.append(dataset._worker_shard_indices())

        with then("the assignments are disjoint and cover every shard"):
            flat = sorted(idx for part in assignments for idx in part)
            assert flat == list(range(4))
            assert not set(assignments[0]) & set(assignments[1])
        return

    def test_dataloader_workers_cover_all_games(self, tmp_path):
        with given("a streaming dataset with enough games per worker for one epoch"):
            directory = tmp_path / "train"
            for i in range(4):
                write_stream_shard(directory, f"s{i}.parquet", range(i * 6, i * 6 + 6))
            dataset = StreamingWindowDataset(make_config(directory))
            loader = DataLoader(
                dataset,
                batch_size=2,
                num_workers=2,
                collate_fn=collate_windows,
            )

        with when("one full epoch (24 games) is consumed across two workers"):
            signatures = []
            for batch in loader:
                for row in batch["tokens"].numpy().astype(np.int16):
                    signatures.append(np.trim_zeros(row, "b").tobytes())
                if len(signatures) >= len(dataset):
                    break

        with then("every game appears exactly once, no epoch wraparound"):
            expected = sorted(g.tobytes() for g in expected_packed_games(directory))
            assert sorted(signatures) == expected
        return


class TestStreamingUnigramCounts:
    def test_matches_preloaded_counts(self, tmp_path):
        with given("a directory with two shards"):
            directory = tmp_path / "train"
            write_stream_shard(directory, "a.parquet", (1, 2))
            write_stream_shard(directory, "b.parquet", (3,))

        with when("unigram counts are computed both ways"):
            streamed = streaming_unigram_counts(str(directory))
            games = _load_game_segments(directory)
            preloaded = unigram_counts(games)

        with then("the counts are identical"):
            assert np.array_equal(streamed, preloaded)
        return

    def test_progress_is_logged(self, tmp_path):
        with given("a directory with two shards and a log collector"):
            directory = tmp_path / "train"
            write_stream_shard(directory, "a.parquet", (1, 2))
            write_stream_shard(directory, "b.parquet", (3,))
            messages: list[str] = []

        with when("unigram counts are computed with a log callback"):
            streaming_unigram_counts(str(directory), log=messages.append)

        with then("progress messages cover start, per-shard progress, and finish"):
            assert any("2 shards" in message for message in messages)
            assert any("1/2" in message for message in messages)
            assert any("2/2" in message for message in messages)
            assert any("done" in message for message in messages)
        return

    def test_sidecar_counts_are_summed_without_reading_shards(
        self, tmp_path, monkeypatch
    ):
        with given("shards whose sidecars each carry a unigram histogram"):
            directory = tmp_path / "train"
            write_stream_shard(directory, "a.parquet", (1, 2))
            write_stream_shard(directory, "b.parquet", (3,))
            histograms = {
                "a.parquet": [10] * VOCAB_SIZE,
                "b.parquet": [20] * VOCAB_SIZE,
            }
            for name, histogram in histograms.items():
                sidecar = directory / name.replace(".parquet", ".meta.json")
                sidecar.write_text(json.dumps({"stats": {"unigram_counts": histogram}}))
            messages: list[str] = []

        with when("shard reading is disabled and counts are requested"):
            monkeypatch.setattr(
                "src.training.window_dataset.pq.read_table",
                lambda *args, **kwargs: pytest.fail("shards must not be reread"),
            )
            counts = streaming_unigram_counts(str(directory), log=messages.append)

        with then("the sidecar histograms are summed directly"):
            expected = np.zeros(VOCAB_SIZE, dtype=np.int64)
            for histogram in histograms.values():
                expected += np.asarray(histogram, dtype=np.int64)
            assert np.array_equal(counts, expected)
            assert any("sidecar" in message for message in messages)
        return

    def test_missing_sidecar_falls_back_to_parquet(self, tmp_path):
        with given("one shard with a true sidecar histogram and one without"):
            directory = tmp_path / "train"
            write_stream_shard(directory, "a.parquet", (1, 2))
            write_stream_shard(directory, "b.parquet", (3,))
            histogram = unigram_counts(_load_game_segments(directory / "a.parquet"))
            sidecar = directory / "a.meta.json"
            sidecar.write_text(
                json.dumps({"stats": {"unigram_counts": histogram.tolist()}})
            )

        with when("unigram counts are computed"):
            counts = streaming_unigram_counts(str(directory))

        with then("the result matches the full recomputation reference"):
            expected = unigram_counts(_load_game_segments(directory))
            assert np.array_equal(counts, expected)
        return


class TestStreamingPrefetch:
    def test_prefetch_keeps_epoch_content_identical(self, tmp_path):
        with given("a streaming dataset with a two-shard lookahead cache"):
            directory = tmp_path / "train"
            write_stream_shard(directory, "a.parquet", (1, 2, 3))
            write_stream_shard(directory, "b.parquet", (4, 5, 6))
            dataset = StreamingWindowDataset(make_config(directory, prefetch_shards=2))

        with when("one full epoch is consumed"):
            samples = []
            for sample in dataset:
                samples.append(sample_signature(sample))
                if len(samples) == len(dataset):
                    break

        with then("the samples are exactly the packed games, as a multiset"):
            expected = sorted(g.tobytes() for g in expected_packed_games(directory))
            assert sorted(samples) == expected
        return

    def test_prefetch_works_across_workers(self, tmp_path):
        with given("a two-worker loader over a prefetched streaming dataset"):
            directory = tmp_path / "train"
            write_stream_shard(directory, "a.parquet", (1, 2, 3))
            write_stream_shard(directory, "b.parquet", (4, 5, 6))
            write_stream_shard(directory, "c.parquet", (7, 8, 9))
            write_stream_shard(directory, "d.parquet", (10, 11, 12))
            dataset = StreamingWindowDataset(make_config(directory, prefetch_shards=2))
            loader = DataLoader(
                dataset,
                batch_size=2,
                num_workers=2,
                collate_fn=collate_windows,
            )

        with when("one full epoch (12 games) is consumed across two workers"):
            signatures = []
            for batch in loader:
                for row in batch["tokens"].numpy().astype(np.int16):
                    signatures.append(np.trim_zeros(row, "b").tobytes())
                if len(signatures) >= len(dataset):
                    break

        with then("every game appears exactly once, no epoch wraparound"):
            expected = sorted(g.tobytes() for g in expected_packed_games(directory))
            assert sorted(signatures) == expected
        return


class TestStreamResume:
    def test_worker_skip_windows_round_robin_math(self):
        with given("five steps at batch size three over two workers"):
            resume_step, batch_size, num_workers = 5, 3, 2

        with when("each worker computes how many windows it already consumed"):
            skip0 = worker_skip_windows(resume_step, batch_size, num_workers, 0)
            skip1 = worker_skip_windows(resume_step, batch_size, num_workers, 1)

        with then("the skips match the round-robin batch counts and sum to the total"):
            assert (skip0, skip1) == (9, 6)
            assert skip0 + skip1 == resume_step * batch_size
        return

    def test_worker_skip_windows_edge_cases(self):
        with given("a resume helper"):
            with when("the resume step is zero"):
                skip = worker_skip_windows(0, 8, 4, 2)
            with then("no windows are skipped on a fresh start"):
                assert skip == 0

            with when("there is a single process (num_workers=0)"):
                skip = worker_skip_windows(7, 8, 0, 0)
            with then("the skip covers every window consumed so far"):
                assert skip == 7 * 8

            with when("a worker whose batches all lie beyond the stop point"):
                skip = worker_skip_windows(3, 8, 4, 3)
            with then("that worker has consumed nothing yet"):
                assert skip == 0
        return

    def test_resume_continues_stream_without_repeats(self, tmp_path):
        with given("a seeded single-worker streaming dataset"):
            directory = tmp_path / "train"
            write_stream_shard(directory, "a.parquet", (1, 2, 3))
            write_stream_shard(directory, "b.parquet", (4, 5, 6))

            def uninterrupted(count):
                dataset = StreamingWindowDataset(make_config(directory))
                return [
                    sample_signature(sample) for sample, _ in zip(dataset, range(count))
                ]

        with when("a run stops after four windows and a fresh dataset resumes"):
            reference = uninterrupted(10)
            resumed_dataset = StreamingWindowDataset(
                make_config(directory, resume_step=4, batch_size=1, num_workers=1)
            )
            continuation = [
                sample_signature(sample) for sample, _ in zip(resumed_dataset, range(6))
            ]

        with then("the continuation is exactly the uninterrupted tail, with no replay"):
            assert continuation == reference[4:]
        return

    def test_resume_with_dataloader_matches_uninterrupted(self, tmp_path):
        with given("a two-worker loader at batch size two"):
            directory = tmp_path / "train"
            for i in range(4):
                write_stream_shard(directory, f"s{i}.parquet", range(i * 6, i * 6 + 6))

            def run_batches(resume_step, count):
                dataset = StreamingWindowDataset(
                    make_config(
                        directory,
                        resume_step=resume_step,
                        batch_size=2,
                        num_workers=2,
                    )
                )
                loader = DataLoader(
                    dataset,
                    batch_size=2,
                    num_workers=2,
                    collate_fn=collate_windows,
                )
                out = []
                for batch in loader:
                    out.append(batch["tokens"].numpy().tobytes())
                    if len(out) == count:
                        break
                return out

        with when("a run stops after five batches and a fresh loader resumes"):
            reference = run_batches(0, 13)
            continuation = run_batches(5, 8)

        with then(
            "the continued batches cover exactly the uninterrupted tail; "
            "order may rotate within a worker cycle"
        ):
            assert sorted(continuation) == sorted(reference[5:13])
        return

    def test_resume_with_shuffle_buffer_matches_uninterrupted(self, tmp_path):
        with given("a seeded streaming dataset with a shuffle buffer"):
            directory = tmp_path / "train"
            for i in range(4):
                write_stream_shard(directory, f"s{i}.parquet", range(i * 6, i * 6 + 6))

            def stream(count, **overrides):
                dataset = StreamingWindowDataset(make_config(directory, **overrides))
                return [
                    sample_signature(sample) for sample, _ in zip(dataset, range(count))
                ]

        with when("a run stops after five windows and a fresh dataset resumes"):
            reference = stream(12, shuffle_buffer=4)
            continuation = stream(
                7, shuffle_buffer=4, resume_step=5, batch_size=1, num_workers=1
            )

        with then("the continuation matches the uninterrupted tail through the buffer"):
            assert continuation == reference[5:]
        return


class TestValidationGamePool:
    def test_pool_is_capped_and_uses_shard_prefetch(self, tmp_path):
        with given("four validation shards with three games each"):
            directory = tmp_path / "val"
            for i in range(4):
                write_stream_shard(directory, f"s{i}.parquet", range(i * 3, i * 3 + 3))

        with when("a five-game pool is loaded with two-shard prefetch"):
            pool = load_game_pool(directory, max_games=5, prefetch_shards=2)

        with then("only the bounded prefix is retained"):
            expected = expected_packed_games(directory)[:5]
            assert [packed_from_segments(game) for game in pool] == [
                game.tobytes() for game in expected
            ]
        return

    def test_pool_never_loads_the_whole_directory(self, tmp_path, monkeypatch):
        with given("a validation directory and a spy on shard loads"):
            directory = tmp_path / "val"
            for i in range(6):
                write_stream_shard(directory, f"s{i}.parquet", (i,))
            loaded = []
            original = window_dataset._load_game_segments

            def spy(path, *args, **kwargs):
                loaded.append(path)
                return original(path, *args, **kwargs)

            monkeypatch.setattr(window_dataset, "_load_game_segments", spy)

        with when("a one-game pool is loaded with one-shard prefetch"):
            pool = load_game_pool(directory, max_games=1, prefetch_shards=1)

        with then("only individual shards are read, never the directory"):
            assert len(pool) == 1
            assert loaded
            assert all(path.is_file() for path in loaded)

        with then("the prefetch window bounds the number of decoded shards"):
            assert len(loaded) <= 2  # one decoded shard plus one lookahead
        return

    def test_pool_is_deterministic(self, tmp_path):
        with given("a validation directory with several shards"):
            directory = tmp_path / "val"
            for i in range(4):
                write_stream_shard(directory, f"s{i}.parquet", range(i * 4, i * 4 + 4))

        with when("the same bounded pool is loaded twice"):
            first = load_game_pool(directory, max_games=7, prefetch_shards=2)
            second = load_game_pool(directory, max_games=7, prefetch_shards=2)

        with then("both pools contain the same games in the same order"):
            assert [packed_from_segments(g) for g in first] == [
                packed_from_segments(g) for g in second
            ]
        return
