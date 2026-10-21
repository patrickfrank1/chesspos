import hashlib
import io
import json
import re
import sys
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pyarrow as pa
import pytest
import zstandard

from bdd import given, then, when
from src.dataset.archive_pipeline import (
    ArchivePipeline,
    Batch,
    BatchUploader,
    _Prefetcher,
    _RayExtractor,
)
from src.dataset.chunker import PgnChunker
from src.dataset.config import (
    DatasetConfig,
    GameSubsampling,
    GameSubsampleTier,
    PreprocessingConfig,
    SourceConfig,
)
from src.dataset.source import WorkItem

# These tests boot real local Ray instances; keep them in one xdist worker
# (with --dist loadgroup) so parallel Ray clusters cannot collide on ports.
pytestmark = pytest.mark.xdist_group("ray")


def make_pipeline(
    debug: bool = False,
    data_path: str | None = "/tmp",
    archive_urls: list[str] | None = None,
    worker_count: int = 1,
    batch_size: int = 10,
) -> ArchivePipeline:
    return ArchivePipeline(
        dataset_config=DatasetConfig(
            repo_name="test/chesspos-test",
            batch_size=batch_size,
            train_ratio=0.8,
            data_path=data_path,
        ),
        preprocessing_config=PreprocessingConfig(
            worker_count=worker_count,
            memory_limit_mb=1024,
            debug=debug,
            subsampling=GameSubsampling(tiers=[GameSubsampleTier(min_elo=0, rate=1.0)]),
        ),
        source_config=SourceConfig(data_path=data_path, archive_urls=archive_urls),
    )


def make_rows(num_games=5, game_ids=None):
    ids = game_ids or [f"{i:016x}" for i in range(num_games)]
    rows = []
    for game_id in ids:
        rows.append(
            {
                "packed": [3, 4, 2, 5],
                "n_positions": 2,
                "game_id": game_id,
                "white_elo": 2200,
                "black_elo": 2100,
                "result": "1-0",
                "utc_date": "2023-01-15",
            }
        )
    return rows


class TestBuildBatches:
    def test_batches_are_split_and_bounded(self):
        with given("250 extracted game rows and a batch size of 100"):
            rows = make_rows(250)

        with when("the rows are built into batches for chunk 0"):
            batches, carry = ArchivePipeline.build_batches(
                rows=rows, batch_size=100, train_ratio=0.8, chunk_index=0
            )

        with then(
            "only full 100-game batches are emitted and the remainder is carried"
        ):
            emitted = sum(b.table.num_rows for b in batches)
            assert emitted + len(carry["train"]) + len(carry["test"]) == 250
            assert all(b.table.num_rows == 100 for b in batches)
            assert set(b.split for b in batches) <= {"train", "test"}
            for batch in batches:
                ids = batch.table.column("game_id").to_pylist()
                assert ids == sorted(ids)
        return

    def test_carry_combines_with_next_chunk_rows_to_fill_a_batch(self):
        with given(
            "a chunk 0 with 6 games carried over and a chunk 1 with 8 new games"
        ):
            carry = {"train": make_rows(6), "test": []}
            rows = make_rows(8, game_ids=[f"{i:016x}" for i in range(100, 108)])

        with when("chunk 1 is built with batch size 10"):
            batches, new_carry = ArchivePipeline.build_batches(
                rows=rows, batch_size=10, train_ratio=1.0, chunk_index=1, carry=carry
            )

        with then(
            "one full batch is emitted from both chunks' games and 4 are carried"
        ):
            assert len(batches) == 1
            ids = batches[0].table.column("game_id").to_pylist()
            assert len(ids) == 10
            assert set(ids) == {f"{i:016x}" for i in range(6)} | {
                f"{i:016x}" for i in range(100, 104)
            }
            assert len(new_carry["train"]) == 4
            assert new_carry["duplicates"] == 0
        return

    def test_final_flush_emits_the_remaining_partial_batch(self):
        with given("7 carried games and no further rows"):
            carry = {"train": make_rows(7), "test": []}

        with when("the archive end flushes the accumulator"):
            batches, new_carry = ArchivePipeline.build_batches(
                rows=[],
                batch_size=10,
                train_ratio=1.0,
                chunk_index=5,
                carry=carry,
                flush=True,
            )

        with then("a single partial batch is emitted and the carry is empty"):
            assert len(batches) == 1
            assert batches[0].table.num_rows == 7
            assert batches[0].chunk_index == 5
            assert new_carry == {"train": [], "test": [], "duplicates": 0}
        return

    def test_duplicate_games_are_dropped_across_chunks(self):
        with given("a game that appears in two different chunks"):
            rows_a = make_rows(game_ids=["a" * 16, "b" * 16])
            rows_b = make_rows(game_ids=["b" * 16, "c" * 16])

        with when("both chunks are built with carry-over and flushed"):
            _, carry = ArchivePipeline.build_batches(
                rows=rows_a, batch_size=10, train_ratio=1.0, chunk_index=0
            )
            batches, carry = ArchivePipeline.build_batches(
                rows=rows_b, batch_size=10, train_ratio=1.0, chunk_index=1, carry=carry
            )
            emitted, carry = ArchivePipeline.build_batches(
                rows=[],
                batch_size=10,
                train_ratio=1.0,
                chunk_index=2,
                carry=carry,
                flush=True,
            )

        with then("the duplicate is dropped once and counted on the emitted batch"):
            all_batches = batches + emitted
            assert len(all_batches) == 1
            ids = all_batches[0].table.column("game_id").to_pylist()
            assert ids == sorted(["a" * 16, "b" * 16, "c" * 16])
            assert all_batches[0].duplicates_dropped == 1
        return

    def test_batch_naming_uses_completion_chunk_and_per_call_seq(self):
        with given("260 game rows and a batch size of 100"):
            rows = make_rows(260)

        with when("the rows are built into batches for chunk 2"):
            batches, _ = ArchivePipeline.build_batches(
                rows=rows, batch_size=100, train_ratio=0.8, chunk_index=2
            )

        with then("each batch records the chunk that completed it, seq in call order"):
            assert all(b.chunk_index == 2 for b in batches)
            train_seqs = [b.seq for b in batches if b.split == "train"]
            assert train_seqs == sorted(train_seqs)
        return

    def test_batches_are_deterministic(self):
        with given("the same rows and carry built twice"):
            rows = make_rows(50)
            carry = {"train": make_rows(game_ids=["d" * 16]), "test": []}

        with when("build_batches runs twice"):
            first_batches, first_carry = ArchivePipeline.build_batches(
                rows=rows, batch_size=10, train_ratio=0.8, chunk_index=0, carry=carry
            )
            second_batches, second_carry = ArchivePipeline.build_batches(
                rows=rows, batch_size=10, train_ratio=0.8, chunk_index=0, carry=carry
            )

        with then("identical splits, game ids and carries are produced"):
            for b1, b2 in zip(first_batches, second_batches):
                assert b1.split == b2.split
                assert b1.seq == b2.seq
                assert (
                    b1.table.column("game_id").to_pylist()
                    == b2.table.column("game_id").to_pylist()
                )
            assert [r["game_id"] for r in first_carry["train"]] == [
                r["game_id"] for r in second_carry["train"]
            ]
        return


class TestBatchUploader:
    def _uploader_with_existing(
        self, existing_meta_json=None, flush_interval_seconds=900.0, clock=None
    ):
        api = MagicMock()
        kwargs = {"hf_api": api, "repo_id": "test/repo"}
        if flush_interval_seconds is not None:
            kwargs["flush_interval_seconds"] = flush_interval_seconds
        if clock is not None:
            kwargs["clock"] = clock
        uploader = BatchUploader(**kwargs)
        patcher = patch(
            "src.dataset.archive_pipeline.hf_hub_download",
            return_value=existing_meta_json,
        )
        return uploader, api, patcher

    @staticmethod
    def _meta(parquet_path, sha):
        meta = MagicMock()
        meta.repo_parquet_path.return_value = f"train/arch/{parquet_path.name}"
        meta.repo_meta_path.return_value = f"train/arch/{parquet_path.stem}.meta.json"
        meta.to_json.return_value = '{"schema_version": 1}'
        meta.file_sha256 = sha
        return meta

    def test_queue_batch_defers_commit_until_flush(self, tmp_path):
        with given("a written parquet file and its batch meta, nothing on the hub yet"):
            parquet_path = tmp_path / "batch_0000_0000.parquet"
            parquet_path.write_bytes(b"parquet")
            meta = self._meta(parquet_path, hashlib.sha256(b"parquet").hexdigest())
            uploader, api, patcher = self._uploader_with_existing(None)

        with when("the batch is queued and the archive is flushed"):
            with patcher:
                queued = uploader.queue_batch(parquet_path, meta)
                uploader.flush("arch", [meta.repo_parquet_path.return_value])

        with then("one commit contains the batch files and the completion marker"):
            assert queued is True
            api.create_commit.assert_called_once()
            ops = api.create_commit.call_args.kwargs["operations"]
            paths = sorted(op.path_in_repo for op in ops)
            assert paths == [
                "archives/arch/_COMPLETE.json",
                "train/arch/batch_0000_0000.meta.json",
                "train/arch/batch_0000_0000.parquet",
            ]
        return

    def test_queue_batch_skipped_when_identical_batch_exists(self, tmp_path):
        with given("a batch already on the hub with the same content hash"):
            parquet_path = tmp_path / "batch_0000_0000.parquet"
            parquet_path.write_bytes(b"parquet")
            sha = hashlib.sha256(b"parquet").hexdigest()
            existing_meta = '{"schema_version": 1, "file_sha256": "%s"}' % sha
            meta = self._meta(parquet_path, sha)
            uploader, api, patcher = self._uploader_with_existing(existing_meta)

        with when("the batch is queued and the archive is flushed"):
            with patcher:
                queued = uploader.queue_batch(parquet_path, meta)
                uploader.flush("arch", [meta.repo_parquet_path.return_value])

        with then("the batch is not queued and the flush only commits the marker"):
            assert queued is False
            api.create_commit.assert_called_once()
            ops = api.create_commit.call_args.kwargs["operations"]
            assert [op.path_in_repo for op in ops] == ["archives/arch/_COMPLETE.json"]
        return

    def test_queue_batch_queued_when_content_differs(self, tmp_path):
        with given("a batch on the hub with a different content hash"):
            parquet_path = tmp_path / "batch_0000_0000.parquet"
            parquet_path.write_bytes(b"new-bytes")
            sha = hashlib.sha256(b"new-bytes").hexdigest()
            existing_meta = (
                '{"schema_version": 1, "file_sha256": "%s"}'
                % hashlib.sha256(b"old-bytes").hexdigest()
            )
            meta = self._meta(parquet_path, sha)
            uploader, api, patcher = self._uploader_with_existing(existing_meta)

        with when("the batch is queued and the archive is flushed"):
            with patcher:
                queued = uploader.queue_batch(parquet_path, meta)
                uploader.flush("arch", [meta.repo_parquet_path.return_value])

        with then("the batch is re-uploaded together with the marker"):
            assert queued is True
            api.create_commit.assert_called_once()
            ops = api.create_commit.call_args.kwargs["operations"]
            paths = sorted(op.path_in_repo for op in ops)
            assert paths == [
                "archives/arch/_COMPLETE.json",
                "train/arch/batch_0000_0000.meta.json",
                "train/arch/batch_0000_0000.parquet",
            ]
        return

    def test_queue_batch_within_interval_defers_bulk_commit(self, tmp_path):
        with given("an uploader that committed less than the flush interval ago"):
            now = [100.0]
            parquet_path = tmp_path / "batch_0000_0000.parquet"
            parquet_path.write_bytes(b"parquet")
            meta = self._meta(parquet_path, hashlib.sha256(b"parquet").hexdigest())
            uploader, api, patcher = self._uploader_with_existing(
                None, flush_interval_seconds=900.0, clock=lambda: now[0]
            )

        with when("a batch is queued shortly after the uploader was created"):
            with patcher:
                uploader.queue_batch(parquet_path, meta)

        with then("no commit is made yet"):
            api.create_commit.assert_not_called()
        return

    def test_queue_batch_flushes_pending_in_bulk_when_interval_elapsed(self, tmp_path):
        with given("an uploader with a pending batch from more than 15 minutes ago"):
            now = [0.0]

            def clock():
                return now[0]

            first = tmp_path / "batch_0000_0000.parquet"
            first.write_bytes(b"parquet")
            second = tmp_path / "batch_0001_0000.parquet"
            second.write_bytes(b"parquet-2")
            meta_first = self._meta(first, hashlib.sha256(b"parquet").hexdigest())
            meta_second = self._meta(second, hashlib.sha256(b"parquet-2").hexdigest())
            uploader, api, patcher = self._uploader_with_existing(
                None, flush_interval_seconds=900.0, clock=clock
            )

        with when(
            "the first batch is queued, time advances past the interval, "
            "and the second batch is queued"
        ):
            with patcher:
                uploader.queue_batch(first, meta_first)
                now[0] = 901.0
                uploader.queue_batch(second, meta_second)

        with then("the pending batch is bulk-committed without the marker"):
            api.create_commit.assert_called_once()
            ops = api.create_commit.call_args.kwargs["operations"]
            paths = sorted(op.path_in_repo for op in ops)
            assert paths == [
                "train/arch/batch_0000_0000.meta.json",
                "train/arch/batch_0000_0000.parquet",
            ]
        return

    def test_bulk_committed_batches_are_not_requeued_after_flush_at_archive_end(
        self, tmp_path
    ):
        with given("a batch already bulk-committed by the interval timer"):
            now = [0.0]

            def clock():
                return now[0]

            first = tmp_path / "batch_0000_0000.parquet"
            first.write_bytes(b"parquet")
            second = tmp_path / "batch_0001_0000.parquet"
            second.write_bytes(b"parquet-2")
            meta_first = self._meta(first, hashlib.sha256(b"parquet").hexdigest())
            meta_second = self._meta(second, hashlib.sha256(b"parquet-2").hexdigest())
            uploader, api, patcher = self._uploader_with_existing(
                None, flush_interval_seconds=900.0, clock=clock
            )

        with when(
            "the interval elapses, a second batch is queued, and the archive is flushed"
        ):
            with patcher:
                uploader.queue_batch(first, meta_first)
                now[0] = 901.0
                uploader.queue_batch(second, meta_second)
                uploader.flush("arch", [meta_second.repo_parquet_path.return_value])

        with then(
            "the flush commits only the second batch and the marker, "
            "not the already bulk-committed first batch"
        ):
            assert api.create_commit.call_count == 2
            last_ops = api.create_commit.call_args_list[-1].kwargs["operations"]
            paths = sorted(op.path_in_repo for op in last_ops)
            assert paths == [
                "archives/arch/_COMPLETE.json",
                "train/arch/batch_0001_0000.meta.json",
                "train/arch/batch_0001_0000.parquet",
            ]
        return

    def test_flush_clears_pending_operations(self, tmp_path):
        with given("an uploader with one queued batch"):
            parquet_path = tmp_path / "batch_0000_0000.parquet"
            parquet_path.write_bytes(b"parquet")
            meta = self._meta(parquet_path, hashlib.sha256(b"parquet").hexdigest())
            uploader, api, patcher = self._uploader_with_existing(None)

        with when("the archive is flushed twice"):
            with patcher:
                uploader.queue_batch(parquet_path, meta)
                uploader.flush("arch", [meta.repo_parquet_path.return_value])
                uploader.flush("arch", [meta.repo_parquet_path.return_value])

        with then("the second flush commits only the marker again"):
            assert api.create_commit.call_count == 2
            second_ops = api.create_commit.call_args_list[1].kwargs["operations"]
            assert [op.path_in_repo for op in second_ops] == [
                "archives/arch/_COMPLETE.json"
            ]
        return

    def test_flush_chunks_large_queues_with_marker_in_last_commit(self, tmp_path):
        with given("more queued batch files than fit in one commit"):
            uploader, api, _ = self._uploader_with_existing(None)
            queued = 0
            for i in range(30):
                parquet_path = tmp_path / f"batch_{i:04d}.parquet"
                parquet_path.write_bytes(b"parquet")
                meta = self._meta(parquet_path, hashlib.sha256(b"parquet").hexdigest())
                with patch(
                    "src.dataset.archive_pipeline.hf_hub_download",
                    return_value=None,
                ):
                    uploader.queue_batch(parquet_path, meta)
                queued += 1

        with when("the archive is flushed"):
            uploader.flush("arch", ["train/arch/batch_0000.parquet"])

        with then(
            "the operations are split across commits and the marker rides in the last"
        ):
            commits = api.create_commit.call_args_list
            assert len(commits) >= 2
            for commit in commits[:-1]:
                ops = commit.kwargs["operations"]
                assert all(not op.path_in_repo.endswith("_COMPLETE.json") for op in ops)
            marker_ops = [
                op
                for op in commits[-1].kwargs["operations"]
                if op.path_in_repo.endswith("_COMPLETE.json")
            ]
            assert len(marker_ops) == 1
        return


class TestBatchNumbering:
    def _emit_batch(self, tmp_path, batch, batch_number):
        pipeline = make_pipeline()
        item = WorkItem(
            archive_id="arch", provider="lichess", url="https://x/arch.pgn.zst"
        )
        return pipeline._emit_batch(
            batch,
            item,
            "fp",
            MagicMock(),
            dry_run=True,
            output_dir=str(tmp_path),
            scratch=tmp_path,
            batch_number=batch_number,
        )

    def test_emit_batch_names_files_with_the_consecutive_batch_number(self, tmp_path):
        with given("a batch produced by chunk 1078, numbered 0 at emit time"):
            batch = Batch(
                split="train",
                chunk_index=1078,
                seq=0,
                table=pa.table({"n_positions": pa.array([1], type=pa.int32())}),
                duplicates_dropped=0,
                utc_dates=["2023-01-15"],
            )

        with when("the batch is emitted in dry-run mode"):
            self._emit_batch(tmp_path, batch, batch_number=0)

        with then("files land at consecutive-numbered paths, keeping chunk provenance"):
            assert (tmp_path / "train" / "arch" / "batch_0000.parquet").exists()
            meta = json.loads(
                (tmp_path / "train" / "arch" / "batch_0000.meta.json").read_text()
            )
            assert meta["batch"]["batch_number"] == 0
            assert meta["batch"]["chunk_index"] == 1078
            assert meta["config"]["subsampling"]["tiers"] == [
                {"min_elo": 0, "rate": 1.0}
            ]
            assert meta["config"]["time_control_filter"]["min_seconds"] == 300
            assert meta["config"]["train_ratio"] == 0.8
            assert meta["config"]["batch_size"] == 10
        return

    def test_run_names_batches_consecutively_independent_of_chunking(self, tmp_path):
        with given("a 15-game archive streamed in dry-run mode with batch size 10"):
            games = b"".join(
                f'[Event "G{i}"]\n[Site "s"]\n\n1. e4 *\n\n'.encode() for i in range(15)
            )
            payload = _zst(games)
            pipeline = make_pipeline(
                data_path=None,
                archive_urls=["https://x/test-a.pgn.zst"],
                batch_size=5,
            )

        with when("the pipeline runs to completion"):
            with patch(
                "src.dataset.archive_pipeline.urllib.request.urlopen",
                side_effect=lambda url: io.BytesIO(payload),
            ):
                pipeline.run(dry_run=True, output_dir=str(tmp_path))

        with then("every split's batch files are numbered consecutively from 0000"):
            for split in ("train", "test"):
                names = sorted(
                    p.name for p in (tmp_path / split / "test-a").glob("*.parquet")
                )
                assert len(names) >= 1
                assert names == [f"batch_{i:04d}.parquet" for i in range(len(names))]
            train_names = sorted(
                p.name for p in (tmp_path / "train" / "test-a").glob("*.parquet")
            )
            assert len(train_names) >= 2
        return


class TestRunUploadsPerArchiveCommit:
    def test_non_dry_run_commits_all_batches_and_marker_once(self, tmp_path):
        with given("a 15-game archive with an empty hub and a stubbed HF API"):
            games = b"".join(
                f'[Event "G{i}"]\n[Site "s"]\n\n1. e4 *\n\n'.encode() for i in range(15)
            )
            payload = _zst(games)
            pipeline = make_pipeline(
                debug=True,
                data_path=None,
                archive_urls=["https://x/test-a.pgn.zst"],
            )
            api = MagicMock()

            def fake_init(self, repo_id=""):
                self.hf_api = api
                self.repo_id = repo_id
                self.flush_interval_seconds = 900.0
                self._clock = time.monotonic
                self._pending = []
                self._last_commit = self._clock()
                return

        with when("the pipeline runs with upload enabled"):
            with patch(
                "src.dataset.archive_pipeline.urllib.request.urlopen",
                side_effect=lambda url: io.BytesIO(payload),
            ):
                with patch(
                    "src.dataset.archive_pipeline.hf_hub_download",
                    return_value=None,
                ):
                    with patch.object(BatchUploader, "__init__", fake_init):
                        pipeline.run(dry_run=False, output_dir=str(tmp_path))

        with then("all queued batches and the marker land in a single commit"):
            api.create_commit.assert_called_once()
            ops = api.create_commit.call_args.kwargs["operations"]
            parquet_ops = [op for op in ops if op.path_in_repo.endswith(".parquet")]
            marker_ops = [
                op for op in ops if op.path_in_repo.endswith("_COMPLETE.json")
            ]
            assert len(parquet_ops) >= 2
            assert len(marker_ops) == 1
        return


class TestRayExtractor:
    def test_ray_initialized_as_fresh_local_cluster(self, monkeypatch):
        with given("a stubbed ray module that records init kwargs"):
            monkeypatch.delenv("RAY_ADDRESS", raising=False)
            init_kwargs = {}

            class FakeRemote:
                def remote(self, *args, **kwargs):
                    return ("object-ref", args, kwargs)

            class FakeRay:
                @staticmethod
                def remote(fn):
                    return FakeRemote()

                @staticmethod
                def is_initialized():
                    return False

                @staticmethod
                def init(**kwargs):
                    init_kwargs.update(kwargs)
                    return

                @staticmethod
                def get(obj):
                    return ["row"]

            monkeypatch.setitem(sys.modules, "ray", FakeRay)

        with when("a ray extractor is created and used"):
            extractor = _RayExtractor(None, None)
            handle = extractor.submit(b"chunk")
            rows = extractor.result(handle)

        with then("ray joins no ambient cluster and starts a fresh local one"):
            assert init_kwargs.get("address") == "local"
            assert init_kwargs.get("dashboard_host") == "0.0.0.0"
            assert rows == ["row"]

    def test_ray_dashboard_host_from_env(self, monkeypatch):
        with given("RAY_DASHBOARD_HOST restricts the dashboard bind address"):
            monkeypatch.setenv("RAY_DASHBOARD_HOST", "127.0.0.1")
            init_kwargs = {}

            class FakeRemote:
                def remote(self, *args, **kwargs):
                    return ("object-ref", args, kwargs)

            class FakeRay:
                @staticmethod
                def remote(fn):
                    return FakeRemote()

                @staticmethod
                def is_initialized():
                    return False

                @staticmethod
                def init(**kwargs):
                    init_kwargs.update(kwargs)
                    return

            monkeypatch.setitem(sys.modules, "ray", FakeRay)

        with when("a ray extractor is created"):
            _RayExtractor(None, None)

        with then("ray binds the dashboard to the address from the environment"):
            assert init_kwargs.get("dashboard_host") == "127.0.0.1"
        return

    def test_ray_joins_cluster_from_ray_address_env(self, monkeypatch):
        with given("RAY_ADDRESS points at an existing cluster"):
            monkeypatch.setenv("RAY_ADDRESS", "127.0.0.1:6379")
            init_kwargs = {}

            class FakeRemote:
                def remote(self, *args, **kwargs):
                    return ("object-ref", args, kwargs)

            class FakeRay:
                @staticmethod
                def remote(fn):
                    return FakeRemote()

                @staticmethod
                def is_initialized():
                    return False

                @staticmethod
                def init(**kwargs):
                    init_kwargs.update(kwargs)
                    return

                @staticmethod
                def get(obj):
                    return ["row"]

            monkeypatch.setitem(sys.modules, "ray", FakeRay)

        with when("a ray extractor is created"):
            _RayExtractor(None, None)

        with then("ray connects to the cluster named by RAY_ADDRESS"):
            assert init_kwargs.get("address") == "127.0.0.1:6379"
        return


def _zst(payload: bytes) -> bytes:
    return zstandard.ZstdCompressor().compress(payload)


class TestPrefetch:
    def test_prefetcher_returns_downloaded_file(self, tmp_path):
        with given("a prefetcher started on a URL whose download writes the archive"):
            dest = tmp_path / "test-a.pgn.zst"
            prefetcher = _Prefetcher()
            with patch(
                "src.dataset.archive_pipeline.download_file",
                side_effect=lambda url, d: Path(d).write_bytes(b"archive-bytes"),
            ):
                prefetcher.start(
                    WorkItem(
                        archive_id="test-a",
                        provider="lichess",
                        url="https://x/test-a.pgn.zst",
                    ),
                    dest,
                )

        with when("the prefetched file is taken"):
            path = prefetcher.take()

        with then("the completed download is returned"):
            assert path == str(dest)
            assert Path(path).read_bytes() == b"archive-bytes"
        return

    def test_prefetcher_failure_yields_none_for_streaming_fallback(self, tmp_path):
        with given("a prefetcher whose background download fails"):
            prefetcher = _Prefetcher()
            with patch(
                "src.dataset.archive_pipeline.download_file",
                side_effect=OSError("connection reset"),
            ):
                prefetcher.start(
                    WorkItem(
                        archive_id="test-a",
                        provider="lichess",
                        url="https://x/test-a.pgn.zst",
                    ),
                    tmp_path / "test-a.pgn.zst",
                )

        with when("the prefetched file is taken"):
            path = prefetcher.take()

        with then("no file is returned and the caller can fall back to streaming"):
            assert path is None
        return

    def test_run_streams_first_archive_without_downloading_it(self, tmp_path):
        with given(
            "a pipeline over one archive URL and a fake downloader that must never run"
        ):
            payload = _zst(b'[Event "A"]\n[Site "s"]\n[Result "1-0"]\n\n1. e4 *\n\n')
            events: list[str] = []

            def fake_download(url, dest):
                events.append("download:test-a")
                Path(dest).write_bytes(payload)
                return

            pipeline = make_pipeline(
                data_path=None, archive_urls=["https://x/test-a.pgn.zst"]
            )
            original_process = pipeline._process

            def spy_process(item, *args, **kwargs):
                assert kwargs.get("local_path") is None
                return original_process(item, *args, **kwargs)

            pipeline._process = spy_process

        with when("the pipeline runs in dry-run mode"):
            with (
                patch(
                    "src.dataset.archive_pipeline.download_file",
                    side_effect=fake_download,
                ),
                patch(
                    "src.dataset.archive_pipeline.urllib.request.urlopen",
                    side_effect=lambda url: io.BytesIO(payload),
                ),
            ):
                result = pipeline.run(dry_run=True, output_dir=str(tmp_path))

        with then("the archive is processed by streaming, with no download attempt"):
            assert result.processed == ["test-a"]
            assert events == []
        return

    def test_run_downloads_next_archive_while_processing_current(self, tmp_path):
        with given(
            "a pipeline over two archive URLs, a fake downloader recording events, "
            "and a fake HTTP stream for the first archive"
        ):
            events: list[str] = []
            payloads = {
                "test-a": _zst(
                    b'[Event "A"]\n[Site "s"]\n[Result "1-0"]\n\n1. e4 *\n\n'
                ),
                "test-b": _zst(
                    b'[Event "B"]\n[Site "s"]\n[Result "1-0"]\n\n1. d4 *\n\n'
                ),
            }

            def fake_download(url, dest):
                events.append("download:test-b")
                Path(dest).write_bytes(payloads["test-b"])
                return

            pipeline = make_pipeline(
                data_path=None,
                archive_urls=[
                    "https://x/test-a.pgn.zst",
                    "https://x/test-b.pgn.zst",
                ],
            )
            original_process = pipeline._process

            def spy_process(item, *args, **kwargs):
                events.append(f"process:{item.archive_id}")
                return original_process(item, *args, **kwargs)

            pipeline._process = spy_process

        with when("the pipeline runs in dry-run mode"):
            with (
                patch(
                    "src.dataset.archive_pipeline.download_file",
                    side_effect=fake_download,
                ),
                patch(
                    "src.dataset.archive_pipeline.urllib.request.urlopen",
                    side_effect=lambda url: io.BytesIO(payloads["test-a"]),
                ),
            ):
                result = pipeline.run(dry_run=True, output_dir=str(tmp_path))

        with then(
            "the first archive streams directly while the second downloads in "
            "the background"
        ):
            assert result.processed == ["test-a", "test-b"]
            assert "download:test-a" not in events
            assert events.index("download:test-b") < events.index("process:test-a")
        return


class TestOpenPgnStream:
    def test_local_pgn_is_passed_through_unmodified(self, tmp_path):
        with given("a local .pgn work item with known bytes"):
            pgn = tmp_path / "a.pgn"
            pgn.write_bytes(b"[Event x]\n\n1. e4 *")
            pipeline = make_pipeline(data_path=str(tmp_path))
            item = WorkItem(archive_id="a", provider="local", path=str(pgn))

        with when("the pgn stream is opened"):
            with pipeline._open_pgn_stream(item) as stream:
                data = stream.read()

        with then("the raw bytes are returned as-is"):
            assert data == b"[Event x]\n\n1. e4 *"
        return

    def test_local_zst_is_stream_decompressed_without_intermediate_file(self, tmp_path):
        with given("a local .zst archive containing a PGN"):
            raw = b"[Event y]\n\n1. d4 *"
            zst = tmp_path / "a.pgn.zst"
            zst.write_bytes(_zst(raw))
            pipeline = make_pipeline(data_path=str(tmp_path))
            item = WorkItem(archive_id="a", provider="local", path=str(zst))

        with when("the pgn stream is opened"):
            with pipeline._open_pgn_stream(item) as stream:
                data = stream.read()

        with then("the decompressed PGN is returned and no decompressed copy exists"):
            assert data == raw
            assert not list(tmp_path.glob("*.pgn"))
        return

    def test_url_zst_stream_is_decompressed_and_never_written_to_disk(self):
        with given("a URL work item whose response body is zstd-compressed PGN"):
            raw = b"[Event z]\n\n1. c4 *"
            pipeline = make_pipeline(
                data_path=None, archive_urls=["https://x/r.pgn.zst"]
            )
            item = WorkItem(
                archive_id="r", provider="lichess", url="https://x/r.pgn.zst"
            )

        with when("the pgn stream is opened over a mocked HTTP response"):
            response = io.BytesIO(_zst(raw))
            with patch(
                "src.dataset.archive_pipeline.urllib.request.urlopen",
                return_value=response,
            ):
                with pipeline._open_pgn_stream(item) as stream:
                    data = stream.read()

        with then("the stream yields the decompressed PGN"):
            assert data == raw
        return


class _FakeExtractor:
    """Extractor double whose handles resolve out of band, in any order."""

    def __init__(self):
        self.submit_order: list[bytes] = []
        self.resolve_order: list[bytes] = []
        self.max_outstanding = 0
        self._outstanding = 0
        return

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def submit(self, data: bytes):
        self.submit_order.append(data)
        self._outstanding += 1
        self.max_outstanding = max(self.max_outstanding, self._outstanding)
        return data

    def result(self, handle: bytes) -> list[dict]:
        self.resolve_order.append(handle)
        self._outstanding -= 1
        event = re.search(rb'\[Event "(.*)"\]', handle).group(1).decode()
        return [{"game_id": f"gid-{event}"}]


def _three_game_stream() -> io.BytesIO:
    games = b"".join(
        f'[Event "G{i}"]\n[Site "s"]\n\n1. e4 *\n\n'.encode() for i in range(3)
    )
    return io.BytesIO(games)


class _DirRecordingChunker(PgnChunker):
    """Chunker that remembers the directory its chunks are written to."""

    def __init__(self, chunk_target_bytes: int):
        super().__init__(chunk_target_bytes)
        self.seen_dirs: list[Path] = []
        return

    def iter_chunks(self, stream, out_dir):
        self.seen_dirs.append(Path(out_dir))
        return super().iter_chunks(stream, out_dir)


class TestChunkPipeline:
    def test_chunks_are_consumed_in_production_order(self):
        with given(
            "a chunker that cuts a 3-game stream into 3 chunks "
            "and an extractor whose results are resolved in call order"
        ):
            pipeline = make_pipeline()
            chunker = PgnChunker(10)
            extractor = _FakeExtractor()
            consumed = []

        with when("the chunk loop runs"):
            pipeline._run_chunk_loop(
                _three_game_stream(),
                chunker,
                extractor,
                max_inflight=3,
                on_chunk=lambda i, rows: consumed.append((i, rows)),
            )

        with then("chunks are submitted and consumed in production order"):
            events = [
                re.search(rb'\[Event "(G\d)"\]', data).group(1)
                for data in extractor.submit_order
            ]
            assert events == [b"G0", b"G1", b"G2"]
            assert [i for i, _ in consumed] == [0, 1, 2]
            assert [rows[0]["game_id"] for _, rows in consumed] == [
                "gid-G0",
                "gid-G1",
                "gid-G2",
            ]
        return

    def test_in_flight_extractions_are_bounded(self):
        with given("a pipeline loop with a max of 2 in-flight extractions"):
            pipeline = make_pipeline()
            chunker = PgnChunker(10)
            extractor = _FakeExtractor()

        with when("the chunk loop runs over 3 chunks"):
            pipeline._run_chunk_loop(
                _three_game_stream(),
                chunker,
                extractor,
                max_inflight=2,
                on_chunk=lambda i, rows: None,
            )

        with then("no more than 2 extractions are ever outstanding at once"):
            assert extractor.max_outstanding <= 2
        return

    def test_chunk_files_are_deleted_after_submission(self):
        with given("a chunk loop over a 3-game stream whose chunk dir is observable"):
            pipeline = make_pipeline()
            chunker = _DirRecordingChunker(10)
            extractor = _FakeExtractor()
            files_per_chunk: list[int] = []

            def on_chunk(index: int, rows: list[dict]) -> None:
                chunk_dir = chunker.seen_dirs[0]
                files_per_chunk.append(len(list(chunk_dir.glob("chunk_*.pgn"))))
                return

        with when("the chunk loop runs to completion"):
            pipeline._run_chunk_loop(
                _three_game_stream(),
                chunker,
                extractor,
                max_inflight=3,
                on_chunk=on_chunk,
            )

        with then("every chunk file is already deleted when its result is consumed"):
            # A 17GB zst archive produces ~90GB of chunks; keeping them on
            # disk until the end of the archive filled the disk overnight.
            assert files_per_chunk == [0, 0, 0]
        return
