import hashlib
import io
import re
import sys
from unittest.mock import MagicMock, patch

import zstandard

from bdd import given, then, when
from http_support import RangedHttpServer
from src.dataset.archive_pipeline import (
    ArchivePipeline,
    BatchUploader,
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


def make_pipeline(
    debug: bool = False,
    data_path: str | None = "/tmp",
    archive_urls: list[str] | None = None,
    worker_count: int = 1,
) -> ArchivePipeline:
    return ArchivePipeline(
        dataset_config=DatasetConfig(
            repo_name="test/chesspos-test",
            batch_size=10,
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
                "ply": 1,
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
    def _uploader_with_existing(self, existing_meta_json=None):
        api = MagicMock()
        uploader = BatchUploader(hf_api=api, repo_id="test/repo")
        patcher = patch(
            "src.dataset.archive_pipeline.hf_hub_download",
            return_value=None,
        )
        if existing_meta_json is not None:
            patcher = patch(
                "src.dataset.archive_pipeline.hf_hub_download",
                return_value=existing_meta_json,
            )
        return uploader, api, patcher

    def test_upload_creates_single_commit_with_parquet_and_meta(self, tmp_path):
        with given("a written parquet file and its batch meta, nothing on the hub yet"):
            parquet_path = tmp_path / "batch_0000_0000.parquet"
            parquet_path.write_bytes(b"parquet")
            meta = MagicMock()
            meta.repo_parquet_path.return_value = "train/arch/batch_0000_0000.parquet"
            meta.repo_meta_path.return_value = "train/arch/batch_0000_0000.meta.json"
            meta.to_json.return_value = '{"schema_version": 1}'
            uploader, api, patcher = self._uploader_with_existing(None)

        with when("the batch is uploaded"):
            with patcher:
                uploaded = uploader.upload_batch(parquet_path, meta)

        with then("one commit contains both files"):
            assert uploaded is True
            api.create_commit.assert_called_once()
            kwargs = api.create_commit.call_args.kwargs
            ops = kwargs["operations"]
            paths = sorted(op.path_in_repo for op in ops)
            assert paths == [
                "train/arch/batch_0000_0000.meta.json",
                "train/arch/batch_0000_0000.parquet",
            ]
        return

    def test_upload_skipped_when_identical_batch_exists(self, tmp_path):
        with given("a batch already on the hub with the same content hash"):
            parquet_path = tmp_path / "batch_0000_0000.parquet"
            parquet_path.write_bytes(b"parquet")
            sha = hashlib.sha256(b"parquet").hexdigest()
            existing_meta = '{"schema_version": 1, "file_sha256": "%s"}' % sha
            meta = MagicMock()
            meta.repo_parquet_path.return_value = "train/arch/batch_0000_0000.parquet"
            meta.repo_meta_path.return_value = "train/arch/batch_0000_0000.meta.json"
            meta.to_json.return_value = existing_meta
            meta.file_sha256 = sha
            uploader, api, patcher = self._uploader_with_existing(existing_meta)

        with when("the batch is uploaded again"):
            with patcher:
                uploaded = uploader.upload_batch(parquet_path, meta)

        with then("no commit is made"):
            assert uploaded is False
            api.create_commit.assert_not_called()
        return

    def test_upload_overwrites_when_content_differs(self, tmp_path):
        with given("a batch on the hub with a different content hash"):
            parquet_path = tmp_path / "batch_0000_0000.parquet"
            parquet_path.write_bytes(b"new-bytes")
            sha = hashlib.sha256(b"new-bytes").hexdigest()
            existing_meta = (
                '{"schema_version": 1, "file_sha256": "%s"}'
                % hashlib.sha256(b"old-bytes").hexdigest()
            )
            meta = MagicMock()
            meta.repo_parquet_path.return_value = "train/arch/batch_0000_0000.parquet"
            meta.repo_meta_path.return_value = "train/arch/batch_0000_0000.meta.json"
            meta.to_json.return_value = (
                '{"schema_version": 1, "file_sha256": "%s"}' % sha
            )
            meta.file_sha256 = sha
            uploader, api, patcher = self._uploader_with_existing(existing_meta)

        with when("the batch is uploaded"):
            with patcher:
                uploaded = uploader.upload_batch(parquet_path, meta)

        with then("the batch is re-uploaded"):
            assert uploaded is True
            api.create_commit.assert_called_once()
        return

    def test_mark_complete_uploads_completion_marker(self):
        with given("an uploader and the list of uploaded batch files"):
            uploader, api, _ = self._uploader_with_existing(None)

        with when("the archive is marked complete"):
            uploader.mark_complete(
                archive_id="arch",
                batch_files=[
                    "train/arch/batch_0000_0000.parquet",
                    "test/arch/batch_0000_0000.parquet",
                ],
            )

        with then("a _COMPLETE.json marker is committed at the archive root"):
            api.create_commit.assert_called_once()
            kwargs = api.create_commit.call_args.kwargs
            ops = kwargs["operations"]
            assert [op.path_in_repo for op in ops] == ["archives/arch/_COMPLETE.json"]
        return


class TestRayExtractor:
    def test_ray_initialized_as_fresh_local_cluster(self, monkeypatch):
        with given("a stubbed ray module that records init kwargs"):
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
            assert rows == ["row"]


def _zst(payload: bytes) -> bytes:
    return zstandard.ZstdCompressor().compress(payload)


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

    def test_url_archive_downloaded_to_scratch_then_processed_and_removed(
        self, tmp_path
    ):
        with given(
            "an HTTP server serving a zstd PGN archive and a pipeline over its URL"
        ):
            games = b"".join(_three_game_stream())
            server = RangedHttpServer(_zst(games))
            server.start()
            pipeline = make_pipeline(data_path=None, archive_urls=[server.url])
            output_dir = tmp_path / "out"

        with when("the pipeline runs in dry-run mode"):
            try:
                result = pipeline.run(dry_run=True, output_dir=str(output_dir))
            finally:
                server.stop()

        with then(
            "the archive was downloaded via ranged requests, processed, and the scratch file is gone"
        ):
            assert result.processed == ["test-archive"]
            assert server.range_requests
            meta_files = list(output_dir.glob("*/*/*.meta.json"))
            assert meta_files
            assert not list(output_dir.rglob("*.pgn.zst"))
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
