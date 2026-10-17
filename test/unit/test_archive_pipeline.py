import hashlib
import io
import re
import sys
from unittest.mock import MagicMock, patch

import zstandard

from bdd import given, then, when
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
            batches = ArchivePipeline.build_batches(
                rows=rows, batch_size=100, train_ratio=0.8, chunk_index=0
            )

        with then("each split produces bounded, game_id-sorted batches"):
            by_split = {}
            for batch in batches:
                by_split.setdefault(batch.split, []).extend(batch)
            assert set(by_split) == {"train", "test"}
            assert sum(len(b) for b in batches) == 250
            for batch in batches:
                assert batch.table.num_rows <= 100
                ids = batch.table.column("game_id").to_pylist()
                assert ids == sorted(ids)

    def test_batch_file_names_are_numbered_per_chunk_and_split(self):
        with given("260 game rows and a batch size of 100"):
            rows = make_rows(260)

        with when("the rows are built into batches for chunk 2"):
            batches = ArchivePipeline.build_batches(
                rows=rows, batch_size=100, train_ratio=0.8, chunk_index=2
            )

        with then("sequences restart per split and encode the chunk index"):
            train_seqs = [b.seq for b in batches if b.split == "train"]
            test_seqs = [b.seq for b in batches if b.split == "test"]
            assert all(b.chunk_index == 2 for b in batches)
            assert train_seqs == sorted(train_seqs)
            assert test_seqs == sorted(test_seqs)
            assert 2 in train_seqs or 2 in test_seqs  # seq 2 exists for a 3-file split

    def test_duplicate_games_are_dropped_within_chunk(self):
        with given("rows containing an exact duplicate game"):
            ids = [f"{i:016x}" for i in range(4)]
            rows = make_rows(game_ids=ids + [ids[0]])

        with when("the rows are built into batches"):
            batches = ArchivePipeline.build_batches(
                rows=rows, batch_size=100, train_ratio=0.8, chunk_index=0
            )

        with then("the duplicate is dropped and counted"):
            total = sum(len(b) for b in batches)
            duplicates = sum(b.duplicates_dropped for b in batches)
            assert total == 4
            assert duplicates == 1

    def test_batches_are_deterministic(self):
        with given("the same rows built twice"):
            rows = make_rows(50)

        with when("build_batches runs twice"):
            first = ArchivePipeline.build_batches(
                rows=rows, batch_size=10, train_ratio=0.8, chunk_index=0
            )
            second = ArchivePipeline.build_batches(
                rows=rows, batch_size=10, train_ratio=0.8, chunk_index=0
            )

        with then("identical splits and game ids are produced"):
            for b1, b2 in zip(first, second):
                assert b1.split == b2.split
                assert b1.seq == b2.seq
                assert (
                    b1.table.column("game_id").to_pylist()
                    == b2.table.column("game_id").to_pylist()
                )


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


class TestRayExtractor:
    def test_ray_initialized_as_fresh_local_cluster(self, monkeypatch):
        # Given a stubbed ray module that records init kwargs
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

            @staticmethod
            def get(obj):
                return ["row"]

        monkeypatch.setitem(sys.modules, "ray", FakeRay)

        # When a ray extractor is created
        extractor = _RayExtractor(None, None)

        # Then ray joins no ambient cluster and starts a fresh local one
        assert init_kwargs.get("address") == "local"
        handle = extractor.submit(b"chunk")
        assert extractor.result(handle) == ["row"]


def _zst(payload: bytes) -> bytes:
    return zstandard.ZstdCompressor().compress(payload)


class TestOpenPgnStream:
    def test_local_pgn_is_passed_through_unmodified(self, tmp_path):
        # Given a local .pgn work item with known bytes
        pgn = tmp_path / "a.pgn"
        pgn.write_bytes(b"[Event x]\n\n1. e4 *")
        pipeline = make_pipeline(data_path=str(tmp_path))
        item = WorkItem(archive_id="a", provider="local", path=str(pgn))

        # When the pgn stream is opened
        with pipeline._open_pgn_stream(item) as stream:
            data = stream.read()

        # Then the raw bytes are returned as-is
        assert data == b"[Event x]\n\n1. e4 *"

    def test_local_zst_is_stream_decompressed_without_intermediate_file(self, tmp_path):
        # Given a local .zst archive containing a PGN
        raw = b"[Event y]\n\n1. d4 *"
        zst = tmp_path / "a.pgn.zst"
        zst.write_bytes(_zst(raw))
        pipeline = make_pipeline(data_path=str(tmp_path))
        item = WorkItem(archive_id="a", provider="local", path=str(zst))

        # When the pgn stream is opened
        with pipeline._open_pgn_stream(item) as stream:
            data = stream.read()

        # Then the decompressed PGN is returned and no decompressed copy exists
        assert data == raw
        assert not list(tmp_path.glob("*.pgn"))

    def test_url_zst_stream_is_decompressed_and_never_written_to_disk(self):
        # Given a URL work item whose response body is zstd-compressed PGN
        raw = b"[Event z]\n\n1. c4 *"
        pipeline = make_pipeline(data_path=None, archive_urls=["https://x/r.pgn.zst"])
        item = WorkItem(archive_id="r", provider="lichess", url="https://x/r.pgn.zst")

        # When the pgn stream is opened over a mocked HTTP response
        response = io.BytesIO(_zst(raw))
        with patch(
            "src.dataset.archive_pipeline.urllib.request.urlopen",
            return_value=response,
        ):
            with pipeline._open_pgn_stream(item) as stream:
                data = stream.read()

        # Then the stream yields the decompressed PGN
        assert data == raw


class _FakeExtractor:
    """Extractor double whose handles resolve out of band, in any order."""

    def __init__(self):
        self.submit_order: list[bytes] = []
        self.resolve_order: list[bytes] = []
        self.max_outstanding = 0
        self._outstanding = 0

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
        # Given a chunker that cuts a 3-game stream into 3 chunks
        # and an extractor whose results are resolved in call order
        pipeline = make_pipeline()
        chunker = PgnChunker(10)
        extractor = _FakeExtractor()
        consumed = []

        # When the chunk loop runs
        pipeline._run_chunk_loop(
            _three_game_stream(),
            chunker,
            extractor,
            max_inflight=3,
            on_chunk=lambda i, rows: consumed.append((i, rows)),
        )

        # Then chunks are submitted and consumed in production order
        # with rows matching their own chunk
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

    def test_in_flight_extractions_are_bounded(self):
        # Given a pipeline loop with a max of 2 in-flight extractions
        pipeline = make_pipeline()
        chunker = PgnChunker(10)
        extractor = _FakeExtractor()

        # When the chunk loop runs over 3 chunks
        pipeline._run_chunk_loop(
            _three_game_stream(),
            chunker,
            extractor,
            max_inflight=2,
            on_chunk=lambda i, rows: None,
        )

        # Then no more than 2 extractions are ever outstanding at once
        assert extractor.max_outstanding <= 2
