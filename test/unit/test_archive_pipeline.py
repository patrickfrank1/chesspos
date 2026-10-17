import hashlib
from unittest.mock import MagicMock, patch


from bdd import given, then, when
from src.dataset.archive_pipeline import ArchivePipeline, BatchUploader


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
