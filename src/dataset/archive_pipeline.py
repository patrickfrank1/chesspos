"""Archive-to-HuggingFace pipeline.

For every incomplete archive (local file or remote lichess URL) the
pipeline: downloads/decompresses it as a stream, cuts it into
game-aligned chunks, extracts one row per game per chunk, sorts and
deduplicates the rows, splits them into train/test batches, and uploads
the parquet shards together with their ``.meta.json`` sidecars in bulk
commits (batches are queued and committed every 15 minutes, and any
remainder is flushed together with the completion marker when the
archive finishes). A completed archive is marked with
``archives/<archive_id>/_COMPLETE.json``; marked archives are skipped on
every subsequent run, which makes the whole pipeline idempotent on
ephemeral disks.
"""

from __future__ import annotations

import json
import os
import shutil
import tempfile
import threading
import time
import urllib.request
from collections import deque
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import BinaryIO

import pyarrow as pa
import pyarrow.parquet as pq
import zstandard
from huggingface_hub import CommitOperationAdd, HfApi, hf_hub_download

from src.dataset.batch_meta import BatchMeta, build_batch_meta, config_fingerprint
from src.dataset.chunker import PgnChunker
from src.dataset.config import DatasetConfig, PreprocessingConfig, SourceConfig
from src.dataset.download import download_file
from src.dataset.etl import ChessPositionDataset
from src.dataset.source import ArchivePlanner, WorkItem, completed_archives


def _utcnow() -> str:
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).isoformat()


class _InlineExtractor:
    """Extract chunks on a single background thread (debug mode)."""

    def __init__(self, pool: ThreadPoolExecutor, fn):
        self._pool = pool
        self._fn = fn
        return

    def submit(self, data: bytes):
        return self._pool.submit(self._fn, data)

    def result(self, handle) -> list[dict]:
        return handle.result()


class _RayExtractor:
    """Extract chunks as Ray tasks on the local cluster."""

    def __init__(self, subsampling, time_control_filter):
        import ray

        if not ray.is_initialized():
            ray.init(
                address=os.environ.get("RAY_ADDRESS") or "local",
                dashboard_host=os.environ.get("RAY_DASHBOARD_HOST", "0.0.0.0"),
                ignore_reinit_error=True,
            )
        remote = ray.remote(ChessPositionDataset._extract_positions)
        self._remote = remote
        self._subsampling = subsampling
        self._time_control_filter = time_control_filter
        return

    def submit(self, data: bytes):
        return self._remote.remote(
            {"bytes": data, "path": ""},
            self._subsampling,
            self._time_control_filter,
        )

    def result(self, handle) -> list[dict]:
        import ray

        return ray.get(handle)


@dataclass
class Batch:
    """One parquet file's worth of games (single split)."""

    split: str
    chunk_index: int
    seq: int
    table: pa.Table
    duplicates_dropped: int
    utc_dates: list[str] = field(default_factory=list)

    def __len__(self) -> int:
        return self.table.num_rows

    def __iter__(self):
        return iter(self.table.to_pylist())


@dataclass
class RunResult:
    processed: list[str] = field(default_factory=list)
    skipped: list[str] = field(default_factory=list)


class _Prefetcher:
    """Download the next archive in the background while the current one processes.

    Downloads are link-bound (~10s per monthly archive), so prefetching
    hides them behind the (much slower) extraction work entirely. If the
    background download fails, ``take`` returns ``None`` and the caller
    falls back to streaming the URL directly.
    """

    def __init__(self):
        self._thread: threading.Thread | None = None
        self._dest: Path | None = None
        return

    def start(self, item: WorkItem, dest: Path) -> None:
        if item.provider != "lichess" or not item.url:
            return
        self._dest = Path(dest)
        self._thread = threading.Thread(
            target=download_file,
            args=(item.url, self._dest),
            daemon=True,
        )
        self._thread.start()
        return

    def take(self) -> str | None:
        """Wait for the background download; return its path, or None on failure."""
        if self._thread is None:
            return None
        self._thread.join()
        self._thread = None
        if self._dest is not None and self._dest.exists():
            return str(self._dest)
        return None


class BatchUploader:
    """Queued, idempotent uploader for batch parquet + sidecar pairs.

    Batches are queued with :meth:`queue_batch` and committed in bulk:
    whenever the queue is non-empty and ``flush_interval_seconds`` have
    elapsed since the last commit, the pending operations are committed
    without the completion marker. :meth:`flush` commits any remainder
    together with the ``archives/<archive_id>/_COMPLETE.json`` marker at
    archive end. This keeps the number of Hub commits low (rate limit:
    128 per hour) while bounding re-upload loss after a crash to the
    flush interval.
    """

    MAX_OPS_PER_COMMIT = 32

    def __init__(
        self,
        hf_api: HfApi | None = None,
        repo_id: str = "",
        flush_interval_seconds: float = 900.0,
        clock=time.monotonic,
    ):
        self.hf_api = hf_api or HfApi()
        self.repo_id = repo_id
        self.flush_interval_seconds = flush_interval_seconds
        self._clock = clock
        self._pending: list[CommitOperationAdd] = []
        self._last_commit: float = clock()
        return

    def _existing_meta_json(self, path_in_repo: str) -> str | None:
        try:
            result = hf_hub_download(
                repo_id=self.repo_id,
                filename=path_in_repo,
                repo_type="dataset",
            )
        except Exception:
            return None
        if result is None:
            return None
        try:
            return Path(result).read_text()
        except (OSError, TypeError):
            return result if isinstance(result, str) else None

    def _batch_already_uploaded(self, meta: BatchMeta) -> bool:
        """True when an identical batch (same ``file_sha256``) is on the hub."""
        existing = self._existing_meta_json(meta.repo_meta_path())
        if existing:
            try:
                data = json.loads(existing)
                recorded = data.get("file_sha256")
                if recorded is None:
                    recorded = data.get("stats", {}).get("file_sha256")
                return recorded == meta.file_sha256
            except (ValueError, AttributeError, TypeError):
                pass
        return False

    def _commit_operations(self, operations: list[CommitOperationAdd]) -> None:
        for start in range(0, len(operations), self.MAX_OPS_PER_COMMIT):
            chunk = operations[start : start + self.MAX_OPS_PER_COMMIT]
            self.hf_api.create_commit(
                repo_id=self.repo_id,
                repo_type="dataset",
                operations=chunk,
                commit_message="Add chess position batches",
            )
        self._last_commit = self._clock()
        return

    def _marker_operation(self, archive_id: str, batch_files: list[str]):
        payload = json.dumps(
            {
                "archive_id": archive_id,
                "batch_files": batch_files,
                "created_at": _utcnow(),
            }
        )
        return CommitOperationAdd(
            path_in_repo=f"archives/{archive_id}/_COMPLETE.json",
            path_or_fileobj=payload.encode("utf-8"),
        )

    def queue_batch(self, parquet_path: Path, meta: BatchMeta) -> bool:
        """Queue a batch's parquet + sidecar for bulk commit.

        Commits all pending operations first when the flush interval has
        elapsed. Returns False (and queues nothing) when an identical
        batch (same ``file_sha256``) already exists on the hub.
        """
        if (
            self._pending
            and self._clock() - self._last_commit >= self.flush_interval_seconds
        ):
            pending, self._pending = self._pending, []
            self._commit_operations(pending)
        if self._batch_already_uploaded(meta):
            return False
        self._pending.extend(
            [
                CommitOperationAdd(
                    path_in_repo=meta.repo_parquet_path(),
                    path_or_fileobj=str(parquet_path),
                ),
                CommitOperationAdd(
                    path_in_repo=meta.repo_meta_path(),
                    path_or_fileobj=meta.to_json().encode("utf-8"),
                ),
            ]
        )
        return True

    def flush(self, archive_id: str, batch_files: list[str]) -> None:
        """Commit pending operations together with the completion marker.

        The marker rides in the last chunk when the pending queue is
        chunked; a queue that fits a single commit lands as one commit.
        """
        operations = self._pending
        self._pending = []
        marker = self._marker_operation(archive_id, batch_files)
        marker_path = marker.path_in_repo
        chunks = [
            operations[start : start + self.MAX_OPS_PER_COMMIT]
            for start in range(0, len(operations), self.MAX_OPS_PER_COMMIT)
        ] or [[]]
        chunks[-1] = chunks[-1] + [marker]
        for chunk in chunks:
            message = (
                f"Mark archive {archive_id} complete"
                if any(op.path_in_repo == marker_path for op in chunk)
                else "Add chess position batches"
            )
            self.hf_api.create_commit(
                repo_id=self.repo_id,
                repo_type="dataset",
                operations=chunk,
                commit_message=message,
            )
        self._last_commit = self._clock()
        return


@dataclass
class ArchivePipeline:
    dataset_config: DatasetConfig
    preprocessing_config: PreprocessingConfig
    source_config: SourceConfig

    @classmethod
    def build_batches(
        cls,
        rows: list[dict],
        batch_size: int,
        train_ratio: float,
        chunk_index: int,
        carry: dict | None = None,
        flush: bool = False,
    ) -> tuple[list[Batch], dict]:
        """Sort, deduplicate, split and batch extracted game rows.

        ``carry`` holds the not-yet-emitted remainder from previous chunks
        (per split) plus a pending duplicates count. Only full
        ``batch_size`` batches are emitted; the remainder is returned as
        the new carry. An archive's final partial batch is produced by
        calling this with ``rows=[]`` and ``flush=True`` at the end of the
        stream.

        Rows are sorted by ``game_id`` so that duplicates are adjacent
        and every run produces identical batches. The ``utc_date`` key is
        carried into the batch metadata, not into the parquet schema.
        """
        carry = carry or {"train": [], "test": [], "duplicates": 0}
        groups: dict[str, list[dict]] = {
            "train": list(carry.get("train", [])),
            "test": list(carry.get("test", [])),
        }
        for row in rows:
            split = ChessPositionDataset._split_for_game(row["game_id"], train_ratio)
            groups[split].append(row)

        duplicates_total = int(carry.get("duplicates", 0))
        batches: list[Batch] = []
        new_carry: dict = {"train": [], "test": [], "duplicates": 0}
        for split in ("train", "test"):
            rows_split = sorted(groups[split], key=lambda r: r["game_id"])
            deduped: list[dict] = []
            last_id: str | None = None
            for row in rows_split:
                if row["game_id"] == last_id:
                    duplicates_total += 1
                    continue
                deduped.append(row)
                last_id = row["game_id"]

            emit_up_to = (
                len(deduped) if flush else (len(deduped) // batch_size) * batch_size
            )
            emit_rows, new_carry[split] = deduped[:emit_up_to], deduped[emit_up_to:]
            for seq, start in enumerate(range(0, len(emit_rows), batch_size)):
                part = emit_rows[start : start + batch_size]
                table = pa.table(
                    {
                        "packed": pa.array(
                            [r["packed"] for r in part], type=pa.list_(pa.int16())
                        ),
                        "n_positions": pa.array(
                            [r["n_positions"] for r in part], type=pa.int32()
                        ),
                        "game_id": pa.array(
                            [r["game_id"] for r in part], type=pa.string()
                        ),
                        "white_elo": pa.array(
                            [r["white_elo"] for r in part], type=pa.int32()
                        ),
                        "black_elo": pa.array(
                            [r["black_elo"] for r in part], type=pa.int32()
                        ),
                        "result": pa.array(
                            [r["result"] for r in part], type=pa.string()
                        ),
                    }
                )
                batches.append(
                    Batch(
                        split=split,
                        chunk_index=chunk_index,
                        seq=seq,
                        table=table,
                        duplicates_dropped=0,
                        utc_dates=[r.get("utc_date") or "" for r in part],
                    )
                )
        if batches:
            batches[0].duplicates_dropped = duplicates_total
        else:
            new_carry["duplicates"] = duplicates_total
        return batches, new_carry

    def run(self, dry_run: bool = False, output_dir: str | None = None) -> RunResult:
        completed = self._completed(dry_run, output_dir)
        planner = ArchivePlanner(self.source_config)
        all_items = planner.plan(completed=set())
        uploader = BatchUploader(repo_id=self.dataset_config.repo_name)

        result = RunResult()
        pending = [item for item in all_items if item.archive_id not in completed]
        result.skipped = [
            item.archive_id for item in all_items if item.archive_id in completed
        ]
        with tempfile.TemporaryDirectory(prefix="chesspos-prefetch-") as prefetch_dir:
            prefetcher = _Prefetcher()

            def dest_for(item: WorkItem) -> Path:
                return Path(prefetch_dir) / f"{item.archive_id}.pgn.zst"

            # The first archive streams directly from its URL; from the
            # second onwards each archive is downloaded to disk in the
            # background while its predecessor processes.
            if len(pending) > 1:
                prefetcher.start(pending[1], dest_for(pending[1]))
            for index, item in enumerate(pending):
                local_path: str | None = None
                if index > 0:
                    local_path = prefetcher.take()
                    if index + 1 < len(pending):
                        prefetcher.start(
                            pending[index + 1], dest_for(pending[index + 1])
                        )
                self._process(
                    item,
                    uploader,
                    dry_run=dry_run,
                    output_dir=output_dir,
                    local_path=local_path,
                )
                # Free the disk space of the processed archive immediately:
                # monthly lichess archives are tens of GB, so letting them
                # accumulate in the prefetch dir exhausts the disk.
                if local_path is not None:
                    Path(local_path).unlink(missing_ok=True)
                result.processed.append(item.archive_id)
        return result

    def _completed(self, dry_run: bool, output_dir: str | None) -> set[str]:
        if dry_run:
            if not output_dir:
                return set()
            root = Path(output_dir) / "archives"
            return {p.parent.name for p in root.glob("*/_COMPLETE.json")}
        try:
            files = self.hf_api().list_repo_files(
                repo_id=self.dataset_config.repo_name, repo_type="dataset"
            )
        except Exception:
            return set()
        return completed_archives(files)

    @staticmethod
    def hf_api() -> HfApi:
        return HfApi()

    def _process(
        self,
        item: WorkItem,
        uploader: BatchUploader,
        dry_run: bool,
        output_dir: str | None,
        local_path: str | None = None,
    ) -> None:
        fingerprint = config_fingerprint(
            self.preprocessing_config.subsampling,
            self.preprocessing_config.time_control_filter,
        )
        scratch = Path(tempfile.mkdtemp(prefix="chesspos-"))
        try:
            batch_files: list[str] = []
            chunker = PgnChunker(self.source_config.chunk_target_mb * 1024 * 1024)
            max_inflight = max(1, self.preprocessing_config.worker_count)
            carry: dict = {"train": [], "test": [], "duplicates": 0}
            counters: dict = {"train": 0, "test": 0}
            last_chunk_index = -1

            def emit(batches: list[Batch]) -> None:
                for batch in batches:
                    batch_number = counters[batch.split]
                    counters[batch.split] += 1
                    repo_parquet = self._emit_batch(
                        batch,
                        item,
                        fingerprint,
                        uploader,
                        dry_run=dry_run,
                        output_dir=output_dir,
                        scratch=scratch,
                        batch_number=batch_number,
                    )
                    if repo_parquet:
                        batch_files.append(repo_parquet)
                return

            stream_item = item
            if local_path:
                stream_item = replace(item, path=local_path)
            with self._open_pgn_stream(stream_item) as pgn:
                with self._extractor() as extractor:

                    def on_chunk(chunk_index: int, rows: list[dict]) -> None:
                        nonlocal carry, last_chunk_index
                        last_chunk_index = chunk_index
                        batches, carry = self.build_batches(
                            rows=rows,
                            batch_size=self.dataset_config.batch_size,
                            train_ratio=self.dataset_config.train_ratio,
                            chunk_index=chunk_index,
                            carry=carry,
                        )
                        emit(batches)
                        return

                    self._run_chunk_loop(
                        pgn,
                        chunker,
                        extractor,
                        max_inflight=max_inflight,
                        on_chunk=on_chunk,
                    )
            final_batches, _ = self.build_batches(
                rows=[],
                batch_size=self.dataset_config.batch_size,
                train_ratio=self.dataset_config.train_ratio,
                chunk_index=last_chunk_index + 1,
                carry=carry,
                flush=True,
            )
            emit(final_batches)
            self._mark_complete(item, batch_files, uploader, dry_run, output_dir)
        finally:
            shutil.rmtree(scratch, ignore_errors=True)
        return

    @contextmanager
    def _open_pgn_stream(self, item: WorkItem) -> Iterator[BinaryIO]:
        """Yield a decompressed PGN byte stream for the work item.

        A prefetched local copy (``item.path`` set by ``_process``) is read
        from disk; otherwise remote archives are streamed over HTTP and
        ``.zst`` sources (local or remote) are decompressed on the fly
        while the stream is consumed.
        """
        if item.provider == "lichess" and not item.path:
            raw: BinaryIO = urllib.request.urlopen(item.url)
            name = item.url or ""
        else:
            raw = open(item.path, "rb")  # noqa: SIM115 - closed in finally
            name = item.path or ""
        try:
            if name.endswith(".zst"):
                yield zstandard.ZstdDecompressor().stream_reader(raw)
            else:
                yield raw
        finally:
            raw.close()
        return

    def _run_chunk_loop(
        self,
        pgn: BinaryIO,
        chunker: PgnChunker,
        extractor,
        max_inflight: int,
        on_chunk,
    ) -> None:
        """Cut ``pgn`` into chunks and consume extractions in chunk order.

        Each chunk is submitted for extraction as soon as it is cut, so
        extraction of chunk N+1 overlaps with emission of chunk N. At most
        ``max_inflight`` extractions are outstanding at any time;
        ``on_chunk(chunk_index, rows)`` is called strictly in chunk order.
        """
        max_inflight = max(1, max_inflight)
        pending: deque[tuple[int, object]] = deque()
        with tempfile.TemporaryDirectory(prefix="chesspos-chunks-") as chunk_dir:
            for chunk_index, chunk_path in enumerate(
                chunker.iter_chunks(pgn, out_dir=chunk_dir)
            ):
                handle = extractor.submit(chunk_path.read_bytes())
                # The bytes are now in flight for extraction; the file is
                # dead weight (a large archive would otherwise accumulate
                # its entire decompressed size on disk before finishing).
                chunk_path.unlink()
                pending.append((chunk_index, handle))
                while len(pending) >= max_inflight:
                    index, done = pending.popleft()
                    on_chunk(index, extractor.result(done))
            while pending:
                index, done = pending.popleft()
                on_chunk(index, extractor.result(done))
        return

    @contextmanager
    def _extractor(self):
        """Yield an extractor with ``submit(data)`` / ``result(handle)``.

        Debug mode extracts inline on a single background thread; otherwise
        extraction runs as Ray tasks on the local cluster.
        """
        if self.preprocessing_config.debug:

            def extract(data: bytes) -> list[dict]:
                return ChessPositionDataset._extract_positions(
                    {"bytes": data, "path": ""},
                    self.preprocessing_config.subsampling,
                    self.preprocessing_config.time_control_filter,
                )

            with ThreadPoolExecutor(max_workers=1) as pool:
                yield _InlineExtractor(pool, extract)
        else:
            yield _RayExtractor(
                self.preprocessing_config.subsampling,
                self.preprocessing_config.time_control_filter,
            )

    def _emit_batch(
        self,
        batch: Batch,
        item: WorkItem,
        fingerprint: str,
        uploader: BatchUploader,
        dry_run: bool,
        output_dir: str | None,
        scratch: Path,
        batch_number: int,
    ) -> str | None:
        if dry_run:
            root = Path(output_dir) if output_dir else scratch / "out"
        else:
            root = scratch / "out"
        stem = f"batch_{batch_number:04d}"
        parquet_path = root / batch.split / item.archive_id / f"{stem}.parquet"
        parquet_path.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(batch.table, parquet_path)
        meta = build_batch_meta(
            table=batch.table,
            file_bytes=parquet_path.read_bytes(),
            split=batch.split,
            batch_number=batch_number,
            chunk_index=batch.chunk_index,
            seq=batch.seq,
            provider=item.provider,
            archive_id=item.archive_id,
            url=item.url,
            duplicates_dropped=batch.duplicates_dropped,
            config_fingerprint=fingerprint,
            utc_dates=batch.utc_dates,
            subsampling=self.preprocessing_config.subsampling,
            time_control_filter=self.preprocessing_config.time_control_filter,
            train_ratio=self.dataset_config.train_ratio,
            batch_size=self.dataset_config.batch_size,
        )
        (root / batch.split / item.archive_id / f"{stem}.meta.json").write_text(
            meta.to_json()
        )
        if not dry_run:
            uploader.queue_batch(parquet_path, meta)
            return meta.repo_parquet_path()
        return None

    def _mark_complete(
        self,
        item: WorkItem,
        batch_files: list[str],
        uploader: BatchUploader,
        dry_run: bool,
        output_dir: str | None,
    ) -> None:
        if dry_run:
            root = Path(output_dir) if output_dir else None
            if root is None:
                return
            marker_dir = root / "archives" / item.archive_id
            marker_dir.mkdir(parents=True, exist_ok=True)
            payload = json.dumps(
                {
                    "archive_id": item.archive_id,
                    "batch_files": batch_files,
                    "created_at": _utcnow(),
                }
            )
            (marker_dir / "_COMPLETE.json").write_text(payload)
        else:
            uploader.flush(item.archive_id, batch_files)
        return
