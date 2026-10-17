"""Archive-to-HuggingFace pipeline.

For every incomplete archive (local file or remote lichess URL) the
pipeline: downloads/decompresses it as a stream, cuts it into
game-aligned chunks, extracts one row per game per chunk, sorts and
deduplicates the rows, splits them into train/test batches, and uploads
each parquet shard together with a ``.meta.json`` sidecar in a single
atomic commit. A completed archive is marked with
``archives/<archive_id>/_COMPLETE.json``; marked archives are skipped on
every subsequent run, which makes the whole pipeline idempotent on
ephemeral disks.
"""

from __future__ import annotations

import json
import shutil
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from huggingface_hub import CommitOperationAdd, HfApi, hf_hub_download

from src.dataset.batch_meta import BatchMeta, build_batch_meta, config_fingerprint
from src.dataset.chunker import PgnChunker, decompress_zst
from src.dataset.config import DatasetConfig, PreprocessingConfig, SourceConfig
from src.dataset.download import download_archive
from src.dataset.etl import ChessPositionDataset
from src.dataset.source import ArchivePlanner, WorkItem, completed_archives


def _utcnow() -> str:
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).isoformat()


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


class BatchUploader:
    """Idempotent uploader for batch parquet + sidecar pairs."""

    def __init__(self, hf_api: HfApi | None = None, repo_id: str = ""):
        self.hf_api = hf_api or HfApi()
        self.repo_id = repo_id

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

    def upload_batch(self, parquet_path: Path, meta: BatchMeta) -> bool:
        """Upload parquet + sidecar in one commit.

        Returns False (and uploads nothing) when an identical batch
        (same ``file_sha256``) already exists on the hub; overwrites when
        the content differs.
        """
        existing = self._existing_meta_json(meta.repo_meta_path())
        if existing:
            try:
                data = json.loads(existing)
                recorded = data.get("file_sha256")
                if recorded is None:
                    recorded = data.get("stats", {}).get("file_sha256")
                if recorded == meta.file_sha256:
                    return False
            except (ValueError, AttributeError, TypeError):
                pass
        operations = [
            CommitOperationAdd(
                path_in_repo=meta.repo_parquet_path(),
                path_or_fileobj=str(parquet_path),
            ),
            CommitOperationAdd(
                path_in_repo=meta.repo_meta_path(),
                path_or_fileobj=meta.to_json().encode("utf-8"),
            ),
        ]
        self.hf_api.create_commit(
            repo_id=self.repo_id,
            repo_type="dataset",
            operations=operations,
            commit_message=f"Add {meta.split} batch {meta.file_name}",
        )
        return True

    def mark_complete(self, archive_id: str, batch_files: list[str]) -> None:
        payload = json.dumps(
            {
                "archive_id": archive_id,
                "batch_files": batch_files,
                "created_at": _utcnow(),
            }
        )
        operations = [
            CommitOperationAdd(
                path_in_repo=f"archives/{archive_id}/_COMPLETE.json",
                path_or_fileobj=payload.encode("utf-8"),
            )
        ]
        self.hf_api.create_commit(
            repo_id=self.repo_id,
            repo_type="dataset",
            operations=operations,
            commit_message=f"Mark archive {archive_id} complete",
        )


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
    ) -> list[Batch]:
        """Sort, deduplicate, split and batch extracted game rows.

        Rows are sorted by ``game_id`` so that duplicates are adjacent
        and every run produces identical batches. The ``utc_date`` key is
        carried into the batch metadata, not into the parquet schema.
        """
        rows = sorted(rows, key=lambda r: r["game_id"])
        deduped: list[dict] = []
        duplicates = 0
        last_id: str | None = None
        for row in rows:
            if row["game_id"] == last_id:
                duplicates += 1
                continue
            deduped.append(row)
            last_id = row["game_id"]

        groups: dict[str, list[dict]] = {"train": [], "test": []}
        for row in deduped:
            split = ChessPositionDataset._split_for_game(row["game_id"], train_ratio)
            groups[split].append(row)

        batches: list[Batch] = []
        for split in ("train", "test"):
            for seq, start in enumerate(range(0, len(groups[split]), batch_size)):
                part = groups[split][start : start + batch_size]
                table = pa.table(
                    {
                        "packed": pa.array(
                            [r["packed"] for r in part], type=pa.list_(pa.int16())
                        ),
                        "n_positions": pa.array(
                            [r["n_positions"] for r in part], type=pa.int32()
                        ),
                        "ply": pa.array([r["ply"] for r in part], type=pa.int32()),
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
            batches[0].duplicates_dropped = duplicates
        return batches

    def run(self, dry_run: bool = False, output_dir: str | None = None) -> RunResult:
        completed = self._completed(dry_run, output_dir)
        planner = ArchivePlanner(self.source_config)
        all_items = planner.plan(completed=set())
        uploader = BatchUploader(repo_id=self.dataset_config.repo_name)

        result = RunResult()
        for item in all_items:
            if item.archive_id in completed:
                result.skipped.append(item.archive_id)
                continue
            self._process(item, uploader, dry_run=dry_run, output_dir=output_dir)
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
    ) -> None:
        fingerprint = config_fingerprint(
            self.preprocessing_config.subsampling,
            self.preprocessing_config.time_control_filter,
        )
        scratch = Path(tempfile.mkdtemp(prefix="chesspos-"))
        try:
            archive_path = self._materialize(item, scratch)
            chunk_dir = scratch / "chunks"
            batch_files: list[str] = []
            chunker = PgnChunker(self.source_config.chunk_target_mb * 1024 * 1024)
            with open(archive_path, "rb") as stream:
                for chunk_index, chunk_path in enumerate(
                    chunker.iter_chunks(stream, out_dir=str(chunk_dir))
                ):
                    rows = self._extract_rows(chunk_path)
                    batches = self.build_batches(
                        rows=rows,
                        batch_size=self.dataset_config.batch_size,
                        train_ratio=self.dataset_config.train_ratio,
                        chunk_index=chunk_index,
                    )
                    for batch in batches:
                        repo_parquet = self._emit_batch(
                            batch,
                            item,
                            fingerprint,
                            uploader,
                            dry_run=dry_run,
                            output_dir=output_dir,
                            scratch=scratch,
                        )
                        if repo_parquet:
                            batch_files.append(repo_parquet)
            self._mark_complete(item, batch_files, uploader, dry_run, output_dir)
        finally:
            shutil.rmtree(scratch, ignore_errors=True)

    @staticmethod
    def _materialize(item: WorkItem, scratch: Path) -> Path:
        """Return a local, decompressed PGN path for the work item."""
        if item.provider == "lichess":
            local = download_archive(item.url, str(scratch))
        else:
            local = Path(item.path)
        if local.suffix == ".zst":
            decompressed = scratch / (local.name[: -len(".zst")])
            with open(decompressed, "wb") as out:
                for block in decompress_zst(local):
                    out.write(block)
            return decompressed
        return local

    def _extract_rows(self, chunk_path: Path) -> list[dict]:
        data = chunk_path.read_bytes()
        row = {"bytes": data, "path": str(chunk_path)}
        if self.preprocessing_config.debug:
            return ChessPositionDataset._extract_positions(
                row,
                self.preprocessing_config.subsampling,
                self.preprocessing_config.time_control_filter,
            )
        return self._extract_rows_remote(data)

    def _extract_rows_remote(self, data: bytes) -> list[dict]:
        import ray

        remote = ray.remote(ChessPositionDataset._extract_positions)
        if not ray.is_initialized():
            ray.init(address="local", ignore_reinit_error=True)
        return ray.get(
            remote.remote(
                {"bytes": data, "path": ""},
                self.preprocessing_config.subsampling,
                self.preprocessing_config.time_control_filter,
            )
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
    ) -> str | None:
        if dry_run:
            root = Path(output_dir) if output_dir else scratch / "out"
        else:
            root = scratch / "out"
        stem = f"batch_{batch.chunk_index:04d}_{batch.seq:04d}"
        parquet_path = root / batch.split / item.archive_id / f"{stem}.parquet"
        parquet_path.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(batch.table, parquet_path)
        meta = build_batch_meta(
            table=batch.table,
            file_bytes=parquet_path.read_bytes(),
            split=batch.split,
            chunk_index=batch.chunk_index,
            seq=batch.seq,
            provider=item.provider,
            archive_id=item.archive_id,
            url=item.url,
            duplicates_dropped=batch.duplicates_dropped,
            config_fingerprint=fingerprint,
            utc_dates=batch.utc_dates,
        )
        (root / batch.split / item.archive_id / f"{stem}.meta.json").write_text(
            meta.to_json()
        )
        if not dry_run:
            uploader.upload_batch(parquet_path, meta)
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
            uploader.mark_complete(item.archive_id, batch_files)
