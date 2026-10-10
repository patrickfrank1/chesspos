from __future__ import annotations

import hashlib
import io
import os
import random
from dataclasses import dataclass, field
from functools import partial
from typing import Iterator

import chess
import chess.pgn
import pyarrow as pa
import pyarrow.parquet as pq
import ray
import ray.data

from src.dataset.huggingface_client import HuggingFaceClient
from src.dataset.config import (
    DatasetConfig,
    PreprocessingConfig,
    GameSubsampling,
    TimeControlFilter,
)
from src.dataset.pgn_processor import PGNProcessor
from src.dataset.token_stream import TokenStreamEncoder, pack_stream
from src.utils.fileops import file_paths_from_directory


@dataclass
class ChessPositionDataset:
    dataset_config: DatasetConfig
    preprocessing_config: PreprocessingConfig = field(
        default_factory=PreprocessingConfig
    )
    hf_client: HuggingFaceClient | None = None

    def __post_init__(self):
        if self.hf_client is None:
            self.hf_client = HuggingFaceClient(repo_name=self.dataset_config.repo_name)

    @staticmethod
    def _extract_positions(
        row: dict,
        subsampling: GameSubsampling,
        time_control_filter: TimeControlFilter,
    ) -> list[dict]:
        processor = PGNProcessor(
            subsampling=subsampling,
            time_control_filter=time_control_filter,
        )
        bytes_data = row["bytes"]
        random.seed(int.from_bytes(hashlib.sha1(bytes_data).digest()[:8], "big"))
        games = []

        pgn_file = io.StringIO(bytes_data.decode("utf-8", errors="ignore"))

        # Pass 1: header-only scan (no movetext parsing); record the offset of
        # every game that passes the filters so the expensive parse can skip
        # the rest.
        keep_offsets = []
        while True:
            offset = pgn_file.tell()
            headers = chess.pgn.read_headers(pgn_file)
            if headers is None:
                break
            if processor.keep_game(headers):
                keep_offsets.append(offset)

        # Pass 2: fully parse only the games that passed the filters, and
        # encode tokens straight from the live board (no per-position board
        # copies, no FEN roundtrip).
        encoder = TokenStreamEncoder()
        for offset in keep_offsets:
            pgn_file.seek(offset)
            game = chess.pgn.read_game(pgn_file)
            if game is None:
                continue
            metadata = processor.extract_metadata(game.headers)
            board = chess.Board()
            segments = []
            for move in game.mainline_moves():
                board.push(move)
                segments.append(encoder.encode(board))
            if not segments:
                continue
            games.append(
                {
                    "packed": pack_stream(segments, add_cls=True).tolist(),
                    "n_positions": len(segments),
                    "ply": len(segments) - 1,
                    "game_id": ChessPositionDataset._game_id(game),
                    "white_elo": int(metadata.white_elo or 0),
                    "black_elo": int(metadata.black_elo or 0),
                    "result": metadata.result or "",
                }
            )
        return games

    @staticmethod
    def _game_id(game: chess.pgn.Game) -> str:
        parts = [
            game.headers.get(key, "?")
            for key in (
                "Event",
                "Site",
                "Date",
                "Round",
                "White",
                "Black",
                "WhiteElo",
                "BlackElo",
            )
        ]
        identity = "|".join(parts)
        return hashlib.sha1(identity.encode("utf-8")).hexdigest()[:16]

    @staticmethod
    def _split_for_game(game_id: str, train_ratio: float) -> str:
        digest = hashlib.sha1(game_id.encode("utf-8")).digest()
        uniform = int.from_bytes(digest[:8], "big") / 2**64
        return "train" if uniform < train_ratio else "test"

    @staticmethod
    def _finalize_batch(batch: pa.Table, train_ratio: float) -> pa.Table:
        split = [
            ChessPositionDataset._split_for_game(game_id, train_ratio)
            for game_id in batch.column("game_id").to_pylist()
        ]
        return pa.table(
            {
                "packed": batch.column("packed").cast(pa.list_(pa.int16())),
                "n_positions": batch.column("n_positions").cast(pa.int32()),
                "ply": batch.column("ply").cast(pa.int32()),
                "game_id": batch.column("game_id"),
                "white_elo": batch.column("white_elo").cast(pa.int32()),
                "black_elo": batch.column("black_elo").cast(pa.int32()),
                "result": batch.column("result"),
                "split": pa.array(split, type=pa.string()),
            }
        )

    def generate(
        self,
        num_batches: int = 1,
        resume: bool = False,
        dry_run: bool = False,
        output_dir: str | None = None,
    ) -> Iterator[tuple[ray.data.Dataset, ray.data.Dataset]]:
        ray_kwargs: dict = {
            "ignore_reinit_error": True,
            "local_mode": self.preprocessing_config.debug,
        }
        if os.environ.get("RAY_ADDRESS") is None:
            ray_kwargs["num_cpus"] = self.preprocessing_config.worker_count
            ray_kwargs["object_store_memory"] = (
                self.preprocessing_config.memory_limit_mb * 1024 * 1024
            )
        try:
            ray.init(**ray_kwargs)
        except ValueError:
            ray_kwargs.pop("num_cpus", None)
            ray_kwargs.pop("object_store_memory", None)
            ray.init(**ray_kwargs)

        try:
            file_paths = file_paths_from_directory(
                self.dataset_config.data_path, ".pgn"
            )
            encoded = self._build_encoded(file_paths)
            start_batch = self._get_start_batch(resume)
            batch_size = self.dataset_config.batch_size

            produced = 0
            batch_num = 0
            stream = encoded.iter_batches(batch_size=batch_size, batch_format="pyarrow")
            for chunk in self._iter_deduped_chunks(stream, batch_size):
                batch_num += 1
                if batch_num < start_batch:
                    continue
                if produced >= num_batches:
                    break

                train_table, test_table = self._split_chunk(chunk)
                train_ds = ray.data.from_arrow(train_table)
                test_ds = ray.data.from_arrow(test_table)

                if not dry_run:
                    self._push_batch(train_table, test_table, batch_num)
                elif output_dir:
                    self._write_batch(train_table, test_table, batch_num, output_dir)

                produced += 1
                yield train_ds, test_ds
        finally:
            ray.shutdown()

    def _build_encoded(self, file_paths: list[str]) -> ray.data.Dataset:
        subsampling = self.preprocessing_config.subsampling
        time_control_filter = self.preprocessing_config.time_control_filter
        train_ratio = self.dataset_config.train_ratio

        dataset = ray.data.read_binary_files(file_paths, include_paths=True)
        extract_fn = partial(
            self._extract_positions,
            subsampling=subsampling,
            time_control_filter=time_control_filter,
        )
        games = dataset.flat_map(extract_fn)
        finalize_fn = partial(self._finalize_batch, train_ratio=train_ratio)
        encoded = games.map_batches(finalize_fn, batch_format="pyarrow")
        # Global deterministic order (by game_id) so that fixed-size batch
        # chunks contain the same games in every run.
        return encoded.sort("game_id").materialize()

    @staticmethod
    def _dedup_chunk(
        chunk: pa.Table, last_game_id: str | None
    ) -> tuple[pa.Table, str | None]:
        """Drop consecutive duplicate games.

        The dataset is globally sorted by game_id, so all copies of a
        duplicated game are adjacent in the stream. Comparing each row to its
        predecessor (with carry-over across chunk boundaries) therefore yields
        an exact global dedup without an extra shuffle.
        """
        ids = chunk.column("game_id").to_pylist()
        keep = [ids[0] != last_game_id]
        keep.extend(ids[i] != ids[i - 1] for i in range(1, len(ids)))
        deduped = chunk.filter(pa.array(keep, type=pa.bool_()))
        return deduped, ids[-1]

    @classmethod
    def _iter_deduped_chunks(
        cls, stream: Iterator[pa.Table], batch_size: int
    ) -> Iterator[pa.Table]:
        """Yield deduplicated chunks of exactly batch_size games (last may be short).

        Deduplication runs across the whole stream (duplicates are adjacent in
        the sorted order), and a rolling buffer tops each chunk back up to
        batch_size so fixed-size batches survive row removal.
        """
        last_game_id: str | None = None
        buffer: pa.Table | None = None
        for raw in stream:
            deduped, last_game_id = cls._dedup_chunk(raw, last_game_id)
            if deduped.num_rows == 0:
                continue
            buffer = deduped if buffer is None else pa.concat_tables([buffer, deduped])
            while buffer.num_rows >= batch_size:
                yield buffer.slice(0, batch_size)
                buffer = buffer.slice(batch_size)
        if buffer is not None and buffer.num_rows > 0:
            yield buffer

    @staticmethod
    def _split_chunk(chunk: pa.Table) -> tuple[pa.Table, pa.Table]:
        split_col = chunk.column("split")
        train_mask = pa.compute.equal(split_col, "train")
        train_table = chunk.filter(train_mask)
        test_table = chunk.filter(pa.compute.invert(train_mask))
        return train_table, test_table

    def _get_start_batch(self, resume: bool) -> int:
        if not resume:
            return 1
        return self.hf_client.get_next_batch_number() if self.hf_client else 1

    def _push_batch(
        self,
        train_table: pa.Table,
        test_table: pa.Table,
        batch_num: int,
    ) -> None:
        import tempfile
        from pathlib import Path

        from huggingface_hub import HfApi

        api = HfApi()

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            for split, table in (("train", train_table), ("test", test_table)):
                file_name = f"batch_{batch_num:04d}_000000.parquet"
                local_path = temp_path / split / file_name
                local_path.parent.mkdir()
                pq.write_table(table, local_path)
                api.upload_file(
                    path_or_fileobj=str(local_path),
                    path_in_repo=f"{split}/{file_name}",
                    repo_id=self.dataset_config.repo_name,
                    repo_type="dataset",
                    commit_message=f"Add {split} batch {batch_num}",
                )

    def _write_batch(
        self,
        train_table: pa.Table,
        test_table: pa.Table,
        batch_num: int,
        output_dir: str,
    ) -> None:
        from pathlib import Path

        out = Path(output_dir)
        for split, table in (("train", train_table), ("test", test_table)):
            split_dir = out / split
            split_dir.mkdir(parents=True, exist_ok=True)
            pq.write_table(
                table,
                split_dir / f"batch_{batch_num:04d}_000000.parquet",
            )

    def create_dataset_card(self) -> str:
        features = {
            "packed": {
                "dtype": "int16",
                "description": (
                    "Packed token stream for one game: CLS-prefixed, one "
                    "variable-length segment per position separated by SEP "
                    "(see token_stream.py for the segment layout)"
                ),
            },
            "n_positions": {
                "shape": "()",
                "dtype": "int32",
                "description": "Number of encoded positions in the game",
            },
            "ply": {
                "shape": "()",
                "dtype": "int32",
                "description": "Ply index of the final position (n_positions - 1)",
            },
            "game_id": {
                "dtype": "string",
                "description": "Deterministic hash of the PGN identifying headers",
            },
            "white_elo": {
                "shape": "()",
                "dtype": "int32",
                "description": "White player rating (0 when missing)",
            },
            "black_elo": {
                "shape": "()",
                "dtype": "int32",
                "description": "Black player rating (0 when missing)",
            },
            "result": {
                "dtype": "string",
                "description": "Game result header (1-0, 0-1, 1/2-1/2)",
            },
        }

        usage = f'''from datasets import load_dataset

dataset = load_dataset("{self.dataset_config.repo_name}", split="train")
for sample in dataset:
    packed = sample["packed"]
    game_id = sample["game_id"]
'''

        return self.hf_client.create_dataset_card(
            description="Chess position dataset for ML training",
            features=features,
            usage_example=usage,
        )
