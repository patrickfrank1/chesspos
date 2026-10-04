from __future__ import annotations

import hashlib
import io
from dataclasses import dataclass, field
from functools import partial
from typing import Iterator

import chess
import chess.pgn
import pyarrow as pa
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
        games = []

        pgn_file = io.StringIO(bytes_data.decode("utf-8", errors="ignore"))
        while True:
            game = chess.pgn.read_game(pgn_file)
            if game is None:
                break
            record = processor.extract_game(game)
            if record is not None:
                games.append(
                    {
                        "fens": [pos.board.fen() for pos in record.positions],
                        "n_positions": len(record.positions),
                        "ply": len(record.positions) - 1,
                        "game_id": ChessPositionDataset._game_id(game),
                        "white_elo": int(record.metadata.white_elo or 0),
                        "black_elo": int(record.metadata.black_elo or 0),
                        "result": record.metadata.result or "",
                    }
                )
        return games

    @staticmethod
    def _game_id(game: chess.pgn.Game) -> str:
        parts = [
            game.headers.get(key, "?")
            for key in ("Event", "Site", "Date", "Round", "White", "Black")
        ]
        identity = "|".join(parts)
        return hashlib.sha1(identity.encode("utf-8")).hexdigest()[:16]

    @staticmethod
    def _split_for_game(game_id: str, train_ratio: float) -> str:
        digest = hashlib.sha1(game_id.encode("utf-8")).digest()
        uniform = int.from_bytes(digest[:8], "big") / 2**64
        return "train" if uniform < train_ratio else "test"

    @staticmethod
    def _encode_batch(batch: pa.Table, train_ratio: float) -> pa.Table:
        encoder = TokenStreamEncoder()
        packed_rows = []
        split = []
        for fens, game_id in zip(
            batch.column("fens").to_pylist(),
            batch.column("game_id").to_pylist(),
        ):
            segments = [encoder.encode(chess.Board(fen)) for fen in fens]
            packed_rows.append(pack_stream(segments, add_cls=True))
            split.append(ChessPositionDataset._split_for_game(game_id, train_ratio))
        return pa.table(
            {
                "packed": pa.array(packed_rows, type=pa.list_(pa.int16())),
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
    ) -> Iterator[tuple[ray.data.Dataset, ray.data.Dataset]]:
        ray.init(
            num_cpus=self.preprocessing_config.worker_count,
            object_store_memory=self.preprocessing_config.memory_limit_mb * 1024 * 1024,
            ignore_reinit_error=True,
            local_mode=self.preprocessing_config.debug,
        )

        try:
            file_paths = file_paths_from_directory(
                self.dataset_config.data_path, ".pgn"
            )
            start_batch = self._get_start_batch(resume)

            for batch_num in range(start_batch, start_batch + num_batches):
                train_ds, test_ds = self._process_batch(file_paths, batch_num)

                if not dry_run:
                    self._push_batch(train_ds, test_ds, batch_num)

                yield train_ds, test_ds
        finally:
            ray.shutdown()

    def _process_batch(
        self,
        file_paths: list[str],
        batch_num: int,
    ) -> tuple[ray.data.Dataset, ray.data.Dataset]:
        subsampling = self.preprocessing_config.subsampling
        time_control_filter = self.preprocessing_config.time_control_filter
        batch_size = self.dataset_config.batch_size
        train_ratio = self.dataset_config.train_ratio

        dataset = ray.data.read_binary_files(file_paths, include_paths=True)

        extract_fn = partial(
            self._extract_positions,
            subsampling=subsampling,
            time_control_filter=time_control_filter,
        )

        games = dataset.flat_map(extract_fn)
        limited = games.limit(batch_size)
        encode_fn = partial(self._encode_batch, train_ratio=train_ratio)
        encoded = limited.map_batches(encode_fn, batch_format="pyarrow")
        encoded = encoded.materialize()
        train_ds = encoded.filter(lambda row: row["split"] == "train")
        test_ds = encoded.filter(lambda row: row["split"] == "test")

        return train_ds, test_ds

    def _get_start_batch(self, resume: bool) -> int:
        if not resume:
            return 1
        return self.hf_client.get_next_batch_number() if self.hf_client else 1

    def _push_batch(
        self,
        train_ds: ray.data.Dataset,
        test_ds: ray.data.Dataset,
        batch_num: int,
    ) -> None:
        import tempfile
        from pathlib import Path

        from huggingface_hub import HfApi

        temp_dir = Path(tempfile.mkdtemp())

        train_path = temp_dir / "train"
        test_path = temp_dir / "test"
        train_path.mkdir()
        test_path.mkdir()

        train_ds.write_parquet(str(train_path))
        test_ds.write_parquet(str(test_path))

        api = HfApi()

        for pq_file in train_path.glob("*.parquet"):
            api.upload_file(
                path_or_fileobj=str(pq_file),
                path_in_repo=f"train/batch_{batch_num:04d}_{pq_file.name}",
                repo_id=self.dataset_config.repo_name,
                repo_type="dataset",
                commit_message=f"Add train batch {batch_num}",
            )

        for pq_file in test_path.glob("*.parquet"):
            api.upload_file(
                path_or_fileobj=str(pq_file),
                path_in_repo=f"test/batch_{batch_num:04d}_{pq_file.name}",
                repo_id=self.dataset_config.repo_name,
                repo_type="dataset",
                commit_message=f"Add test batch {batch_num}",
            )

        import shutil

        shutil.rmtree(temp_dir)

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
