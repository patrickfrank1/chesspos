from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterator

import numpy as np
import ray
import ray.data

from src.dataset.client import HuggingFaceClient
from src.dataset.config import DatasetConfig, EncoderConfig, PreprocessingConfig
from src.dataset.encoder import get_encoder
from src.dataset.processor import PGNProcessor
from src.dataset.types import ENCODING_SHAPES
from src.utils.fileops import file_paths_from_directory


@dataclass
class ChessPositionDataset:
    dataset_config: DatasetConfig
    preprocessing_config: PreprocessingConfig = field(
        default_factory=PreprocessingConfig
    )
    encoder_config: EncoderConfig = field(default_factory=EncoderConfig)
    hf_client: HuggingFaceClient | None = None

    def __post_init__(self):
        if self.hf_client is None:
            self.hf_client = HuggingFaceClient(repo_name=self.dataset_config.repo_name)

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
        sampling_filters = self.preprocessing_config.sampling_filters
        encoding_format = self.dataset_config.encoding
        batch_size = self.dataset_config.batch_size
        train_ratio = self.dataset_config.train_ratio

        dataset = ray.data.read_binary_files(file_paths, include_paths=True)

        def extract_positions(row: dict) -> list[dict]:
            import io

            import chess.pgn

            processor = PGNProcessor(sampling_filters=sampling_filters)
            bytes_data = row["bytes"]
            positions = []

            pgn_file = io.StringIO(bytes_data.decode("utf-8", errors="ignore"))
            while True:
                game = chess.pgn.read_game(pgn_file)
                if game is None:
                    break
                record = processor.extract_game(game)
                if record is not None:
                    for pos in record.positions:
                        positions.append(
                            {
                                "fen": pos.board.fen(),
                                "ply": pos.ply,
                                "white_elo": pos.metadata.white_elo,
                                "black_elo": pos.metadata.black_elo,
                                "result": pos.metadata.result or "",
                            }
                        )
            return positions

        def encode_batch(batch: dict) -> dict:
            import chess

            encoder = get_encoder(encoding_format)
            fens = batch["fen"]
            boards = [chess.Board(fen) for fen in fens]
            encoded = encoder.encode_batch(boards)
            return {
                "encoded": encoded,
                "ply": np.array(batch["ply"], dtype=np.int32),
                "white_elo": np.array(batch["white_elo"], dtype=np.int32),
                "black_elo": np.array(batch["black_elo"], dtype=np.int32),
                "result": batch["result"],
            }

        positions = dataset.flat_map(extract_positions)

        limited = positions.limit(batch_size)

        encoded = limited.map_batches(encode_batch, batch_format="numpy")

        train_ds, test_ds = encoded.train_test_split(train_ratio)

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
            "encoded": {
                "shape": ENCODING_SHAPES[self.dataset_config.encoding],
                "dtype": "int8",
                "description": "Encoded chess position",
            },
            "ply": {
                "shape": "()",
                "dtype": "int32",
                "description": "Move number (half-moves)",
            },
        }

        usage = f'''from datasets import load_dataset

dataset = load_dataset("{self.dataset_config.repo_name}", split="train")
for sample in dataset:
    encoded = sample["encoded"]  # {ENCODING_SHAPES[self.dataset_config.encoding]}
    ply = sample["ply"]
'''

        return self.hf_client.create_dataset_card(
            description="Chess position dataset for ML training",
            features=features,
            usage_example=usage,
        )
