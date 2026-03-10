from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterator

import numpy as np
import ray

from src.dataset.client import HuggingFaceClient
from src.dataset.config import DatasetConfig, EncoderConfig, PreprocessingConfig
from src.dataset.encoder import PositionEncoder, get_encoder
from src.dataset.ray_actors import (
    RayEncoderActor,
    RayPGNActor,
    RayProgress,
    init_ray,
    shutdown_ray,
)
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
    _encoder: PositionEncoder | None = field(default=None, repr=False)

    def __post_init__(self):
        self._encoder = get_encoder(self.dataset_config.encoding)
        if self.hf_client is None:
            self.hf_client = HuggingFaceClient(repo_name=self.dataset_config.repo_name)

    @property
    def encoder(self) -> PositionEncoder:
        if self._encoder is None:
            self._encoder = get_encoder(self.dataset_config.encoding)
        return self._encoder

    def generate(
        self,
        num_batches: int = 1,
        resume: bool = False,
        dry_run: bool = False,
    ) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        init_ray(
            worker_count=self.preprocessing_config.worker_count,
            memory_limit_mb=self.preprocessing_config.memory_limit_mb,
        )

        try:
            file_paths = file_paths_from_directory(
                self.dataset_config.data_path, ".pgn"
            )
            start_batch = self._get_start_batch(resume)

            progress = RayProgress(total_files=len(file_paths))

            for batch_num in range(start_batch, start_batch + num_batches):
                batch_positions = self._collect_batch_positions(file_paths, progress)

                if len(batch_positions) < self.dataset_config.batch_size:
                    print(f"Warning: Only collected {len(batch_positions)} positions")

                encoded = self._encode_positions(batch_positions)

                train_data, test_data = self._split_data(encoded)

                if not dry_run:
                    self._push_batch(train_data, test_data, batch_num)

                yield train_data, test_data
        finally:
            shutdown_ray()

    def _get_start_batch(self, resume: bool) -> int:
        if not resume:
            return 1
        return self.hf_client.get_next_batch_number() if self.hf_client else 1

    def _collect_batch_positions(
        self,
        file_paths: list[str],
        progress: RayProgress,
    ) -> list[dict]:
        num_workers = self.preprocessing_config.worker_count
        actors = [
            RayPGNActor.remote(self.preprocessing_config) for _ in range(num_workers)
        ]

        file_chunks = np.array_split(file_paths, num_workers)
        futures = []
        for actor, chunk in zip(actors, file_chunks):
            for file_path in chunk:
                futures.append(actor.process_file.remote(str(file_path)))

        positions = []
        completed = 0

        while futures and len(positions) < self.dataset_config.batch_size:
            ready, futures = ray.wait(futures, num_returns=1)
            for future in ready:
                games_data = ray.get(future)
                for game in games_data:
                    for pos in game["positions"]:
                        positions.append(pos)
                        if len(positions) >= self.dataset_config.batch_size:
                            progress.files_processed = completed
                            return positions
                completed += 1

        progress.files_processed = completed
        return positions

    def _encode_positions(self, positions: list[dict]) -> np.ndarray:
        num_workers = self.preprocessing_config.worker_count
        actors = [
            RayEncoderActor.remote(self.dataset_config.encoding)
            for _ in range(num_workers)
        ]

        chunk_size = max(1, len(positions) // num_workers)
        chunks = [
            positions[i : i + chunk_size] for i in range(0, len(positions), chunk_size)
        ]

        futures = []
        for actor, chunk in zip(actors, chunks):
            if chunk:
                futures.append(actor.encode_positions.remote(chunk))

        encoded_chunks = ray.get(futures)
        return np.concatenate(encoded_chunks, axis=0)

    def _split_data(self, data: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        split_idx = int(len(data) * self.dataset_config.train_ratio)
        return data[:split_idx], data[split_idx:]

    def _push_batch(
        self,
        train_data: np.ndarray,
        test_data: np.ndarray,
        batch_num: int,
    ) -> None:
        train_dict = self._array_to_dict(train_data)
        test_dict = self._array_to_dict(test_data)

        self.hf_client.push_batch(
            train_dict,
            split="train",
            batch_number=batch_num,
        )
        self.hf_client.push_batch(
            test_dict,
            split="test",
            batch_number=batch_num,
        )

    def _array_to_dict(self, data: np.ndarray) -> dict[str, np.ndarray]:
        shape = ENCODING_SHAPES[self.dataset_config.encoding]
        window_size = self.encoder_config.window_size

        num_samples = len(data)
        window_data = np.zeros((num_samples, window_size, *shape), dtype=np.int8)
        for i in range(num_samples):
            for j in range(window_size):
                window_data[i, j] = data[i]

        scalars = np.zeros((num_samples, 3), dtype=np.int8)

        return {
            "window": window_data,
            "scalars": scalars,
        }

    def validate_schema(self, data: np.ndarray) -> bool:
        expected_shape = ENCODING_SHAPES[self.dataset_config.encoding]
        if data.shape[1:] != expected_shape:
            raise ValueError(
                f"Data shape {data.shape[1:]} does not match expected {expected_shape}"
            )
        return True

    def create_dataset_card(self) -> str:
        features = {
            "window": {
                "shape": f"({self.encoder_config.window_size}, {ENCODING_SHAPES[self.dataset_config.encoding]})",
                "dtype": "int8",
                "description": "Temporal window of encoded positions",
            },
            "scalars": {
                "shape": "(3,)",
                "dtype": "int8",
                "description": "Placeholder scalars for future use",
            },
        }

        usage = f'''from datasets import load_dataset

dataset = load_dataset("{self.dataset_config.repo_name}", split="train")
for sample in dataset:
    window = sample["window"]  # ({self.encoder_config.window_size}, 69)
    scalars = sample["scalars"]  # (3,)
'''

        return self.hf_client.create_dataset_card(
            description="Chess position dataset for ML training",
            features=features,
            usage_example=usage,
        )
