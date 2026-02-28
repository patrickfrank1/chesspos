from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Iterator

import ray

import chess
import numpy as np

from src.dataset.config import PreprocessingConfig
from src.dataset.encoder import PositionEncoder, get_encoder
from src.dataset.processor import PGNProcessor
from src.dataset.types import EncodingFormat, GameRecord, PositionRecord


@ray.remote
class RayPGNActor:
    def __init__(self, preprocessing_config: PreprocessingConfig):
        self.processor = PGNProcessor(
            sampling_filters=preprocessing_config.sampling_filters
        )
        self.files_processed = 0
        self.positions_extracted = 0

    def process_file(self, file_path: str) -> list[dict]:
        games_data = []
        for game_record in self.processor.process_file(file_path):
            games_data.append(game_record.to_dict())
            self.positions_extracted += len(game_record.positions)
        self.files_processed += 1
        return games_data

    def get_stats(self) -> dict:
        return {
            "files_processed": self.files_processed,
            "positions_extracted": self.positions_extracted,
        }


@ray.remote
class RayEncoderActor:
    def __init__(self, encoding_format: EncodingFormat):
        self.encoder = get_encoder(encoding_format)
        self.positions_encoded = 0

    def encode_positions(self, positions: list[dict]) -> np.ndarray:
        boards = []
        for pos_dict in positions:
            board = chess.Board(pos_dict["fen"])
            boards.append(board)

        encoded = self.encoder.encode_batch(boards)
        self.positions_encoded += len(boards)
        return encoded

    def get_stats(self) -> dict:
        return {"positions_encoded": self.positions_encoded}


@dataclass
class RayProgress:
    total_files: int = 0
    files_processed: int = 0
    total_positions: int = 0
    positions_processed: int = 0
    start_time: float = field(default_factory=time.time)

    @property
    def elapsed_seconds(self) -> float:
        return time.time() - self.start_time

    @property
    def positions_per_second(self) -> float:
        if self.elapsed_seconds == 0:
            return 0.0
        return self.positions_processed / self.elapsed_seconds

    def report(self) -> str:
        return (
            f"Files: {self.files_processed}/{self.total_files} | "
            f"Positions: {self.positions_processed} | "
            f"Rate: {self.positions_per_second:.1f}/s | "
            f"Elapsed: {self.elapsed_seconds:.1f}s"
        )


def distribute_pgn_files(
    file_paths: list[str],
    preprocessing_config: PreprocessingConfig,
) -> ray.ObjectRef:
    num_workers = preprocessing_config.worker_count
    actors = [RayPGNActor.remote(preprocessing_config) for _ in range(num_workers)]

    file_chunks = np.array_split(file_paths, num_workers)

    futures = []
    for actor, chunk in zip(actors, file_chunks):
        for file_path in chunk:
            futures.append(actor.process_file.remote(str(file_path)))

    return futures, actors


def process_with_retry(
    actor: ray.actor.ActorHandle,
    method_name: str,
    args: tuple,
    max_retries: int = 3,
    retry_delay: float = 1.0,
):
    last_error = None
    for attempt in range(max_retries):
        try:
            method = getattr(actor, method_name)
            return ray.get(method.remote(*args))
        except Exception as e:
            last_error = e
            if attempt < max_retries - 1:
                time.sleep(retry_delay * (attempt + 1))
    raise RuntimeError(f"Failed after {max_retries} retries: {last_error}")


def monitor_progress(
    actors: list[ray.actor.ActorHandle],
    progress: RayProgress,
) -> RayProgress:
    for actor in actors:
        stats = ray.get(actor.get_stats.remote())
        progress.positions_processed = max(
            progress.positions_processed,
            stats.get("positions_extracted", 0),
        )
    return progress


def init_ray(
    worker_count: int = 4,
    memory_limit_mb: int = 4096,
) -> ray.runtime_context.RuntimeContext:
    if not ray.is_initialized():
        ray.init(
            num_cpus=worker_count,
            object_store_memory=memory_limit_mb * 1024 * 1024,
            ignore_reinit_error=True,
        )
    return ray.runtime_context.get_runtime_context()


def shutdown_ray() -> None:
    if ray.is_initialized():
        ray.shutdown()
