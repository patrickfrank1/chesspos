from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Any

from src.dataset.types import EncodingFormat, TOKEN_SEQUENCE


@dataclass
class DatasetConfig:
    repo_name: str
    batch_size: int = 100_000
    encoding: EncodingFormat = TOKEN_SEQUENCE
    train_ratio: float = 0.95
    data_path: str = "./data/raw"

    def __post_init__(self):
        if not self.repo_name:
            raise ValueError("repo_name cannot be empty")
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if not 0 < self.train_ratio < 1:
            raise ValueError("train_ratio must be between 0 and 1")

    def to_json(self) -> str:
        return json.dumps(asdict(self))

    @classmethod
    def from_json(cls, json_str: str) -> "DatasetConfig":
        data = json.loads(json_str)
        return cls(**data)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class SamplingFilters:
    min_elo: int = 2000
    min_ply: int = 0
    max_ply: int | None = None
    subsample_rate: float = 0.33

    def __post_init__(self):
        if self.min_elo < 0:
            raise ValueError("min_elo cannot be negative")
        if self.min_ply < 0:
            raise ValueError("min_ply cannot be negative")
        if self.max_ply is not None and self.max_ply < self.min_ply:
            raise ValueError("max_ply cannot be less than min_ply")
        if not 0 < self.subsample_rate <= 1:
            raise ValueError("subsample_rate must be between 0 and 1")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SamplingFilters":
        return cls(**data)


@dataclass
class PreprocessingConfig:
    worker_count: int = 4
    memory_limit_mb: int = 4096
    sampling_filters: SamplingFilters = field(default_factory=SamplingFilters)
    debug: bool = False

    def __post_init__(self):
        if self.worker_count <= 0:
            raise ValueError("worker_count must be positive")
        if self.memory_limit_mb <= 0:
            raise ValueError("memory_limit_mb must be positive")

    def to_json(self) -> str:
        return json.dumps(asdict(self))

    @classmethod
    def from_json(cls, json_str: str) -> "PreprocessingConfig":
        data = json.loads(json_str)
        data["sampling_filters"] = SamplingFilters.from_dict(data["sampling_filters"])
        return cls(**data)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class EncoderConfig:
    encoding_format: EncodingFormat = TOKEN_SEQUENCE
    window_size: int = 10

    def __post_init__(self):
        if self.window_size <= 0:
            raise ValueError("window_size must be positive")

    def to_json(self) -> str:
        return json.dumps(asdict(self))

    @classmethod
    def from_json(cls, json_str: str) -> "EncoderConfig":
        data = json.loads(json_str)
        return cls(**data)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
