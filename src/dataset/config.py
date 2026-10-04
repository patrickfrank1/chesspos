from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class DatasetConfig:
    repo_name: str
    batch_size: int = 100_000
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

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "DatasetConfig":
        return cls(**data)


@dataclass
class GameSubsampleTier:
    """Acceptance tier for games by player strength.

    A game qualifies for a tier when both players' ratings are at least
    ``min_elo``. It is assigned to the strictest qualifying tier and kept
    with probability ``rate``. Games with missing ratings only qualify for
    the ``min_elo=0`` tier.
    """

    min_elo: int = 0
    rate: float = 1.0

    def __post_init__(self):
        if self.min_elo < 0:
            raise ValueError("min_elo cannot be negative")
        if not 0 < self.rate <= 1:
            raise ValueError("rate must be between 0 and 1")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "GameSubsampleTier":
        return cls(**data)


@dataclass
class GameSubsampling:
    """Tiered subsampling of games by metadata (player strength).

    Tiers are evaluated per game: the game is matched against the strictest
    tier it qualifies for and accepted with that tier's rate. Include a
    ``min_elo=0`` catch-all tier to keep weak or unrated games.
    """

    tiers: list[GameSubsampleTier] = field(
        default_factory=lambda: [
            GameSubsampleTier(min_elo=2500, rate=0.40),
            GameSubsampleTier(min_elo=2200, rate=0.30),
            GameSubsampleTier(min_elo=1800, rate=0.25),
            GameSubsampleTier(min_elo=0, rate=0.05),
        ]
    )

    def to_dict(self) -> dict[str, Any]:
        return {"tiers": [tier.to_dict() for tier in self.tiers]}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "GameSubsampling":
        return cls(
            tiers=[GameSubsampleTier.from_dict(t) for t in data.get("tiers", [])]
        )


@dataclass
class PreprocessingConfig:
    worker_count: int = 4
    memory_limit_mb: int = 4096
    subsampling: GameSubsampling = field(default_factory=GameSubsampling)
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
        data["subsampling"] = GameSubsampling.from_dict(data["subsampling"])
        return cls(**data)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "PreprocessingConfig":
        return cls(**data)
