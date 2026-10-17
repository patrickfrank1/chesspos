import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

from etl_support import SAMPLE_PGN
from src.dataset.config import (
    DatasetConfig,
    GameSubsampling,
    GameSubsampleTier,
    PreprocessingConfig,
)
from src.dataset.etl import ChessPositionDataset


@pytest.fixture
def temp_pgn_dir():
    with tempfile.TemporaryDirectory() as tmpdir:
        pgn_path = Path(tmpdir) / "test.pgn"
        pgn_path.write_bytes(SAMPLE_PGN)
        yield str(tmpdir)


@pytest.fixture
def dataset_config(temp_pgn_dir):
    return DatasetConfig(
        repo_name="test/chesspos-test",
        batch_size=10,
        train_ratio=0.8,
        data_path=temp_pgn_dir,
    )


@pytest.fixture
def preprocessing_config():
    return PreprocessingConfig(
        worker_count=1,
        memory_limit_mb=1024,
        subsampling=GameSubsampling(tiers=[GameSubsampleTier(min_elo=0, rate=1.0)]),
    )


@pytest.fixture
def dataset(dataset_config, preprocessing_config):
    with patch.object(ChessPositionDataset, "__post_init__", lambda self: None):
        return ChessPositionDataset(
            dataset_config=dataset_config,
            preprocessing_config=preprocessing_config,
        )
