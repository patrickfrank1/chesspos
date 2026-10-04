import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import chess
import numpy as np
import pytest
import ray.data

from src.dataset.config import (
    DatasetConfig,
    EncoderConfig,
    PreprocessingConfig,
    SamplingFilters,
)
from src.dataset.etl import ChessPositionDataset
from src.dataset.types import TOKEN_SEQUENCE


SAMPLE_PGN = b"""[Event "Test"]
[White "Player1"]
[Black "Player2"]
[Result "1-0"]
[WhiteElo "2200"]
[BlackElo "2100"]

1. e4 e5 2. Nf3 Nc6 3. Bb5 a6 4. Ba4 Nf6 5. O-O Be7 6. Re1 b5 7. Bb3 d6 8. c3 O-O 9. h3 Na5 10. Bc2 c5 11. d4 Qc7 1-0
"""


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
        encoding=TOKEN_SEQUENCE,
        train_ratio=0.8,
        data_path=temp_pgn_dir,
    )


@pytest.fixture
def preprocessing_config():
    return PreprocessingConfig(
        worker_count=1,
        memory_limit_mb=1024,
        sampling_filters=SamplingFilters(min_elo=1500, subsample_rate=1.0),
    )


@pytest.fixture
def encoder_config():
    return EncoderConfig(encoding_format=TOKEN_SEQUENCE, window_size=5)


@pytest.fixture
def dataset(dataset_config, preprocessing_config, encoder_config):
    with patch.object(ChessPositionDataset, "__post_init__", lambda self: None):
        return ChessPositionDataset(
            dataset_config=dataset_config,
            preprocessing_config=preprocessing_config,
            encoder_config=encoder_config,
        )


class TestChessPositionDataset:
    def test_dataset_initialization(self, dataset, dataset_config):
        assert dataset.dataset_config == dataset_config

    def test_extract_positions(self):
        row = {"bytes": SAMPLE_PGN, "path": "/fake/test.pgn"}
        filters = SamplingFilters(min_elo=0, subsample_rate=1.0)
        positions = ChessPositionDataset._extract_positions(row, filters)
        assert len(positions) > 0
        for pos in positions:
            assert "fen" in pos
            assert "ply" in pos
            assert "white_elo" in pos
            assert "black_elo" in pos
            assert "result" in pos

    def test_extract_positions_filters(self):
        row = {"bytes": SAMPLE_PGN, "path": "/fake/test.pgn"}
        filters = SamplingFilters(min_elo=3000, subsample_rate=1.0)
        positions = ChessPositionDataset._extract_positions(row, filters)
        assert len(positions) == 0

    def test_extract_positions_empty_pgn(self):
        row = {"bytes": b"", "path": "/fake/empty.pgn"}
        filters = SamplingFilters(min_elo=0, subsample_rate=1.0)
        positions = ChessPositionDataset._extract_positions(row, filters)
        assert len(positions) == 0

    def test_encode_batch(self):
        batch = {
            "fen": [chess.Board().fen() for _ in range(3)],
            "ply": np.array([1, 2, 3], dtype=np.int32),
            "white_elo": np.array([2000, 2000, 2000], dtype=np.int32),
            "black_elo": np.array([2000, 2000, 2000], dtype=np.int32),
            "result": ["1-0", "1/2-1/2", "0-1"],
        }
        result = ChessPositionDataset._encode_batch(batch, TOKEN_SEQUENCE)
        assert "encoded" in result
        assert result["encoded"].shape == (3, 69)
        np.testing.assert_array_equal(result["ply"], batch["ply"])

    def test_get_start_batch_default(self, dataset):
        assert dataset._get_start_batch(resume=False) == 1

    def test_get_start_batch_resume_unknown(self, dataset):
        mock_client = MagicMock()
        mock_client.get_next_batch_number.return_value = 5
        dataset.hf_client = mock_client
        assert dataset._get_start_batch(resume=True) == 5

    def test_generate_dry_run(self, dataset):
        mock_train = MagicMock(spec=ray.data.Dataset)
        mock_test = MagicMock(spec=ray.data.Dataset)
        dataset._process_batch = MagicMock(return_value=(mock_train, mock_test))

        batches = list(dataset.generate(num_batches=1, dry_run=True))
        assert len(batches) == 1
        assert batches[0] == (mock_train, mock_test)

    def test_create_dataset_card(self, dataset):
        mock_client = MagicMock()
        mock_client.create_dataset_card.return_value = "# Dataset Card"
        dataset.hf_client = mock_client

        card = dataset.create_dataset_card()
        assert "Dataset Card" in card
