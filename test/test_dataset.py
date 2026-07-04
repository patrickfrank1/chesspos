import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from src.dataset.config import (
    DatasetConfig,
    EncoderConfig,
    PreprocessingConfig,
    SamplingFilters,
)
from src.dataset.etl import ChessPositionDataset


SAMPLE_PGN = """[Event "Test"]
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
        pgn_path.write_text(SAMPLE_PGN)
        yield str(tmpdir)


@pytest.fixture
def dataset_config(temp_pgn_dir):
    return DatasetConfig(
        repo_name="test/chesspos-test",
        batch_size=10,
        encoding="token_sequence",
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
    return EncoderConfig(encoding_format="token_sequence", window_size=5)


class TestChessPositionDataset:
    def test_dataset_initialization(
        self, dataset_config, preprocessing_config, encoder_config
    ):
        with patch.object(ChessPositionDataset, "__post_init__", lambda self: None):
            dataset = ChessPositionDataset(
                dataset_config=dataset_config,
                preprocessing_config=preprocessing_config,
                encoder_config=encoder_config,
            )
            assert dataset.dataset_config == dataset_config

    def test_encoder_property(
        self, dataset_config, preprocessing_config, encoder_config
    ):
        with patch.object(ChessPositionDataset, "__post_init__", lambda self: None):
            dataset = ChessPositionDataset(
                dataset_config=dataset_config,
                preprocessing_config=preprocessing_config,
                encoder_config=encoder_config,
            )
            encoder = dataset.encoder
            assert encoder.encoding_format == "token_sequence"

    def test_split_data(self, dataset_config, preprocessing_config, encoder_config):
        with patch.object(ChessPositionDataset, "__post_init__", lambda self: None):
            dataset = ChessPositionDataset(
                dataset_config=dataset_config,
                preprocessing_config=preprocessing_config,
                encoder_config=encoder_config,
            )

            data = np.zeros((100, 69), dtype=np.int8)
            train, test = dataset._split_data(data)

            assert len(train) == 80
            assert len(test) == 20

    def test_array_to_dict_shape(
        self, dataset_config, preprocessing_config, encoder_config
    ):
        with patch.object(ChessPositionDataset, "__post_init__", lambda self: None):
            dataset = ChessPositionDataset(
                dataset_config=dataset_config,
                preprocessing_config=preprocessing_config,
                encoder_config=encoder_config,
            )

            data = np.zeros((50, 69), dtype=np.int8)
            result = dataset._array_to_dict(data)

            assert "window" in result
            assert "scalars" in result
            assert result["window"].shape == (50, 5, 69)
            assert result["scalars"].shape == (50, 3)

    def test_validate_schema_valid(
        self, dataset_config, preprocessing_config, encoder_config
    ):
        with patch.object(ChessPositionDataset, "__post_init__", lambda self: None):
            dataset = ChessPositionDataset(
                dataset_config=dataset_config,
                preprocessing_config=preprocessing_config,
                encoder_config=encoder_config,
            )

            data = np.zeros((10, 69), dtype=np.int8)
            assert dataset.validate_schema(data) is True

    def test_validate_schema_invalid(
        self, dataset_config, preprocessing_config, encoder_config
    ):
        with patch.object(ChessPositionDataset, "__post_init__", lambda self: None):
            dataset = ChessPositionDataset(
                dataset_config=dataset_config,
                preprocessing_config=preprocessing_config,
                encoder_config=encoder_config,
            )

            data = np.zeros((10, 8, 8, 15), dtype=np.int8)
            with pytest.raises(ValueError, match="does not match expected"):
                dataset.validate_schema(data)

    def test_create_dataset_card(
        self, dataset_config, preprocessing_config, encoder_config
    ):
        mock_client = MagicMock()
        mock_client.create_dataset_card.return_value = "# Dataset Card"

        with patch.object(ChessPositionDataset, "__post_init__", lambda self: None):
            dataset = ChessPositionDataset(
                dataset_config=dataset_config,
                preprocessing_config=preprocessing_config,
                encoder_config=encoder_config,
            )
            dataset.hf_client = mock_client

            card = dataset.create_dataset_card()
            assert "Dataset Card" in card

    def test_generate_dry_run(
        self, dataset_config, preprocessing_config, encoder_config
    ):
        with patch.object(ChessPositionDataset, "__post_init__", lambda self: None):
            dataset = ChessPositionDataset(
                dataset_config=dataset_config,
                preprocessing_config=preprocessing_config,
                encoder_config=encoder_config,
            )

            batches = list(dataset.generate(num_batches=1, dry_run=True))
            assert len(batches) == 1
            train, test = batches[0]
            assert train.shape[1] == 69
            assert test.shape[1] == 69
