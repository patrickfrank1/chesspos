import io
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import chess
import numpy as np
import pyarrow as pa
import pytest
import ray.data

from src.dataset.config import (
    DatasetConfig,
    GameSubsampling,
    GameSubsampleTier,
    PreprocessingConfig,
    TimeControlFilter,
)
from src.dataset.etl import ChessPositionDataset
from src.dataset.token_stream import CLS, SEP


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


class TestChessPositionDataset:
    def test_dataset_initialization(self, dataset, dataset_config):
        assert dataset.dataset_config == dataset_config

    def test_extract_positions(self):
        row = {"bytes": SAMPLE_PGN, "path": "/fake/test.pgn"}
        subsampling = GameSubsampling(tiers=[GameSubsampleTier(min_elo=0, rate=1.0)])
        games = ChessPositionDataset._extract_positions(
            row, subsampling, TimeControlFilter()
        )
        assert len(games) == 1
        game = games[0]
        assert len(game["fens"]) == game["n_positions"]
        assert len(game["fens"]) > 0
        assert game["ply"] == game["n_positions"] - 1
        assert game["white_elo"] == 2200
        assert game["black_elo"] == 2100
        assert game["result"] == "1-0"
        assert len(game["game_id"]) == 16

    def test_game_id_is_deterministic(self):
        pgn_file = chess.pgn.read_game(io.StringIO(SAMPLE_PGN.decode("utf-8")))
        assert ChessPositionDataset._game_id(pgn_file) == ChessPositionDataset._game_id(
            pgn_file
        )

    def test_extract_positions_filters(self):
        row = {"bytes": SAMPLE_PGN, "path": "/fake/test.pgn"}
        subsampling = GameSubsampling(tiers=[GameSubsampleTier(min_elo=3000, rate=1.0)])
        games = ChessPositionDataset._extract_positions(
            row, subsampling, TimeControlFilter()
        )
        assert len(games) == 0

    def test_extract_positions_empty_pgn(self):
        row = {"bytes": b"", "path": "/fake/empty.pgn"}
        subsampling = GameSubsampling(tiers=[GameSubsampleTier(min_elo=0, rate=1.0)])
        games = ChessPositionDataset._extract_positions(
            row, subsampling, TimeControlFilter()
        )
        assert len(games) == 0

    def test_extract_positions_filters_bullet(self):
        bullet_pgn = SAMPLE_PGN.replace(
            b'[Result "1-0"]', b'[Result "1-0"]\n[TimeControl "180+0"]'
        )
        row = {"bytes": bullet_pgn, "path": "/fake/bullet.pgn"}
        subsampling = GameSubsampling(tiers=[GameSubsampleTier(min_elo=0, rate=1.0)])
        games = ChessPositionDataset._extract_positions(
            row, subsampling, TimeControlFilter(min_seconds=300)
        )
        assert len(games) == 0

    def test_extract_positions_missing_elo_defaults_to_zero(self):
        no_elo_pgn = SAMPLE_PGN.replace(b'[WhiteElo "2200"]\n', b"").replace(
            b'[BlackElo "2100"]\n', b""
        )
        row = {"bytes": no_elo_pgn, "path": "/fake/no_elo.pgn"}
        subsampling = GameSubsampling(tiers=[GameSubsampleTier(min_elo=0, rate=1.0)])
        games = ChessPositionDataset._extract_positions(
            row, subsampling, TimeControlFilter()
        )
        assert len(games) == 1
        assert games[0]["white_elo"] == 0
        assert games[0]["black_elo"] == 0

    def test_split_for_game_is_deterministic(self):
        first = ChessPositionDataset._split_for_game("abc123", 0.8)
        second = ChessPositionDataset._split_for_game("abc123", 0.8)
        assert first == second
        assert first in {"train", "test"}

    def test_split_for_game_extremes(self):
        assert ChessPositionDataset._split_for_game("abc123", 1.0) == "train"
        assert ChessPositionDataset._split_for_game("abc123", 0.0) == "test"

    def _make_batch(self) -> pa.Table:
        boards = [chess.Board(), chess.Board()]
        boards[1].push_san("e4")
        return pa.table(
            {
                "fens": pa.array(
                    [[b.fen() for b in boards]], type=pa.list_(pa.string())
                ),
                "n_positions": pa.array([2], type=pa.int32()),
                "ply": pa.array([1], type=pa.int32()),
                "game_id": pa.array(["deadbeefdeadbeef"]),
                "white_elo": pa.array([2000], type=pa.int32()),
                "black_elo": pa.array([2000], type=pa.int32()),
                "result": pa.array(["1-0"]),
            }
        )

    def test_encode_batch(self):
        result = ChessPositionDataset._encode_batch(self._make_batch(), 0.8)
        packed = result.column("packed").to_pylist()[0]
        packed = np.asarray(packed, dtype=np.int16)
        assert packed.dtype == np.int16
        assert packed[0] == CLS
        assert SEP in packed.tolist()
        assert result.column("n_positions").to_pylist() == [2]
        assert result.column("ply").to_pylist() == [1]
        assert result.column("game_id").to_pylist() == ["deadbeefdeadbeef"]
        assert result.column("split").to_pylist()[0] in {"train", "test"}

    def test_encode_batch_split_is_deterministic(self):
        first = ChessPositionDataset._encode_batch(self._make_batch(), 0.8)
        second = ChessPositionDataset._encode_batch(self._make_batch(), 0.8)
        assert first.column("split").to_pylist() == second.column("split").to_pylist()

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
