import io
from unittest.mock import MagicMock

import chess
import numpy as np

from bdd import given, then, when
from etl_support import SAMPLE_PGN, make_batch, make_lichess_pgn
from src.dataset.config import GameSubsampling, GameSubsampleTier, TimeControlFilter
from src.dataset.etl import ChessPositionDataset
from src.dataset.token_stream import CLS, SEP


class TestChessPositionDataset:
    def test_dataset_initialization(self, dataset, dataset_config):
        with given("a ChessPositionDataset built from a dataset config"):
            stored = dataset.dataset_config

        with then("the config is stored on the dataset"):
            assert stored == dataset_config

    def test_extract_positions(self):
        with given("a raw PGN row and permissive subsampling"):
            row = {"bytes": SAMPLE_PGN, "path": "/fake/test.pgn"}
            subsampling = GameSubsampling(
                tiers=[GameSubsampleTier(min_elo=0, rate=1.0)]
            )

        with when("positions are extracted from the game"):
            games = ChessPositionDataset._extract_positions(
                row, subsampling, TimeControlFilter()
            )

        with then("one game record with valid packed tokens and metadata is returned"):
            assert len(games) == 1
            game = games[0]
            assert game["n_positions"] > 0
            assert game["ply"] == game["n_positions"] - 1
            assert len(game["packed"]) >= game["n_positions"]
            assert game["packed"][0] == CLS
            assert SEP in game["packed"]
            assert game["white_elo"] == 2200
            assert game["black_elo"] == 2100
            assert game["result"] == "1-0"
            assert len(game["game_id"]) == 16

    def test_game_id_is_deterministic(self):
        with given("a parsed PGN game"):
            pgn_file = chess.pgn.read_game(io.StringIO(SAMPLE_PGN.decode("utf-8")))

        with when("a game id is derived twice from the same game"):
            first = ChessPositionDataset._game_id(pgn_file)
            second = ChessPositionDataset._game_id(pgn_file)

        with then("both derivations produce the same id"):
            assert first == second

    def test_extract_positions_filters(self):
        with given("a raw PGN row and a subsampling tier above the game's rating"):
            row = {"bytes": SAMPLE_PGN, "path": "/fake/test.pgn"}
            subsampling = GameSubsampling(
                tiers=[GameSubsampleTier(min_elo=3000, rate=1.0)]
            )

        with when("positions are extracted from the game"):
            games = ChessPositionDataset._extract_positions(
                row, subsampling, TimeControlFilter()
            )

        with then("the game is dropped"):
            assert len(games) == 0

    def test_extract_positions_empty_pgn(self):
        with given("a row containing empty PGN bytes"):
            row = {"bytes": b"", "path": "/fake/empty.pgn"}
            subsampling = GameSubsampling(
                tiers=[GameSubsampleTier(min_elo=0, rate=1.0)]
            )

        with when("positions are extracted"):
            games = ChessPositionDataset._extract_positions(
                row, subsampling, TimeControlFilter()
            )

        with then("no games are produced"):
            assert len(games) == 0

    def test_extract_positions_filters_bullet(self):
        with given("a bullet game PGN row and a 300s minimum time control"):
            bullet_pgn = SAMPLE_PGN.replace(
                b'[Result "1-0"]', b'[Result "1-0"]\n[TimeControl "180+0"]'
            )
            row = {"bytes": bullet_pgn, "path": "/fake/bullet.pgn"}
            subsampling = GameSubsampling(
                tiers=[GameSubsampleTier(min_elo=0, rate=1.0)]
            )

        with when("positions are extracted"):
            games = ChessPositionDataset._extract_positions(
                row, subsampling, TimeControlFilter(min_seconds=300)
            )

        with then("the game is dropped"):
            assert len(games) == 0

    def test_extract_positions_missing_elo_defaults_to_zero(self):
        with given("a PGN row without Elo headers"):
            no_elo_pgn = SAMPLE_PGN.replace(b'[WhiteElo "2200"]\n', b"").replace(
                b'[BlackElo "2100"]\n', b""
            )
            row = {"bytes": no_elo_pgn, "path": "/fake/no_elo.pgn"}
            subsampling = GameSubsampling(
                tiers=[GameSubsampleTier(min_elo=0, rate=1.0)]
            )

        with when("positions are extracted"):
            games = ChessPositionDataset._extract_positions(
                row, subsampling, TimeControlFilter()
            )

        with then("both Elo fields default to zero"):
            assert len(games) == 1
            assert games[0]["white_elo"] == 0
            assert games[0]["black_elo"] == 0

    def test_extract_positions_captures_utc_date(self):
        with given("a lichess-style PGN row with UTCDate headers"):
            row = {
                "bytes": make_lichess_pgn("Test", date="2023.01.15"),
                "path": "/fake/x.pgn",
            }
            subsampling = GameSubsampling(
                tiers=[GameSubsampleTier(min_elo=0, rate=1.0)]
            )

        with when("positions are extracted"):
            games = ChessPositionDataset._extract_positions(
                row, subsampling, TimeControlFilter()
            )

        with then("each game record carries its UTCDate as an ISO date"):
            assert len(games) == 1
            assert games[0]["utc_date"] == "2023-01-15"

    def test_split_for_game_is_deterministic(self):
        with when("the split is computed twice for a fixed game id and ratio"):
            first = ChessPositionDataset._split_for_game("abc123", 0.8)
            second = ChessPositionDataset._split_for_game("abc123", 0.8)

        with then("both computations agree and yield a valid split"):
            assert first == second
            assert first in {"train", "test"}

    def test_split_for_game_extremes(self):
        with when("the split is computed at ratio 1.0 and 0.0"):
            all_train = ChessPositionDataset._split_for_game("abc123", 1.0)
            all_test = ChessPositionDataset._split_for_game("abc123", 0.0)

        with then("the game lands in train and test respectively"):
            assert all_train == "train"
            assert all_test == "test"

    def test_finalize_batch(self):
        with given("an un-finalized batch of one game"):
            batch = make_batch()

        with when("the batch is finalized with train ratio 0.8"):
            result = ChessPositionDataset._finalize_batch(batch, 0.8)

        with then("packed tokens are int16 with CLS/SEP and metadata is preserved"):
            packed = result.column("packed").to_pylist()[0]
            packed = np.asarray(packed, dtype=np.int16)
            assert packed.dtype == np.int16
            assert packed[0] == CLS
            assert SEP in packed.tolist()
            assert result.column("n_positions").to_pylist() == [2]
            assert result.column("ply").to_pylist() == [1]
            assert result.column("game_id").to_pylist() == ["deadbeefdeadbeef"]
            assert result.column("split").to_pylist()[0] in {"train", "test"}

    def test_finalize_batch_split_is_deterministic(self):
        with given("the same un-finalized batch twice"):
            batch = make_batch()

        with when("each copy is finalized with the same train ratio"):
            first = ChessPositionDataset._finalize_batch(batch, 0.8)
            second = ChessPositionDataset._finalize_batch(batch, 0.8)

        with then("both copies get the same split assignment"):
            assert (
                first.column("split").to_pylist() == second.column("split").to_pylist()
            )

    def test_get_start_batch_default(self, dataset):
        with when("the start batch is computed without resume"):
            start_batch = dataset._get_start_batch(resume=False)

        with then("generation starts at batch 1"):
            assert start_batch == 1

    def test_get_start_batch_resume_unknown(self, dataset):
        with given("a mocked HF client reporting batch 5 as next"):
            mock_client = MagicMock()
            mock_client.get_next_batch_number.return_value = 5
            dataset.hf_client = mock_client

        with when("the start batch is computed with resume"):
            start_batch = dataset._get_start_batch(resume=True)

        with then("generation resumes at batch 5"):
            assert start_batch == 5

    def test_create_dataset_card(self, dataset):
        with given("a mocked HF client returning a fixed dataset card"):
            mock_client = MagicMock()
            mock_client.create_dataset_card.return_value = "# Dataset Card"
            dataset.hf_client = mock_client

        with when("the dataset card is created"):
            card = dataset.create_dataset_card()

        with then("the card content is returned"):
            assert "Dataset Card" in card
