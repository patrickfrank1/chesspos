import random
import tempfile
from pathlib import Path

import chess
import chess.pgn
import pytest

from src.dataset.config import SamplingFilters
from src.dataset.pgn_processor import PGNProcessor
from src.dataset.types import GameRecord, GameMetadata, PositionRecord


SAMPLE_PGN_60 = b"""[Event "Test Match"]
[Site "Test Site"]
[Date "2024.01.01"]
[Round "1"]
[White "Player1"]
[Black "Player2"]
[Result "1-0"]
[WhiteElo "2200"]
[BlackElo "2100"]
[Opening "Italian Game"]

1. e4 e5 2. Nf3 Nc6 3. Bc4 Bc5 4. c3 Nf6 5. d4 exd4 6. cxd4 Bb4+ 7. Bd2 Bxd2+ 8. Nbxd2 d5 9. exd5 Nxd5 10. Qb3 Nce7 11. O-O O-O 12. Rfe1 c6 13. Rad1 Qc7 14. Nc4 b5 15. Nce5 Nd7 16. Nxd7 Bxd7 17. Ne5 Be8 18. Qg3 Kh8 19. Bxd5 cxd5 20. f4 f6 21. Nf3 Rac8 22. f5 Bd7 23. Qf4 Rfe8 24. h4 Qb6 25. Kh2 Bc6 26. Re3 Qb8 27. Rde1 Qd6 28. Ng5 Rc7 29. Nh3 Rf7 30. Rg3 a5 31. Qg4 Rff8 32. Ng5 a4 33. Ne6 Re7 34. Qh5 g6 35. fxg6 hxg6 36. Qh6 Qxd4 37. Rg4 Qe5 38. Ng5 Qf5 39. Re6 Bb7 40. Qg7# 1-0
"""


@pytest.fixture
def temp_pgn_file(tmp_path):
    pgn_path = tmp_path / "test.pgn"
    pgn_path.write_bytes(SAMPLE_PGN_60)
    return str(pgn_path)


@pytest.fixture
def temp_pgn_directory(tmp_path):
    pgn_path = tmp_path / "game.pgn"
    pgn_path.write_bytes(SAMPLE_PGN_60)
    return str(tmp_path)


class TestPGNProcessor:
    @pytest.fixture(autouse=True)
    def _seed_random(self):
        random.seed(42)

    def test_process_file_returns_game_records(self, temp_pgn_file):
        processor = PGNProcessor(SamplingFilters(min_elo=0, subsample_rate=1.0))
        games = list(processor.process_file(temp_pgn_file))
        assert len(games) >= 1
        assert all(isinstance(g, GameRecord) for g in games)

    def test_extract_game_returns_positions(self, temp_pgn_file):
        processor = PGNProcessor(SamplingFilters(min_elo=0, subsample_rate=1.0))
        games = list(processor.process_file(temp_pgn_file))
        assert all(len(g.positions) > 0 for g in games)

    def test_extract_game_metadata(self, temp_pgn_file):
        processor = PGNProcessor(SamplingFilters(min_elo=0, subsample_rate=1.0))
        games = list(processor.process_file(temp_pgn_file))

        first_game = games[0]
        assert first_game.metadata.white_elo == 2200
        assert first_game.metadata.black_elo == 2100
        assert first_game.metadata.result == "1-0"
        assert first_game.metadata.opening == "Italian Game"

    def test_sampling_filters_by_elo(self, temp_pgn_file):
        filters = SamplingFilters(min_elo=2500, subsample_rate=1.0)
        processor = PGNProcessor(sampling_filters=filters)
        games = list(processor.process_file(temp_pgn_file))
        assert len(games) == 0

    def test_sampling_filters_allows_games(self, temp_pgn_file):
        filters = SamplingFilters(min_elo=0, subsample_rate=1.0)
        processor = PGNProcessor(sampling_filters=filters)
        games = list(processor.process_file(temp_pgn_file))
        assert len(games) >= 1

    def test_position_records_have_ply(self, temp_pgn_file):
        processor = PGNProcessor(SamplingFilters(min_elo=0, subsample_rate=1.0))
        games = list(processor.process_file(temp_pgn_file))

        for game in games:
            for pos in game.positions:
                assert isinstance(pos.ply, int)
                assert pos.ply >= 0

    def test_position_records_have_board(self, temp_pgn_file):
        processor = PGNProcessor(SamplingFilters(min_elo=0, subsample_rate=1.0))
        games = list(processor.process_file(temp_pgn_file))

        for game in games:
            for pos in game.positions:
                assert isinstance(pos.board, chess.Board)

    def test_process_directory(self, temp_pgn_directory):
        processor = PGNProcessor(SamplingFilters(min_elo=0, subsample_rate=1.0))
        games = list(processor.process_directory(temp_pgn_directory))
        assert len(games) == 1

    def test_temporal_window_extraction(self, temp_pgn_file):
        processor = PGNProcessor()
        with open(temp_pgn_file) as f:
            game = chess.pgn.read_game(f)

        assert game is not None
        windows = list(processor.extract_temporal_windows(game, window_size=5))
        assert all(len(w) == 5 for w in windows)

    def test_game_record_iteration(self, temp_pgn_file):
        processor = PGNProcessor(SamplingFilters(min_elo=0, subsample_rate=1.0))
        games = list(processor.process_file(temp_pgn_file))

        for game in games:
            positions = list(game)
            assert len(positions) == len(game.positions)
