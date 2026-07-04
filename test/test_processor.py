import tempfile
from pathlib import Path

import chess
import chess.pgn
import pytest

from src.dataset.config import SamplingFilters
from src.dataset.pgn_processor import PGNProcessor
from src.dataset.types import GameRecord, GameMetadata, PositionRecord


SAMPLE_PGN = """[Event "Test Match"]
[Site "Test Site"]
[Date "2024.01.01"]
[Round "1"]
[White "Player1"]
[Black "Player2"]
[Result "1-0"]
[WhiteElo "2200"]
[BlackElo "2100"]
[Opening "Italian Game"]

1. e4 e5 2. Nf3 Nc6 3. Bc4 Bc5 4. c3 Nf6 5. d4 exd4 6. cxd4 Bb4+ 7. Bd2 Bxd2+ 8. Nbxd2 d5 9. exd5 Nxd5 10. Qb3 Nce7 11. O-O O-O 12. Rfe1 c6 1-0

[Event "Test Match 2"]
[Site "Test Site"]
[Date "2024.01.02"]
[Round "2"]
[White "Player3"]
[Black "Player4"]
[Result "0-1"]
[WhiteElo "1800"]
[BlackElo "1900"]
[Opening "Sicilian Defense"]

1. e4 c5 2. Nf3 d6 3. d4 cxd4 4. Nxd4 Nf6 5. Nc3 a6 6. Be3 e5 7. Nb3 Be6 8. f3 Be7 9. Qd2 O-O 10. O-O-O Nbd7 11. g4 b5 12. g5 b4 13. Ne2 Ne8 14. f4 a5 15. f5 a4 16. Nbd2 exf5 17. Nxf5 Nc5 18. Nd6 Nxd6 19. Qxd6 b3 20. cxb3 axb3 21. a3 Qa5 22. Kc2 Rfc8 0-1
"""


@pytest.fixture
def temp_pgn_file():
    with tempfile.NamedTemporaryFile(mode="w", suffix=".pgn", delete=False) as f:
        f.write(SAMPLE_PGN)
        yield f.name
    Path(f.name).unlink()


@pytest.fixture
def temp_pgn_directory(temp_pgn_file):
    dir_path = Path(temp_pgn_file).parent
    yield str(dir_path)


class TestPGNProcessor:
    def test_process_file_returns_game_records(self, temp_pgn_file):
        processor = PGNProcessor()
        games = list(processor.process_file(temp_pgn_file))
        assert len(games) >= 1
        assert all(isinstance(g, GameRecord) for g in games)

    def test_extract_game_returns_positions(self, temp_pgn_file):
        processor = PGNProcessor()
        games = list(processor.process_file(temp_pgn_file))
        assert all(len(g.positions) > 0 for g in games)

    def test_extract_game_metadata(self, temp_pgn_file):
        processor = PGNProcessor()
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
        filters = SamplingFilters(min_elo=1500, subsample_rate=1.0)
        processor = PGNProcessor(sampling_filters=filters)
        games = list(processor.process_file(temp_pgn_file))
        assert len(games) >= 1

    def test_position_records_have_ply(self, temp_pgn_file):
        processor = PGNProcessor()
        games = list(processor.process_file(temp_pgn_file))

        for game in games:
            for pos in game.positions:
                assert isinstance(pos.ply, int)
                assert pos.ply >= 0

    def test_position_records_have_board(self, temp_pgn_file):
        processor = PGNProcessor()
        games = list(processor.process_file(temp_pgn_file))

        for game in games:
            for pos in game.positions:
                assert isinstance(pos.board, chess.Board)

    def test_process_directory(self, temp_pgn_directory):
        processor = PGNProcessor()
        games = list(processor.process_directory(temp_pgn_directory))
        assert len(games) >= 1

    def test_temporal_window_extraction(self, temp_pgn_file):
        processor = PGNProcessor()
        with open(temp_pgn_file) as f:
            game = chess.pgn.read_game(f)

        windows = list(processor.extract_temporal_windows(game, window_size=5))
        assert all(len(w) == 5 for w in windows)

    def test_game_record_iteration(self, temp_pgn_file):
        processor = PGNProcessor()
        games = list(processor.process_file(temp_pgn_file))

        for game in games:
            positions = list(game)
            assert len(positions) == len(game.positions)
