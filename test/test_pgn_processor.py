import random
import tempfile
from pathlib import Path
from unittest.mock import patch

import chess
import chess.pgn
import pytest

from src.dataset.config import (
    GameSubsampling,
    GameSubsampleTier,
    PreprocessingConfig,
    TimeControlFilter,
)
from src.dataset.pgn_processor import PGNProcessor
from src.dataset.types import GameRecord, GameMetadata


def keep_all() -> GameSubsampling:
    return GameSubsampling(tiers=[GameSubsampleTier(min_elo=0, rate=1.0)])


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
        processor = PGNProcessor(keep_all())
        games = list(processor.process_file(temp_pgn_file))
        assert len(games) >= 1
        assert all(isinstance(g, GameRecord) for g in games)

    def test_extract_game_returns_positions(self, temp_pgn_file):
        processor = PGNProcessor(keep_all())
        games = list(processor.process_file(temp_pgn_file))
        assert all(len(g.positions) > 0 for g in games)

    def test_extract_game_metadata(self, temp_pgn_file):
        processor = PGNProcessor(keep_all())
        games = list(processor.process_file(temp_pgn_file))

        first_game = games[0]
        assert first_game.metadata.white_elo == 2200
        assert first_game.metadata.black_elo == 2100
        assert first_game.metadata.result == "1-0"
        assert first_game.metadata.opening == "Italian Game"

    def test_subsampling_drops_weak_games(self, temp_pgn_file):
        filters = GameSubsampling(tiers=[GameSubsampleTier(min_elo=2500, rate=1.0)])
        processor = PGNProcessor(subsampling=filters)
        games = list(processor.process_file(temp_pgn_file))
        assert len(games) == 0

    def test_subsampling_allows_games(self, temp_pgn_file):
        filters = keep_all()
        processor = PGNProcessor(subsampling=filters)
        games = list(processor.process_file(temp_pgn_file))
        assert len(games) >= 1

    def test_position_records_have_ply(self, temp_pgn_file):
        processor = PGNProcessor(keep_all())
        games = list(processor.process_file(temp_pgn_file))

        for game in games:
            for pos in game.positions:
                assert isinstance(pos.ply, int)
                assert pos.ply >= 0

    def test_position_records_have_board(self, temp_pgn_file):
        processor = PGNProcessor(keep_all())
        games = list(processor.process_file(temp_pgn_file))

        for game in games:
            for pos in game.positions:
                assert isinstance(pos.board, chess.Board)

    def test_process_directory(self, temp_pgn_directory):
        processor = PGNProcessor(keep_all())
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
        processor = PGNProcessor(keep_all())
        games = list(processor.process_file(temp_pgn_file))

        for game in games:
            positions = list(game)
            assert len(positions) == len(game.positions)


class TestGameSubsampling:
    def test_match_tier_strictest_qualifying_wins(self):
        processor = PGNProcessor()
        assert (
            processor._match_tier(GameMetadata(white_elo=2600, black_elo=2550)).min_elo
            == 2400
        )
        assert (
            processor._match_tier(GameMetadata(white_elo=2700, black_elo=2300)).min_elo
            == 2000
        )
        assert (
            processor._match_tier(GameMetadata(white_elo=1900, black_elo=1850)).min_elo
            == 0
        )
        assert (
            processor._match_tier(GameMetadata(white_elo=1500, black_elo=1200)).min_elo
            == 0
        )

    def test_match_tier_missing_ratings_fall_to_catch_all(self):
        processor = PGNProcessor()
        tier = processor._match_tier(GameMetadata(white_elo=None, black_elo=2600))
        assert tier is not None
        assert tier.min_elo == 0

    def test_no_matching_tier_drops_game(self):
        processor = PGNProcessor(
            GameSubsampling(tiers=[GameSubsampleTier(min_elo=2500, rate=1.0)])
        )
        assert (
            processor._keep_game(GameMetadata(white_elo=2000, black_elo=2000)) is False
        )

    def test_keep_game_respects_rate(self):
        processor = PGNProcessor(
            GameSubsampling(tiers=[GameSubsampleTier(min_elo=0, rate=0.5)])
        )
        with patch("src.dataset.pgn_processor.random.random", return_value=0.4):
            assert processor._keep_game(GameMetadata(white_elo=2000, black_elo=2000))
        with patch("src.dataset.pgn_processor.random.random", return_value=0.6):
            assert not processor._keep_game(
                GameMetadata(white_elo=2000, black_elo=2000)
            )

    def test_all_positions_extracted_without_subsampling(self, temp_pgn_file):
        processor = PGNProcessor(keep_all())
        with open(temp_pgn_file) as f:
            game = chess.pgn.read_game(f)
        record = processor.extract_game(game)
        assert record is not None
        assert len(record.positions) == len(list(game.mainline_moves()))
        assert len(record.positions) > 0

    def test_tier_validation(self):
        with pytest.raises(ValueError):
            GameSubsampleTier(min_elo=-1)
        with pytest.raises(ValueError):
            GameSubsampleTier(rate=0)
        with pytest.raises(ValueError):
            GameSubsampleTier(rate=1.5)

    def test_subsampling_round_trip(self):
        subsampling = GameSubsampling(tiers=[GameSubsampleTier(min_elo=2400, rate=0.5)])
        restored = GameSubsampling.from_dict(subsampling.to_dict())
        assert restored == subsampling


class TestTimeControlFilter:
    @pytest.mark.parametrize(
        ("tc", "expected"),
        [
            ("300", 300),
            ("180+2", 180),
            ("40/900", 900),
            ("600+5", 600),
            ("-", None),
            ("?", None),
            ("*", None),
            (None, None),
            ("garbage", None),
        ],
    )
    def test_parse_time_control(self, tc, expected):
        processor = PGNProcessor()
        assert processor._parse_time_control(tc) == expected

    def test_bullet_game_dropped(self, temp_pgn_file):
        bullet_pgn = SAMPLE_PGN_60.replace(
            b'[Result "1-0"]', b'[Result "1-0"]\n[TimeControl "180+0"]'
        )
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "bullet.pgn"
            path.write_bytes(bullet_pgn)
            processor = PGNProcessor(keep_all())
            games = list(processor.process_file(str(path)))
        assert len(games) == 0

    def test_rapid_game_kept(self, temp_pgn_file):
        rapid_pgn = SAMPLE_PGN_60.replace(
            b'[Result "1-0"]', b'[Result "1-0"]\n[TimeControl "600+0"]'
        )
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "rapid.pgn"
            path.write_bytes(rapid_pgn)
            processor = PGNProcessor(keep_all())
            games = list(processor.process_file(str(path)))
        assert len(games) == 1

    def test_five_minute_boundary_kept(self, temp_pgn_file):
        blitz_pgn = SAMPLE_PGN_60.replace(
            b'[Result "1-0"]', b'[Result "1-0"]\n[TimeControl "300+0"]'
        )
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "blitz.pgn"
            path.write_bytes(blitz_pgn)
            processor = PGNProcessor(keep_all())
            games = list(processor.process_file(str(path)))
        assert len(games) == 1

    def test_unknown_time_control_kept(self, temp_pgn_file):
        processor = PGNProcessor(keep_all())
        games = list(processor.process_file(temp_pgn_file))
        assert len(games) == 1

    def test_disabled_filter_keeps_bullet(self, temp_pgn_file):
        bullet_pgn = SAMPLE_PGN_60.replace(
            b'[Result "1-0"]', b'[Result "1-0"]\n[TimeControl "60+0"]'
        )
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "bullet.pgn"
            path.write_bytes(bullet_pgn)
            processor = PGNProcessor(
                subsampling=keep_all(),
                time_control_filter=TimeControlFilter(min_seconds=None),
            )
            games = list(processor.process_file(str(path)))
        assert len(games) == 1

    def test_metadata_records_time_control(self, temp_pgn_file):
        tc_pgn = SAMPLE_PGN_60.replace(
            b'[Result "1-0"]', b'[Result "1-0"]\n[TimeControl "180+2"]'
        )
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "tc.pgn"
            path.write_bytes(tc_pgn)
            processor = PGNProcessor(
                subsampling=keep_all(),
                time_control_filter=TimeControlFilter(min_seconds=None),
            )
            games = list(processor.process_file(str(path)))
        assert games[0].metadata.time_control == "180+2"

    def test_filter_validation(self):
        with pytest.raises(ValueError):
            TimeControlFilter(min_seconds=-1)

    def test_filter_round_trip(self):
        f = TimeControlFilter(min_seconds=60)
        assert TimeControlFilter.from_dict(f.to_dict()) == f

    def test_preprocessing_config_round_trip(self):
        cfg = PreprocessingConfig(
            time_control_filter=TimeControlFilter(min_seconds=120)
        )
        restored = PreprocessingConfig.from_json(cfg.to_json())
        assert restored.time_control_filter == cfg.time_control_filter
