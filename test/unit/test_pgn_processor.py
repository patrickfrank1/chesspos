import random
import tempfile
from pathlib import Path
from unittest.mock import patch

import chess
import chess.pgn
import pytest

from bdd import given, then, when
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


def write_pgn(tmp_dir: str, name: str, content: bytes) -> str:
    path = Path(tmp_dir) / name
    path.write_bytes(content)
    return str(path)


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
        return

    def test_process_file_returns_game_records(self, temp_pgn_file):
        with given("a processor and a PGN file"):
            processor = PGNProcessor(keep_all())

        with when("the file is processed"):
            games = list(processor.process_file(temp_pgn_file))

        with then("at least one GameRecord is returned"):
            assert len(games) >= 1
            assert all(isinstance(g, GameRecord) for g in games)
        return

    def test_extract_game_returns_positions(self, temp_pgn_file):
        with given("a processor and a PGN file"):
            processor = PGNProcessor(keep_all())

        with when("the file is processed"):
            games = list(processor.process_file(temp_pgn_file))

        with then("every game carries at least one position"):
            assert all(len(g.positions) > 0 for g in games)
        return

    def test_extract_game_metadata(self, temp_pgn_file):
        with given("a processor and a PGN file"):
            processor = PGNProcessor(keep_all())

        with when("the file is processed"):
            games = list(processor.process_file(temp_pgn_file))

        with then("the first game carries the PGN header metadata"):
            first_game = games[0]
            assert first_game.metadata.white_elo == 2200
            assert first_game.metadata.black_elo == 2100
            assert first_game.metadata.result == "1-0"
            assert first_game.metadata.opening == "Italian Game"
        return

    def test_subsampling_drops_weak_games(self, temp_pgn_file):
        with given("a processor with a 2500+ Elo tier"):
            filters = GameSubsampling(tiers=[GameSubsampleTier(min_elo=2500, rate=1.0)])
            processor = PGNProcessor(subsampling=filters)

        with when("a 2200/2100 rated game is processed"):
            games = list(processor.process_file(temp_pgn_file))

        with then("the game is dropped"):
            assert len(games) == 0
        return

    def test_subsampling_allows_games(self, temp_pgn_file):
        with given("a processor with a keep-all tier"):
            processor = PGNProcessor(subsampling=keep_all())

        with when("a rated game is processed"):
            games = list(processor.process_file(temp_pgn_file))

        with then("the game is kept"):
            assert len(games) >= 1
        return

    def test_position_records_have_ply(self, temp_pgn_file):
        with given("a processor and a PGN file"):
            processor = PGNProcessor(keep_all())

        with when("the file is processed"):
            games = list(processor.process_file(temp_pgn_file))

        with then("every position records a non-negative integer ply"):
            for game in games:
                for pos in game.positions:
                    assert isinstance(pos.ply, int)
                    assert pos.ply >= 0
        return

    def test_position_records_have_board(self, temp_pgn_file):
        with given("a processor and a PGN file"):
            processor = PGNProcessor(keep_all())

        with when("the file is processed"):
            games = list(processor.process_file(temp_pgn_file))

        with then("every position records a chess board"):
            for game in games:
                for pos in game.positions:
                    assert isinstance(pos.board, chess.Board)
        return

    def test_process_directory(self, temp_pgn_directory):
        with given("a processor and a directory with one PGN file"):
            processor = PGNProcessor(keep_all())

        with when("the directory is processed"):
            games = list(processor.process_directory(temp_pgn_directory))

        with then("exactly one game is returned"):
            assert len(games) == 1
        return

    def test_temporal_window_extraction(self, temp_pgn_file):
        with given("a processor and a parsed PGN game"):
            processor = PGNProcessor()
            with open(temp_pgn_file) as f:
                game = chess.pgn.read_game(f)

        with when("temporal windows of size 5 are extracted"):
            windows = list(processor.extract_temporal_windows(game, window_size=5))

        with then("every window has exactly 5 positions"):
            assert game is not None
            assert all(len(w) == 5 for w in windows)
        return

    def test_game_record_iteration(self, temp_pgn_file):
        with given("games processed from a PGN file"):
            processor = PGNProcessor(keep_all())
            games = list(processor.process_file(temp_pgn_file))

        with when("each game is iterated over its positions"):
            iterated = [list(game) for game in games]

        with then("iteration yields the same positions as the record holds"):
            for game, positions in zip(games, iterated):
                assert len(positions) == len(game.positions)
        return


class TestGameSubsampling:
    def test_match_tier_strictest_qualifying_wins(self):
        with given("a default processor and games of various strengths"):
            processor = PGNProcessor()

        with when("tiers are matched for each game"):
            matched = [
                processor._match_tier(metadata).min_elo
                for metadata in [
                    GameMetadata(white_elo=2600, black_elo=2550),
                    GameMetadata(white_elo=2700, black_elo=2300),
                    GameMetadata(white_elo=1900, black_elo=1850),
                    GameMetadata(white_elo=1500, black_elo=1200),
                ]
            ]

        with then("the strictest qualifying tier wins each match"):
            assert matched == [2400, 2000, 0, 0]
        return

    def test_match_tier_missing_ratings_fall_to_catch_all(self):
        with given("a game with a missing white rating"):
            processor = PGNProcessor()
            metadata = GameMetadata(white_elo=None, black_elo=2600)

        with when("a tier is matched"):
            tier = processor._match_tier(metadata)

        with then("the catch-all tier applies"):
            assert tier is not None
            assert tier.min_elo == 0
        return

    def test_no_matching_tier_drops_game(self):
        with given("a processor with only a 2500+ Elo tier"):
            processor = PGNProcessor(
                GameSubsampling(tiers=[GameSubsampleTier(min_elo=2500, rate=1.0)])
            )

        with when("a 2000-rated game is checked"):
            keep = processor._keep_game(GameMetadata(white_elo=2000, black_elo=2000))

        with then("the game is dropped"):
            assert keep is False
        return

    def test_keep_game_respects_rate(self):
        with given("a processor with a keep-all tier at rate 0.5"):
            processor = PGNProcessor(
                GameSubsampling(tiers=[GameSubsampleTier(min_elo=0, rate=0.5)])
            )

        with when("the dice roll falls below and above the rate"):
            with patch("src.dataset.pgn_processor.random.random", return_value=0.4):
                kept = processor._keep_game(
                    GameMetadata(white_elo=2000, black_elo=2000)
                )
            with patch("src.dataset.pgn_processor.random.random", return_value=0.6):
                dropped = processor._keep_game(
                    GameMetadata(white_elo=2000, black_elo=2000)
                )

        with then("the game is kept below the rate and dropped above it"):
            assert kept
            assert not dropped
        return

    def test_all_positions_extracted_without_subsampling(self, temp_pgn_file):
        with given("a processor with a keep-all tier and a parsed game"):
            processor = PGNProcessor(keep_all())
            with open(temp_pgn_file) as f:
                game = chess.pgn.read_game(f)

        with when("the game is extracted"):
            record = processor.extract_game(game)

        with then("one position per mainline move is recorded"):
            assert record is not None
            assert len(record.positions) == len(list(game.mainline_moves()))
            assert len(record.positions) > 0
        return

    def test_tier_validation(self):
        with given("invalid tier parameters"):
            invalid = [dict(min_elo=-1), dict(rate=0), dict(rate=1.5)]

        with when("tiers are constructed from them"):
            caught = []
            for kwargs in invalid:
                with pytest.raises(ValueError) as excinfo:
                    GameSubsampleTier(**kwargs)
                caught.append(excinfo.value)

        with then("every construction raises a ValueError"):
            assert len(caught) == len(invalid)
            assert all(isinstance(e, ValueError) for e in caught)
        return

    def test_subsampling_round_trip(self):
        with given("a subsampling config"):
            subsampling = GameSubsampling(
                tiers=[GameSubsampleTier(min_elo=2400, rate=0.5)]
            )

        with when("the config is serialized and restored"):
            restored = GameSubsampling.from_dict(subsampling.to_dict())

        with then("the restored config equals the original"):
            assert restored == subsampling
        return


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
        with given("a processor and a time control string"):
            processor = PGNProcessor()

        with when("the time control is parsed"):
            parsed = processor._parse_time_control(tc)

        with then("the base seconds or None is returned"):
            assert parsed == expected
        return

    def test_bullet_game_dropped(self, temp_pgn_file):
        with given("a processor with a 300s minimum time control"):
            processor = PGNProcessor(keep_all())

        with when("a bullet game is processed"):
            bullet_pgn = SAMPLE_PGN_60.replace(
                b'[Result "1-0"]', b'[Result "1-0"]\n[TimeControl "180+0"]'
            )
            with tempfile.TemporaryDirectory() as tmp:
                path = write_pgn(tmp, "bullet.pgn", bullet_pgn)
                games = list(processor.process_file(path))

        with then("the game is dropped"):
            assert len(games) == 0
        return

    def test_rapid_game_kept(self, temp_pgn_file):
        with given("a processor with a 300s minimum time control"):
            processor = PGNProcessor(keep_all())

        with when("a rapid game is processed"):
            rapid_pgn = SAMPLE_PGN_60.replace(
                b'[Result "1-0"]', b'[Result "1-0"]\n[TimeControl "600+0"]'
            )
            with tempfile.TemporaryDirectory() as tmp:
                path = write_pgn(tmp, "rapid.pgn", rapid_pgn)
                games = list(processor.process_file(path))

        with then("the game is kept"):
            assert len(games) == 1
        return

    def test_five_minute_boundary_kept(self, temp_pgn_file):
        with given("a processor with a 300s minimum time control"):
            processor = PGNProcessor(keep_all())

        with when("a game of exactly 300s is processed"):
            blitz_pgn = SAMPLE_PGN_60.replace(
                b'[Result "1-0"]', b'[Result "1-0"]\n[TimeControl "300+0"]'
            )
            with tempfile.TemporaryDirectory() as tmp:
                path = write_pgn(tmp, "blitz.pgn", blitz_pgn)
                games = list(processor.process_file(path))

        with then("the game is kept at the boundary"):
            assert len(games) == 1
        return

    def test_unknown_time_control_kept(self, temp_pgn_file):
        with given("a processor with a 300s minimum time control"):
            processor = PGNProcessor(keep_all())

        with when("a game without a TimeControl header is processed"):
            games = list(processor.process_file(temp_pgn_file))

        with then("the game is kept"):
            assert len(games) == 1
        return

    def test_disabled_filter_keeps_bullet(self, temp_pgn_file):
        with given("a processor with the time control filter disabled"):
            processor = PGNProcessor(
                subsampling=keep_all(),
                time_control_filter=TimeControlFilter(min_seconds=None),
            )

        with when("a bullet game is processed"):
            bullet_pgn = SAMPLE_PGN_60.replace(
                b'[Result "1-0"]', b'[Result "1-0"]\n[TimeControl "60+0"]'
            )
            with tempfile.TemporaryDirectory() as tmp:
                path = write_pgn(tmp, "bullet.pgn", bullet_pgn)
                games = list(processor.process_file(path))

        with then("the game is kept"):
            assert len(games) == 1
        return

    def test_metadata_records_time_control(self, temp_pgn_file):
        with given("a processor with the time control filter disabled"):
            processor = PGNProcessor(
                subsampling=keep_all(),
                time_control_filter=TimeControlFilter(min_seconds=None),
            )

        with when("a game with a TimeControl header is processed"):
            tc_pgn = SAMPLE_PGN_60.replace(
                b'[Result "1-0"]', b'[Result "1-0"]\n[TimeControl "180+2"]'
            )
            with tempfile.TemporaryDirectory() as tmp:
                path = write_pgn(tmp, "tc.pgn", tc_pgn)
                games = list(processor.process_file(path))

        with then("the metadata records the time control"):
            assert games[0].metadata.time_control == "180+2"
        return

    def test_filter_validation(self):
        with given("an invalid minimum of -1 seconds"):
            min_seconds = -1

        with when("a time control filter is constructed"):
            with pytest.raises(ValueError) as excinfo:
                TimeControlFilter(min_seconds=min_seconds)

        with then("a ValueError is raised"):
            assert isinstance(excinfo.value, ValueError)
        return

    def test_filter_round_trip(self):
        with given("a time control filter"):
            f = TimeControlFilter(min_seconds=60)

        with when("the filter is serialized and restored"):
            restored = TimeControlFilter.from_dict(f.to_dict())

        with then("the restored filter equals the original"):
            assert restored == f
        return

    def test_preprocessing_config_round_trip(self):
        with given("a preprocessing config with a time control filter"):
            cfg = PreprocessingConfig(
                time_control_filter=TimeControlFilter(min_seconds=120)
            )

        with when("the config is serialized and restored"):
            restored = PreprocessingConfig.from_json(cfg.to_json())

        with then("the restored config keeps the time control filter"):
            assert restored.time_control_filter == cfg.time_control_filter
        return
