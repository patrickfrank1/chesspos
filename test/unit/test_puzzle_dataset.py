from __future__ import annotations

import chess
import pyarrow as pa
import pytest

from bdd import given, then, when
from src.dataset.puzzle_dataset import PuzzleDataset, PuzzleDatasetConfig
from src.dataset.token_stream import TokenStreamEncoder, pack_stream

THEMES = ["fork", "mate", "pin"]

START_ROW = {
    "PuzzleId": "abc123",
    "FEN": chess.STARTING_FEN,
    "Moves": "e2e4 e7e5",
    "Rating": "1500",
    "Themes": "fork mate",
}

MIDGAME_FEN = "r6k/pp2r2p/4Rp1Q/3p4/8/1N1P2R1/PqP2bPP/7K b - - 0 24"
MIDGAME_MOVES = "f2g3 e6e7 b2b1 b3c1 b1c1 h6c1"


def boards_after(fen: str, moves: str) -> list[chess.Board]:
    board = chess.Board(fen)
    boards = []
    for uci in moves.split():
        board.push_uci(uci)
        boards.append(board.copy())
    return boards


def expected_packed(fen: str, moves: str) -> list[int]:
    encoder = TokenStreamEncoder()
    return pack_stream(
        [encoder.encode(board) for board in boards_after(fen, moves)]
    ).tolist()


def expected_previous(fen: str) -> list[int]:
    return TokenStreamEncoder().encode(chess.Board(fen)).tolist()


class TestParsePuzzle:
    def test_previous_position_and_every_move_are_encoded(self):
        with given("a puzzle starting from the initial position with two moves"):
            row = dict(START_ROW)

        with when("the puzzle row is parsed"):
            record = PuzzleDataset.parse_puzzle(row, THEMES)

        with then("the FEN board is stored separately from the move sequence"):
            assert record is not None
            assert record["puzzle_id"] == "abc123"
            assert record["previous_position"] == expected_previous(chess.STARTING_FEN)
            assert record["n_positions"] == 2
            assert record["packed"] == expected_packed(chess.STARTING_FEN, "e2e4 e7e5")
            assert record["rating"] == 1500
        return

    def test_previous_position_comes_from_the_fen_not_the_initial_position(self):
        with given("a puzzle starting from a black-to-move midgame FEN"):
            row = {
                "PuzzleId": "midgame",
                "FEN": MIDGAME_FEN,
                "Moves": MIDGAME_MOVES,
                "Rating": "1811",
                "Themes": "crushing middlegame",
            }

        with when("the puzzle row is parsed"):
            record = PuzzleDataset.parse_puzzle(row, ["crushing", "middlegame"])

        with then(
            "the previous_position is the FEN board and the stream is moves only"
        ):
            assert record is not None
            assert record["previous_position"] == expected_previous(MIDGAME_FEN)
            assert record["n_positions"] == 6
            assert record["packed"] == expected_packed(MIDGAME_FEN, MIDGAME_MOVES)
        return

    def test_puzzle_without_moves_is_skipped(self):
        with given("a puzzle row whose Moves field is empty"):
            row = dict(START_ROW, Moves="")

        with when("the puzzle row is parsed"):
            record = PuzzleDataset.parse_puzzle(row, THEMES)

        with then("a puzzle with no moves is malformed"):
            assert record is None
        return

    def test_themes_are_multi_hot_over_the_given_vocabulary(self):
        with given("a puzzle tagged fork and mate and a vocabulary of three themes"):
            row = dict(START_ROW, Themes="fork mate")

        with when("the puzzle row is parsed"):
            record = PuzzleDataset.parse_puzzle(row, THEMES)

        with then("each vocabulary theme gets a 0 or 1 column"):
            assert record is not None
            assert record["fork"] == 1
            assert record["mate"] == 1
            assert record["pin"] == 0
        return

    def test_theme_outside_the_vocabulary_is_ignored(self):
        with given("a puzzle carrying a theme missing from the vocabulary"):
            row = dict(START_ROW, Themes="fork brandNewTheme")

        with when("the puzzle row is parsed"):
            record = PuzzleDataset.parse_puzzle(row, THEMES)

        with then("only known themes are written"):
            assert record is not None
            assert record["fork"] == 1
            assert "brandNewTheme" not in record
        return

    def test_illegal_move_drops_the_puzzle(self):
        with given("a puzzle whose Moves field contains an illegal move"):
            row = dict(START_ROW, Moves="e2e4 e2e4")

        with when("the puzzle row is parsed"):
            record = PuzzleDataset.parse_puzzle(row, THEMES)

        with then("the malformed puzzle is skipped"):
            assert record is None
        return

    def test_invalid_fen_drops_the_puzzle(self):
        with given("a puzzle whose FEN is not a legal chess position"):
            row = dict(START_ROW, FEN="not-a-fen")

        with when("the puzzle row is parsed"):
            record = PuzzleDataset.parse_puzzle(row, THEMES)

        with then("the malformed puzzle is skipped"):
            assert record is None
        return

    def test_unparseable_rating_drops_the_puzzle(self):
        with given("a puzzle whose Rating is not an integer"):
            row = dict(START_ROW, Rating="unrated")

        with when("the puzzle row is parsed"):
            record = PuzzleDataset.parse_puzzle(row, THEMES)

        with then("the malformed puzzle is skipped"):
            assert record is None
        return


class TestDiscoverThemes:
    def test_returns_sorted_unique_themes_from_the_csv(self, tmp_path):
        with given("a CSV with overlapping theme tags"):
            csv_path = tmp_path / "puzzles.csv"
            csv_path.write_text("PuzzleId,Themes\na,fork mate\nb,pin fork\nc,\n")

        with when("the theme vocabulary is discovered"):
            themes = PuzzleDataset.discover_themes(str(csv_path))

        with then("themes are sorted and deduplicated"):
            assert themes == ["fork", "mate", "pin"]
        return


class TestToTable:
    def test_schema_has_typed_columns_and_all_theme_columns(self):
        with given("one parsed puzzle and a three-theme vocabulary"):
            record = PuzzleDataset.parse_puzzle(dict(START_ROW), THEMES)
            assert record is not None

        with when("the record is converted to an Arrow table"):
            table = PuzzleDataset.to_table([record], THEMES)

        with then("the schema uses compact types and named theme columns"):
            assert table.schema.names == [
                "puzzle_id",
                "previous_position",
                "packed",
                "n_positions",
                "rating",
                "fork",
                "mate",
                "pin",
            ]
            assert table.schema.field("previous_position").type == pa.list_(pa.int16())
            assert table.schema.field("packed").type == pa.list_(pa.int16())
            assert table.schema.field("n_positions").type == pa.int32()
            assert table.schema.field("rating").type == pa.int32()
            assert table.schema.field("fork").type == pa.int8()
            assert table.to_pylist() == [record]
        return

    def test_empty_input_keeps_the_full_schema(self):
        with given("no parsed puzzles and a three-theme vocabulary"):
            records: list[dict] = []

        with when("the records are converted to an Arrow table"):
            table = PuzzleDataset.to_table(records, THEMES)

        with then("the table is empty but still fully typed"):
            assert table.num_rows == 0
            assert table.schema.names == [
                "puzzle_id",
                "previous_position",
                "packed",
                "n_positions",
                "rating",
                "fork",
                "mate",
                "pin",
            ]
        return


class TestInitRay:
    def test_debug_mode_uses_a_fresh_local_single_cpu_cluster(self, monkeypatch):
        with given("a dataset config with debug mode enabled"):
            config = PuzzleDatasetConfig(input_path="puzzles.csv", debug=True)
            dataset = PuzzleDataset(config)
            calls: dict = {}

        with when("Ray is initialised"):
            import src.dataset.puzzle_dataset as module

            monkeypatch.setattr(
                module.ray, "init", lambda **kwargs: calls.update(kwargs)
            )
            dataset._init_ray()

        with then("a local cluster with one CPU runs without local_mode"):
            assert calls["address"] == "local"
            assert calls["num_cpus"] == 1
            assert "local_mode" not in calls
        return


class TestConfig:
    def test_rejects_non_positive_shard_size(self):
        with given("a puzzle dataset config with shard_size zero"):
            pass

        with when("the config is constructed"):
            with pytest.raises(ValueError):
                PuzzleDatasetConfig(input_path="puzzles.csv", shard_size=0)

        with then("validation rejects it"):
            pass
        return
