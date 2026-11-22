from __future__ import annotations

from pathlib import Path

import chess
import pyarrow as pa
import pyarrow.parquet as pq

from bdd import given, then, when
from src.chess_eval.puzzles import (
    background_boards,
    load_solved_puzzles,
    sacrificed_square,
)
from src.dataset.token_stream import TokenStreamEncoder, pack_stream


def queen_sac_boards() -> list[chess.Board]:
    """Black to move first, White gives up the queen on h5."""
    board = chess.Board("6k1/8/6p1/8/8/8/7P/3Q2K1 b - - 0 1")
    boards = [board.copy()]
    for uci in ("g8g7", "d1h5", "g6h5"):
        board.push_uci(uci)
        boards.append(board.copy())
    return boards


def quiet_boards() -> list[chess.Board]:
    board = chess.Board("6k1/8/6p1/8/8/8/7P/3Q2K1 b - - 0 1")
    boards = [board.copy()]
    for uci in ("g8g7", "d1d8", "g7f7"):
        board.push_uci(uci)
        boards.append(board.copy())
    return boards


def write_puzzle_shard(path: Path) -> None:
    encoder = TokenStreamEncoder()
    board = chess.Board()
    segments = [encoder.encode(board)]
    for uci in ("e2e4", "e7e5"):
        board.push_uci(uci)
        segments.append(encoder.encode(board))
    previous = segments[0].tolist()
    packed = pack_stream(segments[1:]).tolist()
    table = pa.table(
        {
            "puzzle_id": pa.array(["p1"], type=pa.string()),
            "previous_position": pa.array([previous], type=pa.list_(pa.int16())),
            "packed": pa.array([packed], type=pa.list_(pa.int16())),
            "n_positions": pa.array([2], type=pa.int32()),
            "rating": pa.array([1500], type=pa.int32()),
            "sacrifice": pa.array([1], type=pa.int8()),
        }
    )
    pq.write_table(table, path)
    return


class TestSacrificedSquare:
    def test_finds_the_queen_given_up_on_h5(self):
        with given("a solver queen move whose destination is recaptured"):
            boards = queen_sac_boards()

        with when("the sacrificed piece is located"):
            square = sacrificed_square(boards)

        with then("the queen's start square is returned"):
            assert square == chess.D1
        return

    def test_returns_none_without_an_immediate_recapture(self):
        with given("a quiet sequence with no recapture"):
            boards = quiet_boards()

        with when("the sacrificed piece is located"):
            square = sacrificed_square(boards)

        with then("there is no sacrifice"):
            assert square is None
        return

    def test_returns_none_for_a_sequence_that_is_too_short(self):
        with given("a single position"):
            boards = [chess.Board()]

        with when("the sacrificed piece is located"):
            square = sacrificed_square(boards)

        with then("the result is None"):
            assert square is None
        return


class TestPuzzleLoading:
    def test_groups_solution_positions_and_detects_no_sacrifice(self, tmp_path):
        with given("a tiny puzzle shard with a quiet solution"):
            write_puzzle_shard(tmp_path / "puzzles_0000.parquet")

        with when("the solved puzzles are loaded"):
            puzzles = load_solved_puzzles(tmp_path, max_puzzles=2, shard_limit=1)

        with then("one puzzle has all of its positions and no sacrifice"):
            assert len(puzzles) == 1
            puzzle = puzzles[0]
            assert puzzle.puzzle_id == "p1"
            assert len(puzzle.boards) == 3
            assert puzzle.sacrificed_square is None
            assert all(board.is_valid() for board in puzzle.boards)
        return

    def test_background_boards_use_distinct_puzzle_starts(self, tmp_path):
        with given("a tiny puzzle shard"):
            write_puzzle_shard(tmp_path / "puzzles_0000.parquet")

        with when("background boards are requested"):
            boards = background_boards(tmp_path, count=3, shard_limit=1)

        with then("the available start positions are returned"):
            assert len(boards) == 1
            assert boards[0].fen() == chess.Board().fen()
        return
