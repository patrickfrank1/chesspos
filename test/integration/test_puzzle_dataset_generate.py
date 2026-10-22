from __future__ import annotations

import csv
import os
from pathlib import Path
from unittest.mock import patch

import chess
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import ray

from bdd import given, then, when
from src.dataset.puzzle_dataset import PuzzleDataset, PuzzleDatasetConfig
from src.dataset.token_stream import TokenStreamEncoder, SEGMENT_LEN


@pytest.fixture(scope="module")
def local_ray():
    with patch.dict(os.environ, {"RAY_ADDRESS": "local"}):
        ray.init(ignore_reinit_error=True, include_dashboard=False)
        try:
            with patch.object(ray, "shutdown"):
                yield
        finally:
            ray.shutdown()
    return


def write_puzzle_csv(path: Path) -> None:
    rows = [
        {
            "PuzzleId": "aaaa",
            "FEN": chess.STARTING_FEN,
            "Moves": "e2e4 e7e5",
            "Rating": "1500",
            "Themes": "fork mate",
        },
        {
            "PuzzleId": "bbbb",
            "FEN": "r6k/pp2r2p/4Rp1Q/3p4/8/1N1P2R1/PqP2bPP/7K b - - 0 24",
            "Moves": "f2g3 e6e7",
            "Rating": "1811",
            "Themes": "crushing hangingPiece",
        },
        {
            "PuzzleId": "cccc",
            "FEN": chess.STARTING_FEN,
            "Moves": "e2e5",
            "Rating": "1200",
            "Themes": "pin",
        },
    ]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["PuzzleId", "FEN", "Moves", "Rating", "Themes"],
        )
        writer.writeheader()
        writer.writerows(rows)
    return


def decode_positions(packed: list[int], n_positions: int) -> list[chess.Board]:
    encoder = TokenStreamEncoder()
    boards = []
    for index in range(n_positions):
        start = 1 + index * (SEGMENT_LEN + 1)
        segment = packed[start : start + SEGMENT_LEN]
        boards.append(encoder.decode(segment))
    return boards


def position_key(board: chess.Board) -> tuple:
    return (
        board.board_fen(),
        board.turn,
        board.castling_xfen(),
        board.ep_square if board.has_legal_en_passant() else None,
    )


class TestGenerate:
    def test_writes_sharded_parquet_with_encoded_sequences_and_labels(
        self, tmp_path, local_ray
    ):
        with given("a CSV with two valid puzzles and one illegal puzzle"):
            csv_path = tmp_path / "puzzles.csv"
            write_puzzle_csv(csv_path)
            output_dir = tmp_path / "out"
            config = PuzzleDatasetConfig(
                input_path=str(csv_path),
                output_dir=str(output_dir),
                shard_size=1,
                worker_count=1,
                memory_limit_mb=512,
            )

        with when("the dataset is generated with Ray"):
            written_dir = PuzzleDataset(config).generate()

        with then("parquet shards hold only the valid puzzles with full labels"):
            files = sorted(Path(written_dir).glob("*.parquet"))
            assert files
            table = pa.concat_tables([pq.read_table(path) for path in files])
            assert table.num_rows == 2

            records = {row["puzzle_id"]: row for row in table.to_pylist()}
            assert set(records) == {"aaaa", "bbbb"}

            encoder = TokenStreamEncoder()

            first = records["aaaa"]
            assert first["rating"] == 1500
            assert first["n_positions"] == 2
            assert first["fork"] == 1
            assert first["mate"] == 1
            assert first["pin"] == 0
            assert first["crushing"] == 0

            previous = encoder.decode(first["previous_position"])
            assert position_key(previous) == position_key(chess.Board())

            expected_boards = []
            board = chess.Board()
            for uci in ("e2e4", "e7e5"):
                board.push_uci(uci)
                expected_boards.append(board.copy())
            decoded_boards = decode_positions(first["packed"], first["n_positions"])
            assert [position_key(board) for board in decoded_boards] == [
                position_key(board) for board in expected_boards
            ]

            second = records["bbbb"]
            assert second["rating"] == 1811
            assert second["n_positions"] == 2
            assert second["crushing"] == 1
            assert second["hangingPiece"] == 1
            assert second["fork"] == 0
            assert "cccc" not in records
        return

    def test_theme_vocabulary_is_discovered_from_the_input(self, tmp_path, local_ray):
        with given("a CSV whose themes are not known up front"):
            csv_path = tmp_path / "puzzles.csv"
            write_puzzle_csv(csv_path)
            output_dir = tmp_path / "out"
            config = PuzzleDatasetConfig(
                input_path=str(csv_path),
                output_dir=str(output_dir),
                shard_size=10,
                worker_count=1,
                memory_limit_mb=512,
            )

        with when("the dataset is generated"):
            written_dir = PuzzleDataset(config).generate()

        with then("the output has one int8 column per theme found in the CSV"):
            table = pq.read_table(next(Path(written_dir).glob("*.parquet")))
            assert set(table.schema.names) >= {
                "crushing",
                "fork",
                "hangingPiece",
                "mate",
                "pin",
            }
            assert table.schema.field("crushing").type == pa.int8()
        return


pytestmark = pytest.mark.integration
