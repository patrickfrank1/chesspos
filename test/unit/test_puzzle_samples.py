from __future__ import annotations

from pathlib import Path

import chess
import pyarrow as pa
import pyarrow.parquet as pq
from fastapi.testclient import TestClient

from bdd import given, then, when
from src.app.chess_eval.app import create_app
from src.app.chess_eval.samples import load_puzzle_samples
from src.dataset.token_stream import TokenStreamEncoder, pack_stream


def write_puzzle_shard(path: Path) -> None:
    encoder = TokenStreamEncoder()
    start = encoder.encode(chess.Board()).tolist()
    board = chess.Board()
    board.push_uci("e2e4")
    after_e4 = encoder.encode(board)
    board.push_uci("e7e5")
    after_e5 = encoder.encode(board)
    packed = pack_stream([after_e4, after_e5]).tolist()
    table = pa.table(
        {
            "puzzle_id": pa.array(["p1", "p2"], type=pa.string()),
            "previous_position": pa.array([start, start], type=pa.list_(pa.int16())),
            "packed": pa.array([packed, packed], type=pa.list_(pa.int16())),
            "n_positions": pa.array([2, 2], type=pa.int32()),
            "rating": pa.array([1500, 1600], type=pa.int32()),
            "sacrifice": pa.array([1, 0], type=pa.int8()),
        }
    )
    pq.write_table(table, path)
    return


class TestPuzzleSamples:
    def test_loads_start_and_solution_positions(self, tmp_path):
        with given("a tiny puzzle shard with one positive and one negative"):
            path = tmp_path / "puzzles_0000.parquet"
            write_puzzle_shard(path)

        with when("the puzzle samples are loaded"):
            samples = load_puzzle_samples(
                tmp_path, max_puzzles=2, seed=1, shard_limit=1
            )

        with then("both puzzles expose their start and solution positions"):
            assert len(samples) == 6
            labels = [sample.label for sample in samples]
            assert any("p1" in label and "start" in label for label in labels)
            assert any("p1" in label and "move 1/2" in label for label in labels)
            assert any("p2" in label and "move 2/2" in label for label in labels)
            assert {sample.sacrifice for sample in samples} == {0, 1}
            for sample in samples:
                assert chess.Board(sample.fen).is_valid()
            assert samples[0].sacrifice == 1
        return

    def test_returns_empty_when_the_shard_directory_is_missing(self, tmp_path):
        with given("a missing puzzle directory"):
            missing = tmp_path / "missing"

        with when("samples are requested"):
            samples = load_puzzle_samples(missing, max_puzzles=2, seed=1, shard_limit=1)

        with then("the result is empty"):
            assert samples == []
        return

    def test_api_serves_puzzle_samples(self, tmp_path):
        with given("an app with a tiny puzzle shard"):
            write_puzzle_shard(tmp_path / "puzzles_0000.parquet")
            app = create_app(
                model_roots=[tmp_path / "models"],
                device="cpu",
                puzzle_data_dir=tmp_path,
            )
            client = TestClient(app)

        with when("the samples endpoint is requested"):
            response = client.get("/api/samples")

        with then("the sample positions are returned as JSON"):
            assert response.status_code == 200
            payload = response.json()
            assert len(payload["samples"]) == 6
            assert payload["samples"][0]["fen"]
            assert payload["samples"][0]["label"].startswith("p")
        return
