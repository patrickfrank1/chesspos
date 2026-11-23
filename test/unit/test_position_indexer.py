import datetime as dt
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest


from bdd import given, then, when
from src.dataset.puzzle_variants import pack_segments
from src.search.index import GameRow, build_position_index, index_game_row
from src.search.memory_store import MemoryPositionStore
from src.search.partitioned_store import PartitionedPositionStore
from position_support import FakeEmbedder, game_segments


def _write_games(path: Path, rows: list[dict]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    table = pa.table(
        {
            "packed": pa.array(
                [row["packed"] for row in rows], type=pa.list_(pa.int16())
            ),
            "n_positions": pa.array(
                [row["n_positions"] for row in rows], type=pa.int32()
            ),
            "game_id": pa.array([row["game_id"] for row in rows], pa.string()),
            "white_elo": pa.array([row["white_elo"] for row in rows], pa.int32()),
            "black_elo": pa.array([row["black_elo"] for row in rows], pa.int32()),
            "result": pa.array([row["result"] for row in rows], pa.string()),
            **(
                {"utc_date": pa.array([row["utc_date"] for row in rows], pa.string())}
                if "utc_date" in rows[0]
                else {}
            ),
        }
    )
    pq.write_table(table, path)
    return path


def _packed_games(count: int = 2) -> list[dict]:
    packed, _ = pack_segments(game_segments())
    return [
        {
            "packed": packed.tolist(),
            "n_positions": len(game_segments()),
            "game_id": f"{index:016x}",
            "white_elo": 2500 + index,
            "black_elo": 2400 + index,
            "result": "1-0" if index % 2 == 0 else "0-1",
        }
        for index in range(count)
    ]


def test_index_game_row_expands_each_position():
    with given("a game containing three positions"):
        packed, _ = pack_segments(game_segments())
        game = GameRow(
            packed=packed,
            game_id="0123456789abcdef",
            white_elo=2500,
            black_elo=2400,
            result="1-0",
            split="train",
            utc_date=dt.date(2024, 5, 1),
            year=2024,
        )

    with when("the game row is expanded with a deterministic embedder"):
        rows = index_game_row(game, FakeEmbedder(dim=8))

    with then("one row per ply is produced with metadata and unit vectors"):
        assert len(rows) == 3
        assert [row.ply for row in rows] == [1, 2, 3]
        assert all(row.game_id == 0x0123456789ABCDEF for row in rows)
        assert all(row.utc_date == dt.date(2024, 5, 1) for row in rows)
        assert all(row.year == 2024 for row in rows)
        assert all(row.result == "1-0" for row in rows)
        assert all(row.split == "train" for row in rows)
        assert all(row.n_pieces == 32 for row in rows)
        vectors = np.stack([row.vector for row in rows])
        assert vectors.shape == (3, 8)
        np.testing.assert_allclose(
            np.linalg.norm(vectors, axis=1), np.ones(3), atol=1e-5
        )
    return


def test_index_game_row_rejects_an_incompatible_vocabulary():
    with given("a shard whose tokens exceed the checkpoint vocabulary"):
        packed, _ = pack_segments(game_segments())
        game = GameRow(
            packed=packed,
            game_id="0123456789abcdef",
            white_elo=2500,
            black_elo=2400,
            result="1-0",
            split="train",
        )

    with when("the game is indexed with a 12-token embedder"):
        with pytest.raises(ValueError, match="different tokenizer"):
            index_game_row(game, FakeEmbedder(dim=8, vocab_size=12))

    with then("no rows are produced"):
        pass
    return


def test_build_position_index_from_parquet(tmp_path):
    with given("a parquet shard with two games and no date or split columns"):
        path = tmp_path / "lichess_elite_2024-05" / "train" / "batch_0000.parquet"
        _write_games(path, _packed_games(2))
        store = MemoryPositionStore(dim=8)

    with when("the index is built from the directory"):
        stats = build_position_index(
            store=store,
            embedder=FakeEmbedder(dim=8),
            data_dir=tmp_path,
        )

    with then("every position is indexed with path-derived split and year"):
        assert stats.games == 2
        assert stats.positions == 6
        assert store.count() == 6
        assert all(row.split == "train" for row in store.rows)
        assert all(row.year == 2024 for row in store.rows)
        assert all(row.utc_date == dt.date(1970, 1, 1) for row in store.rows)
    return


def test_build_position_index_reads_utc_date_column(tmp_path):
    with given("a shard that carries per-game ISO dates"):
        rows = _packed_games(2)
        rows[0]["utc_date"] = "2023-01-15"
        rows[1]["utc_date"] = ""
        path = tmp_path / "lichess_elite_2021-07" / "test" / "batch_0000.parquet"
        _write_games(path, rows)
        store = MemoryPositionStore(dim=8)

    with when("the index is built"):
        stats = build_position_index(
            store=store,
            embedder=FakeEmbedder(dim=8),
            data_dir=tmp_path,
        )

    with then("known dates are parsed and missing dates fall back to the path year"):
        first = [row for row in store.rows if row.game_id == 0]
        second = [row for row in store.rows if row.game_id == 1]
        assert {row.utc_date for row in first} == {dt.date(2023, 1, 15)}
        assert {row.year for row in first} == {2023}
        assert {row.utc_date for row in second} == {dt.date(1970, 1, 1)}
        assert {row.year for row in second} == {2021}
        assert stats.games == 2
    return


def test_build_position_index_respects_limit_games(tmp_path):
    with given("a shard with three games and a one-game limit"):
        path = tmp_path / "lichess_elite_2022-03" / "train" / "batch_0000.parquet"
        _write_games(path, _packed_games(3))
        store = MemoryPositionStore(dim=8)

    with when("the index is built with limit_games=1"):
        stats = build_position_index(
            store=store,
            embedder=FakeEmbedder(dim=8),
            data_dir=tmp_path,
            limit_games=1,
        )

    with then("only the first game is indexed"):
        assert stats.games == 1
        assert stats.positions == 3
        assert store.count() == 3
    return


def test_empty_game_stream_is_skipped(tmp_path):
    with given("a shard containing an empty game and a real game"):
        rows = [
            {
                "packed": [3],
                "n_positions": 0,
                "game_id": "0000000000000000",
                "white_elo": 0,
                "black_elo": 0,
                "result": "*",
            },
            *_packed_games(1),
        ]
        path = tmp_path / "lichess_elite_2023-01" / "train" / "batch_0000.parquet"
        _write_games(path, rows)
        store = MemoryPositionStore(dim=8)

    with when("the index is built"):
        stats = build_position_index(
            store=store,
            embedder=FakeEmbedder(dim=8),
            data_dir=tmp_path,
        )

    with then("the empty game is skipped without failing the build"):
        assert stats.games == 1
        assert stats.skipped == 1
        assert store.count() == 3
    return


def test_build_position_index_partitioned_by_year(tmp_path):
    with given("shards from 2024 and 2025"):
        _write_games(
            tmp_path / "lichess_elite_2024-05" / "train" / "batch_0000.parquet",
            _packed_games(2),
        )
        _write_games(
            tmp_path / "lichess_elite_2025-01" / "train" / "batch_0000.parquet",
            _packed_games(2),
        )
        store = PartitionedPositionStore(
            tmp_path / "index", dim=8, create=True, overwrite=True
        )

    with when("the index is built partitioned by year"):
        stats = build_position_index(
            store=store,
            embedder=FakeEmbedder(dim=8),
            data_dir=tmp_path,
            partition_by="year",
        )

    with then("each year has its own partition and position count"):
        assert set(store.partitions) == {"year=2024", "year=2025"}
        assert stats.partitions == {"year=2024": 6, "year=2025": 6}
        assert store.count() == 12
    return


def test_build_position_index_partitioned_by_source(tmp_path):
    with given("shards from two archive sources"):
        _write_games(
            tmp_path / "lichess_elite_2024-05" / "train" / "batch_0000.parquet",
            _packed_games(2),
        )
        _write_games(
            tmp_path / "lichess_elite_2025-01" / "train" / "batch_0000.parquet",
            _packed_games(2),
        )
        store = PartitionedPositionStore(
            tmp_path / "index", dim=8, create=True, overwrite=True
        )

    with when("the index is built partitioned by source"):
        stats = build_position_index(
            store=store,
            embedder=FakeEmbedder(dim=8),
            data_dir=tmp_path,
            partition_by="source",
        )

    with then("the source names become partition labels"):
        assert set(store.partitions) == {
            "source=lichess_elite_2024-05",
            "source=lichess_elite_2025-01",
        }
        assert stats.partitions["source=lichess_elite_2024-05"] == 6
        assert store.count() == 12
    return


def test_build_position_index_appends_to_an_existing_partition(tmp_path):
    with given("an index already built from one 2024 shard"):
        _write_games(
            tmp_path
            / "first"
            / "lichess_elite_2024-05"
            / "train"
            / "batch_0000.parquet",
            _packed_games(2),
        )
        store = PartitionedPositionStore(
            tmp_path / "index", dim=8, create=True, overwrite=True
        )
        build_position_index(
            store=store,
            embedder=FakeEmbedder(dim=8),
            data_dir=tmp_path / "first",
            partition_by="year",
        )

    with when("a second 2024 shard is appended"):
        _write_games(
            tmp_path
            / "second"
            / "lichess_elite_2024-06"
            / "train"
            / "batch_0000.parquet",
            _packed_games(2),
        )
        stats = build_position_index(
            store=store,
            embedder=FakeEmbedder(dim=8),
            data_dir=tmp_path / "second",
            partition_by="year",
        )

    with then("the same year partition grows instead of being replaced"):
        assert stats.partitions == {"year=2024": 6}
        assert store.partitions["year=2024"]["rows"] == 12
        assert store.count() == 12
    return
