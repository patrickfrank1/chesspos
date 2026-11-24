import datetime as dt

import numpy as np
import pytest

from bdd import given, then, when
from src.search.lancedb_store import LancePositionStore
from src.search.memory_store import MemoryPositionStore
from src.search.store import PositionFilter
from position_support import make_position_row


def _rows() -> list:
    rows = []
    for index in range(12):
        game = index // 3
        vector = np.zeros(8, dtype=np.float32)
        vector[index % 8] = 1.0
        rows.append(
            make_position_row(
                vector=vector,
                game_id=game,
                ply=index % 3 + 1,
                white_elo=2400 if index % 2 else 2600,
                black_elo=2400 if index % 2 else 2600,
                utc_date=dt.date(2020, 1, 1) if index % 2 else dt.date(2024, 1, 1),
                year=2020 if index % 2 else 2024,
                result="1-0" if index % 3 else "0-1",
                split="train" if index % 4 else "test",
            )
        )
    return rows


def _stores(tmp_path, dim=8):
    lance = LancePositionStore(tmp_path / "db", dim=dim, create=True, overwrite=True)
    memory = MemoryPositionStore(dim=dim)
    return lance, memory


def test_add_and_count(tmp_path):
    with given("an empty LanceDB position store"):
        lance, _ = _stores(tmp_path)

    with when("twelve rows are added"):
        added = lance.add(_rows())

    with then("the store reports the same row count"):
        assert added == 12
        assert lance.count() == 12
    return


def test_memory_and_lance_agree_on_nearest(tmp_path):
    with given("the same rows in a LanceDB store and a brute-force oracle"):
        rows = _rows()
        lance, memory = _stores(tmp_path)
        lance.add(rows)
        memory.add(rows)

    with when("both stores search for one of the stored vectors"):
        query = rows[5].vector
        lance_hits = lance.search(query, limit=3)
        memory_hits = memory.search(query, limit=3)

    with then("the top result and its distance agree"):
        assert lance_hits[0].game_id == memory_hits[0].game_id
        assert lance_hits[0].ply == memory_hits[0].ply
        assert lance_hits[0].distance == pytest.approx(
            memory_hits[0].distance, abs=1e-3
        )
    return


def test_metadata_filters_are_applied(tmp_path):
    with given("rows with mixed Elo, result, split and year"):
        rows = _rows()
        lance, memory = _stores(tmp_path)
        lance.add(rows)
        memory.add(rows)
        position_filter = PositionFilter(
            min_elo=2500,
            result="1-0",
            split="train",
            min_year=2024,
        )

    with when("both stores search with the same filter"):
        lance_hits = lance.search(rows[0].vector, limit=10, filter=position_filter)
        memory_hits = memory.search(rows[0].vector, limit=10, filter=position_filter)

    with then("only matching rows are returned by either store"):
        expected = {
            (row.game_id, row.ply)
            for row in rows
            if row.white_elo >= 2500
            and row.black_elo >= 2500
            and row.result == "1-0"
            and row.split == "train"
            and row.year >= 2024
        }
        assert {(hit.game_id, hit.ply) for hit in lance_hits} == expected
        assert {(hit.game_id, hit.ply) for hit in memory_hits} == expected
        assert all(hit.white_elo >= 2500 for hit in lance_hits)
        assert all(hit.result == "1-0" for hit in lance_hits)
        assert all(hit.year >= 2024 for hit in lance_hits)
    return


def test_date_filter_uses_date_literals(tmp_path):
    with given("rows from 2020 and 2024"):
        rows = _rows()
        lance, memory = _stores(tmp_path)
        lance.add(rows)
        memory.add(rows)
        position_filter = PositionFilter(min_date=dt.date(2024, 1, 1))

    with when("both stores filter by date"):
        lance_hits = lance.search(rows[0].vector, limit=20, filter=position_filter)
        memory_hits = memory.search(rows[0].vector, limit=20, filter=position_filter)

    with then("only 2024 positions are returned"):
        assert {(hit.game_id, hit.ply) for hit in lance_hits} == {
            (hit.game_id, hit.ply) for hit in memory_hits
        }
        assert all(hit.utc_date >= dt.date(2024, 1, 1) for hit in lance_hits)
    return


def test_exclude_game_id(tmp_path):
    with given("a store containing three games"):
        rows = _rows()
        lance, _ = _stores(tmp_path)
        lance.add(rows)

    with when("searching while excluding game 0"):
        hits = lance.search(
            rows[0].vector, limit=20, filter=PositionFilter(exclude_game_id=0)
        )

    with then("no hit comes from the excluded game"):
        assert hits
        assert all(hit.game_id != 0 for hit in hits)
    return


def test_identical_vector_has_zero_distance(tmp_path):
    with given("a store containing a known vector"):
        row = make_position_row(vector=np.eye(8, dtype=np.float32)[0])
        lance, _ = _stores(tmp_path)
        lance.add([row])

    with when("the same vector is used as the query"):
        hits = lance.search(row.vector, limit=1)

    with then("the nearest hit is the stored row at distance zero"):
        assert hits[0].game_id == row.game_id
        assert hits[0].ply == row.ply
        assert hits[0].distance == pytest.approx(0.0, abs=1e-3)
    return


def test_indexed_search_returns_the_exact_nearest(tmp_path):
    with given("a store with an IVF-PQ vector index"):
        rng = np.random.default_rng(3)
        vectors = rng.normal(size=(64, 16)).astype(np.float32)
        vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
        rows = [
            make_position_row(vector=vector, game_id=index // 2, ply=index % 2 + 1)
            for index, vector in enumerate(vectors)
        ]
        lance = LancePositionStore(tmp_path / "db", dim=16, create=True, overwrite=True)
        lance.add(rows)
        lance.create_index(index_type="ivf_pq", num_partitions=2, num_sub_vectors=4)

    with when("the indexed search and the exact scan are run"):
        indexed = lance.search(vectors[7], limit=3, nprobes=2, refine_factor=4)
        exact = lance.search(vectors[7], limit=3, exact=True)

    with then("the top result is the stored vector in both modes"):
        assert indexed[0].game_id == 3
        assert indexed[0].ply == 2
        assert exact[0].game_id == 3
        assert exact[0].ply == 2
    return


def test_f16_storage_stays_within_budget(tmp_path):
    with given("a 512-dimensional store with 64 positions"):
        rng = np.random.default_rng(5)
        vectors = rng.normal(size=(64, 512)).astype(np.float32)
        vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
        rows = [
            make_position_row(vector=vector, game_id=index // 8, ply=index % 8 + 1)
            for index, vector in enumerate(vectors)
        ]
        lance = LancePositionStore(
            tmp_path / "db", dim=512, create=True, overwrite=True
        )
        lance.add(rows)

    with when("the on-disk size is measured"):
        bytes_per_row = lance.byte_size() / lance.count()

    with then("the compact schema keeps roughly one kilobyte per position"):
        assert bytes_per_row < 1300
    return
