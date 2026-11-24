import datetime as dt

import numpy as np

from bdd import given, then, when
from src.search.partitioned_store import PartitionedPositionStore, partition_name
from src.search.store import PositionFilter
from position_support import make_position_row


def _rows(start: int, count: int, year: int) -> list:
    rows = []
    for offset in range(count):
        index = start + offset
        vector = np.zeros(8, dtype=np.float32)
        vector[index % 8] = 1.0
        rows.append(
            make_position_row(
                vector=vector,
                game_id=index // 2,
                ply=index % 2 + 1,
                year=year,
                utc_date=dt.date(year, 1, 1),
            )
        )
    return rows


def test_partitions_store_and_count_rows(tmp_path):
    with given("an empty partitioned store"):
        store = PartitionedPositionStore(
            tmp_path / "index", dim=8, create=True, overwrite=True
        )

    with when("rows are added to two yearly partitions"):
        store.add(_rows(0, 4, 2024), partition="year=2024", games=1)
        store.add(_rows(4, 4, 2025), partition="year=2025", games=1)

    with then("the roster and total count include both partitions"):
        assert store.count() == 8
        assert store.games_count() == 2
        assert set(store.partitions) == {"year=2024", "year=2025"}
        assert store.partitions["year=2024"]["rows"] == 4
        assert store.partitions["year=2025"]["rows"] == 4
        assert store.byte_size() > 0
    return


def test_search_fans_out_and_merges_partitions(tmp_path):
    with given("two partitions with distinct vectors"):
        store = PartitionedPositionStore(
            tmp_path / "index", dim=8, create=True, overwrite=True
        )
        rows = _rows(0, 8, 2024)
        store.add(rows[:4], partition="year=2024")
        store.add(rows[4:], partition="year=2025")

    with when("a vector from the 2025 partition is queried"):
        hits = store.search(rows[4].vector, limit=4)

    with then("the nearest hit comes from the matching partition"):
        assert hits
        assert hits[0].game_id == rows[4].game_id
        assert hits[0].distance < 1e-3
    return


def test_year_filter_selects_partitions(tmp_path):
    with given("two yearly partitions"):
        store = PartitionedPositionStore(
            tmp_path / "index", dim=8, create=True, overwrite=True
        )
        store.add(_rows(0, 4, 2024), partition="year=2024")
        store.add(_rows(4, 4, 2025), partition="year=2025")

    with when("the search asks for 2025 only"):
        hits = store.search(
            _rows(0, 1, 2024)[0].vector,
            limit=10,
            filter=PositionFilter(min_year=2025),
        )

    with then("only 2025 rows are returned and 2024 is skipped"):
        assert hits
        assert all(hit.year == 2025 for hit in hits)
    return


def test_explicit_partition_selection(tmp_path):
    with given("two yearly partitions"):
        store = PartitionedPositionStore(
            tmp_path / "index", dim=8, create=True, overwrite=True
        )
        store.add(_rows(0, 4, 2024), partition="year=2024")
        store.add(_rows(4, 4, 2025), partition="year=2025")

    with when("only the 2024 partition is named"):
        hits = store.search(
            _rows(0, 1, 2024)[0].vector,
            limit=10,
            partitions=["year=2024"],
        )

    with then("only 2024 rows are returned"):
        assert hits
        assert all(hit.year == 2024 for hit in hits)
    return


def test_roster_persists_across_reopen(tmp_path):
    with given("a partitioned store with rows"):
        path = tmp_path / "index"
        store = PartitionedPositionStore(path, dim=8, create=True, overwrite=True)
        store.add(_rows(0, 4, 2024), partition="year=2024")
        store.close()

    with when("the store is reopened without create"):
        reopened = PartitionedPositionStore(path, dim=8, create=False)

    with then("the partition roster and row count survive"):
        assert reopened.count() == 4
        assert list(reopened.partitions) == ["year=2024"]
    return


def test_append_existing_partition(tmp_path):
    with given("a partition that already contains rows"):
        path = tmp_path / "index"
        store = PartitionedPositionStore(path, dim=8, create=True, overwrite=True)
        store.add(_rows(0, 4, 2024), partition="year=2024")

    with when("more rows are appended to the same partition"):
        store.add(_rows(4, 2, 2024), partition="year=2024")

    with then("the partition count grows"):
        assert store.count() == 6
        assert store.partitions["year=2024"]["rows"] == 6
    return


def test_partition_name_is_sanitised():
    with given("a source label containing path separators"):
        value = "train/lichess_elite_2024-05"

    with when("the partition label is sanitised"):
        name = partition_name(value)

    with then("the label is safe as a directory name"):
        assert name == "train_lichess_elite_2024-05"
        assert "/" not in name
    return
