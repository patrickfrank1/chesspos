import datetime as dt

import pyarrow as pa
import pytest

from bdd import given, then, when
from src.search.schema import (
    CODE_TO_RESULT,
    CODE_TO_SPLIT,
    RESULT_TO_CODE,
    SPLIT_TO_CODE,
    UNKNOWN_DATE,
    IndexManifest,
    game_id_to_hex,
    game_id_to_int,
    position_schema,
    rows_to_table,
)
from position_support import make_position_row


def test_game_id_roundtrips_through_uint64():
    with given("a 16-hex-character game id"):
        value = "0123456789abcdef"

    with when("the id is converted to an integer and back"):
        encoded = game_id_to_int(value)
        decoded = game_id_to_hex(encoded)

    with then("the value survives exactly"):
        assert encoded == 0x0123456789ABCDEF
        assert decoded == value
    return


def test_invalid_game_id_is_rejected():
    with given("a game id that is not 16 hexadecimal characters"):
        value = "not-a-game-id"

    with when("the id is converted"):
        with pytest.raises(ValueError):
            game_id_to_int(value)

    with then("no integer is produced"):
        pass
    return


def test_result_and_split_codes_roundtrip():
    with given("the dataset result and split vocabularies"):
        results = ["1-0", "0-1", "1/2-1/2", "*", ""]
        splits = ["train", "test"]

    with when("each value is converted to a code and back"):
        decoded_results = [CODE_TO_RESULT[RESULT_TO_CODE[value]] for value in results]
        decoded_splits = [CODE_TO_SPLIT[SPLIT_TO_CODE[value]] for value in splits]

    with then("unknown results map to the unknown code and splits are exact"):
        assert decoded_results == ["1-0", "0-1", "1/2-1/2", "*", "*"]
        assert decoded_splits == splits
    return


def test_schema_uses_compact_non_nullable_columns():
    with given("a 512-dimensional index schema"):
        schema = position_schema(512)

    with when("the fields are inspected"):
        fields = {field.name: field for field in schema}

    with then("every column uses the compact type and allows no nulls"):
        assert fields["vector"].type == pa.list_(pa.float16(), 512)
        assert fields["packed_position"].type == pa.binary(34)
        assert fields["game_id"].type == pa.uint64()
        assert fields["ply"].type == pa.uint16()
        assert fields["white_elo"].type == pa.uint16()
        assert fields["black_elo"].type == pa.uint16()
        assert fields["utc_date"].type == pa.date32()
        assert fields["year"].type == pa.uint16()
        assert fields["result"].type == pa.uint8()
        assert fields["split"].type == pa.uint8()
        assert fields["n_pieces"].type == pa.uint8()
        assert all(not field.nullable for field in schema)
    return


def test_rows_to_table_converts_values():
    with given("two position rows with metadata"):
        rows = [
            make_position_row(ply=1, result="1-0", split="train"),
            make_position_row(
                ply=2,
                game_id=0xFEDCBA9876543210,
                utc_date=dt.date(2025, 6, 1),
                year=2025,
                result="0-1",
                split="test",
                white_elo=2400,
                black_elo=2300,
            ),
        ]

    with when("the rows are converted to an Arrow table"):
        table = rows_to_table(rows, dim=8)

    with then("the table carries the compact values"):
        assert table.schema == position_schema(8)
        assert table.column("game_id").to_pylist() == [
            0x0123456789ABCDEF,
            0xFEDCBA9876543210,
        ]
        assert table.column("ply").to_pylist() == [1, 2]
        assert table.column("year").to_pylist() == [2024, 2025]
        assert table.column("utc_date").to_pylist() == [
            dt.date(2024, 1, 1),
            dt.date(2025, 6, 1),
        ]
        assert table.column("result").to_pylist() == [
            RESULT_TO_CODE["1-0"],
            RESULT_TO_CODE["0-1"],
        ]
        assert table.column("split").to_pylist() == [
            SPLIT_TO_CODE["train"],
            SPLIT_TO_CODE["test"],
        ]
        assert table.column("vector").type == pa.list_(pa.float16(), 8)
    return


def test_unknown_date_sentinel_is_clear():
    with given("the schema sentinel for missing dates"):
        expected = dt.date(1970, 1, 1)

    with when("the sentinel is read"):
        actual = UNKNOWN_DATE

    with then("it is the Unix epoch"):
        assert actual == expected
    return


def test_manifest_roundtrips_as_json():
    with given("an index manifest"):
        manifest = IndexManifest(
            index_version=1,
            checkpoint="data/models/run/last.pt",
            checkpoint_sha256="abc123",
            feature="native_positions",
            dim=512,
            dtype="float16",
            schema_version=1,
            rows=123,
            games=4,
            date_coverage=0.5,
            lancedb_version="0.41.0",
            created_at="2026-11-23T00:00:00+00:00",
        )

    with when("the manifest is serialised and parsed"):
        restored = IndexManifest.from_dict(manifest.to_dict())

    with then("the values survive"):
        assert restored == manifest
    return
