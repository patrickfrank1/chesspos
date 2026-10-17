import hashlib

from bdd import given, then, when
from src.dataset.batch_meta import BatchMeta, build_batch_meta, config_fingerprint
from src.dataset.config import GameSubsampling, GameSubsampleTier, TimeControlFilter


def make_games_table(num_games=2, n_positions_per_game=3):
    import pyarrow as pa

    rows = []
    for i in range(num_games):
        rows.append(
            {
                "packed": [3, 4, 2] * n_positions_per_game,
                "n_positions": n_positions_per_game,
                "ply": n_positions_per_game - 1,
                "game_id": f"aaaa{i:012d}",
                "white_elo": 2200,
                "black_elo": 2100,
                "result": "1-0",
            }
        )
    return pa.table(
        {
            "packed": pa.array([r["packed"] for r in rows], type=pa.list_(pa.int16())),
            "n_positions": pa.array([r["n_positions"] for r in rows], type=pa.int32()),
            "ply": pa.array([r["ply"] for r in rows], type=pa.int32()),
            "game_id": pa.array([r["game_id"] for r in rows], type=pa.string()),
            "white_elo": pa.array([r["white_elo"] for r in rows], type=pa.int32()),
            "black_elo": pa.array([r["black_elo"] for r in rows], type=pa.int32()),
            "result": pa.array([r["result"] for r in rows], type=pa.string()),
        }
    )


class TestConfigFingerprint:
    def test_fingerprint_is_deterministic(self):
        with given("a subsampling and time control configuration"):
            subsampling = GameSubsampling(
                tiers=[GameSubsampleTier(min_elo=0, rate=1.0)]
            )

        with when("the fingerprint is computed twice"):
            first = config_fingerprint(subsampling, TimeControlFilter())
            second = config_fingerprint(subsampling, TimeControlFilter())

        with then("both computations agree"):
            assert first == second
            assert len(first) == 40  # sha1 hex

    def test_fingerprint_changes_with_subsampling(self):
        with given("two different subsampling configurations"):
            lenient = GameSubsampling(tiers=[GameSubsampleTier(min_elo=0, rate=1.0)])
            strict = GameSubsampling(tiers=[GameSubsampleTier(min_elo=2400, rate=1.0)])

        with when("a fingerprint is computed for each"):
            fp_lenient = config_fingerprint(lenient, TimeControlFilter())
            fp_strict = config_fingerprint(strict, TimeControlFilter())

        with then("the fingerprints differ"):
            assert fp_lenient != fp_strict

    def test_fingerprint_changes_with_time_control_filter(self):
        with given("two different time control filters"):
            subsampling = GameSubsampling(
                tiers=[GameSubsampleTier(min_elo=0, rate=1.0)]
            )

        with when("a fingerprint is computed for each"):
            fp_default = config_fingerprint(
                subsampling, TimeControlFilter(min_seconds=300)
            )
            fp_disabled = config_fingerprint(
                subsampling, TimeControlFilter(min_seconds=None)
            )

        with then("the fingerprints differ"):
            assert fp_default != fp_disabled


class TestBuildBatchMeta:
    def test_stats_and_range(self):
        with given("a finalized batch table and its source facts"):
            table = make_games_table(num_games=2, n_positions_per_game=3)
            file_bytes = b"parquet-bytes"

        with when("the batch meta is built"):
            meta = build_batch_meta(
                table=table,
                file_bytes=file_bytes,
                split="train",
                chunk_index=0,
                seq=0,
                provider="lichess",
                archive_id="lichess_db_standard_rated_2023-01",
                url="https://database.lichess.org/standard/arch.pgn.zst",
                duplicates_dropped=5,
                config_fingerprint="fp",
                utc_dates=["2023-01-02", "2023-01-31", "2023-01-15"],
            )

        with then("stats, hash, and date range are recorded"):
            assert meta.num_games == 2
            assert meta.num_positions == 6
            assert meta.duplicates_dropped == 5
            assert meta.file_sha256 == hashlib.sha256(file_bytes).hexdigest()
            assert meta.date_min == "2023-01-02"
            assert meta.date_max == "2023-01-31"
            assert meta.split == "train"
            assert meta.archive_id == "lichess_db_standard_rated_2023-01"

    def test_missing_dates_yield_none(self):
        with given("a batch whose games have no UTCDate headers"):
            table = make_games_table(num_games=1)

        with when("the batch meta is built without dates"):
            meta = build_batch_meta(
                table=table,
                file_bytes=b"",
                split="train",
                chunk_index=0,
                seq=0,
                provider="local",
                archive_id="arch",
                url=None,
                duplicates_dropped=0,
                config_fingerprint="fp",
                utc_dates=[],
            )

        with then("the date range is None"):
            assert meta.date_min is None
            assert meta.date_max is None

    def test_json_roundtrip(self):
        with given("a built batch meta"):
            table = make_games_table(num_games=1)
            meta = build_batch_meta(
                table=table,
                file_bytes=b"bytes",
                split="test",
                chunk_index=3,
                seq=1,
                provider="lichess",
                archive_id="arch",
                url="u",
                duplicates_dropped=0,
                config_fingerprint="fp",
                utc_dates=["2023-01-15"],
            )

        with when("the meta is serialized to json and back"):
            restored = BatchMeta.from_json(meta.to_json())

        with then("the restored meta equals the original"):
            assert restored == meta

    def test_batch_file_name(self):
        with given("a batch meta for chunk 3, sequence 1 of the train split"):
            table = make_games_table(num_games=1)
            meta = build_batch_meta(
                table=table,
                file_bytes=b"",
                split="train",
                chunk_index=3,
                seq=1,
                provider="lichess",
                archive_id="arch",
                url=None,
                duplicates_dropped=0,
                config_fingerprint="fp",
                utc_dates=[],
            )

        with when("the repo file paths are derived"):
            parquet_path = meta.repo_parquet_path()
            meta_path = meta.repo_meta_path()

        with then("both files live under split/archive with matching stems"):
            assert parquet_path == "train/arch/batch_0003_0001.parquet"
            assert meta_path == "train/arch/batch_0003_0001.meta.json"
