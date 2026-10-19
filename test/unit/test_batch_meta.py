import hashlib
import json

import numpy as np

from bdd import given, then, when
from src.dataset.batch_meta import BatchMeta, build_batch_meta, config_fingerprint
from src.dataset.config import GameSubsampling, GameSubsampleTier, TimeControlFilter
from src.dataset.token_stream import CLS, SEP, VOCAB_SIZE


def make_games_table(num_games=2, n_positions_per_game=3):
    import pyarrow as pa

    rows = []
    for i in range(num_games):
        rows.append(
            {
                "packed": [3, 4, 2] * n_positions_per_game,
                "n_positions": n_positions_per_game,
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
        return

    def test_fingerprint_changes_with_subsampling(self):
        with given("two different subsampling configurations"):
            lenient = GameSubsampling(tiers=[GameSubsampleTier(min_elo=0, rate=1.0)])
            strict = GameSubsampling(tiers=[GameSubsampleTier(min_elo=2400, rate=1.0)])

        with when("a fingerprint is computed for each"):
            fp_lenient = config_fingerprint(lenient, TimeControlFilter())
            fp_strict = config_fingerprint(strict, TimeControlFilter())

        with then("the fingerprints differ"):
            assert fp_lenient != fp_strict
        return

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
        return


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
                batch_number=0,
                chunk_index=0,
                seq=0,
                provider="lichess",
                archive_id="lichess_db_standard_rated_2023-01",
                url="https://database.lichess.org/standard/arch.pgn.zst",
                duplicates_dropped=5,
                config_fingerprint="fp",
                utc_dates=["2023-01-02", "2023-01-31", "2023-01-15"],
                **_generation_config_kwargs(),
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
        return

    def test_missing_dates_yield_none(self):
        with given("a batch whose games have no UTCDate headers"):
            table = make_games_table(num_games=1)

        with when("the batch meta is built without dates"):
            meta = build_batch_meta(
                table=table,
                file_bytes=b"",
                split="train",
                batch_number=0,
                chunk_index=0,
                seq=0,
                provider="local",
                archive_id="arch",
                url=None,
                duplicates_dropped=0,
                config_fingerprint="fp",
                utc_dates=[],
                **_generation_config_kwargs(),
            )

        with then("the date range is None"):
            assert meta.date_min is None
            assert meta.date_max is None
        return

    def test_json_roundtrip(self):
        with given("a built batch meta with chunk provenance"):
            table = make_games_table(num_games=1)
            meta = build_batch_meta(
                table=table,
                file_bytes=b"bytes",
                split="test",
                batch_number=7,
                chunk_index=2065,
                seq=1,
                provider="lichess",
                archive_id="arch",
                url="u",
                duplicates_dropped=0,
                config_fingerprint="fp",
                utc_dates=["2023-01-15"],
                **_generation_config_kwargs(),
            )

        with when("the meta is serialized to json and back"):
            restored = BatchMeta.from_json(meta.to_json())

        with then("the restored meta equals the original"):
            assert restored == meta
        return

    def test_batch_file_name(self):
        with given("a batch meta numbered 7 that completed chunk 2065"):
            table = make_games_table(num_games=1)
            meta = build_batch_meta(
                table=table,
                file_bytes=b"",
                split="train",
                batch_number=7,
                chunk_index=2065,
                seq=1,
                provider="lichess",
                archive_id="arch",
                url=None,
                duplicates_dropped=0,
                config_fingerprint="fp",
                utc_dates=[],
                **_generation_config_kwargs(),
            )

        with when("the repo file paths are derived"):
            parquet_path = meta.repo_parquet_path()
            meta_path = meta.repo_meta_path()

        with then("the name is the consecutive batch number, independent of chunking"):
            assert meta.file_name == "batch_0007.parquet"
            assert parquet_path == "train/arch/batch_0007.parquet"
            assert meta_path == "train/arch/batch_0007.meta.json"
        return


def _generation_config_kwargs():
    return {
        "subsampling": GameSubsampling(
            tiers=[
                GameSubsampleTier(min_elo=2200, rate=1.0),
                GameSubsampleTier(min_elo=1900, rate=0.1),
            ]
        ),
        "time_control_filter": TimeControlFilter(min_seconds=300),
        "train_ratio": 0.95,
        "batch_size": 100_000,
    }


class TestMetaConfigValues:
    def test_meta_records_generation_config_values(self):
        with given("a batch built with an explicit generation configuration"):
            table = make_games_table(num_games=1)

        with when("the batch meta is built and serialized"):
            meta = build_batch_meta(
                table=table,
                file_bytes=b"",
                split="train",
                batch_number=0,
                chunk_index=0,
                seq=0,
                provider="lichess",
                archive_id="arch",
                url=None,
                duplicates_dropped=0,
                config_fingerprint="fp",
                utc_dates=[],
                **_generation_config_kwargs(),
            )
            data = json.loads(meta.to_json())

        with then(
            "the config values themselves are recorded, not just the fingerprint"
        ):
            assert data["config"]["subsampling"]["tiers"] == [
                {"min_elo": 2200, "rate": 1.0},
                {"min_elo": 1900, "rate": 0.1},
            ]
            assert data["config"]["time_control_filter"]["min_seconds"] == 300
            assert data["config"]["train_ratio"] == 0.95
            assert data["config"]["batch_size"] == 100_000
            assert data["provenance"]["config_fingerprint"] == "fp"
        return

    def test_config_values_survive_json_roundtrip(self):
        with given("a batch meta built with an explicit generation configuration"):
            table = make_games_table(num_games=1)
            meta = build_batch_meta(
                table=table,
                file_bytes=b"",
                split="test",
                batch_number=7,
                chunk_index=2065,
                seq=1,
                provider="lichess",
                archive_id="arch",
                url="u",
                duplicates_dropped=0,
                config_fingerprint="fp",
                utc_dates=[],
                **_generation_config_kwargs(),
            )

        with when("the meta is serialized to json and back"):
            restored = BatchMeta.from_json(meta.to_json())

        with then("the restored meta equals the original, config values included"):
            assert restored == meta
        return


class TestUnigramCountsInMeta:
    def test_build_batch_meta_records_unigram_histogram(self):
        with given("a finalized batch table with packed token streams"):
            table = make_games_table(num_games=2, n_positions_per_game=3)

        with when("the batch meta is built"):
            meta = build_batch_meta(
                table=table,
                file_bytes=b"",
                split="train",
                batch_number=0,
                chunk_index=0,
                seq=0,
                provider="lichess",
                archive_id="arch",
                url="u",
                duplicates_dropped=0,
                config_fingerprint="fp",
                utc_dates=[],
                **_generation_config_kwargs(),
            )

        with then("the histogram excludes SEP tokens and each stream's leading CLS"):
            packed = np.concatenate(
                [
                    np.asarray(row, dtype=np.int64)
                    for row in table.column("packed").to_pylist()
                ]
            )
            expected = np.bincount(packed, minlength=VOCAB_SIZE)
            expected[SEP] -= int((packed == SEP).sum())
            leading = [row[0] for row in table.column("packed").to_pylist()]
            expected[CLS] -= sum(1 for token in leading if token == CLS)
            assert meta.unigram_counts == expected.tolist()
        return

    def test_unigram_histogram_survives_json_roundtrip(self):
        with given("a batch meta built from a packed table"):
            meta = build_batch_meta(
                table=make_games_table(num_games=1),
                file_bytes=b"",
                split="train",
                batch_number=0,
                chunk_index=0,
                seq=0,
                provider="lichess",
                archive_id="arch",
                url="u",
                duplicates_dropped=0,
                config_fingerprint="fp",
                utc_dates=[],
                **_generation_config_kwargs(),
            )

        with when("the meta is serialized to json and back"):
            restored = BatchMeta.from_json(meta.to_json())

        with then("the histogram is preserved"):
            assert restored.unigram_counts == meta.unigram_counts
        return

    def test_from_json_tolerates_old_sidecars_without_histogram(self):
        with given("a sidecar JSON produced before histograms were recorded"):
            meta = build_batch_meta(
                table=make_games_table(num_games=1),
                file_bytes=b"",
                split="train",
                batch_number=0,
                chunk_index=0,
                seq=0,
                provider="lichess",
                archive_id="arch",
                url="u",
                duplicates_dropped=0,
                config_fingerprint="fp",
                utc_dates=[],
                **_generation_config_kwargs(),
            )
            payload = json.loads(meta.to_json())
            del payload["stats"]["unigram_counts"]

        with when("the old sidecar is parsed"):
            restored = BatchMeta.from_json(json.dumps(payload))

        with then("the histogram is simply absent"):
            assert restored.unigram_counts is None
        return
