import json
from pathlib import Path

from bdd import given, then, when
from etl_support import make_lichess_pgn
from src.dataset.archive_pipeline import ArchivePipeline
from src.dataset.config import (
    DatasetConfig,
    GameSubsampling,
    GameSubsampleTier,
    PreprocessingConfig,
    SourceConfig,
)


def read_all(out: Path) -> dict[str, bytes]:
    return {
        str(p.relative_to(out)): p.read_bytes() for p in sorted(out.rglob("*.parquet"))
    }


def make_pipeline(data_path: str) -> ArchivePipeline:
    return ArchivePipeline(
        dataset_config=DatasetConfig(
            repo_name="test/chesspos-test",
            batch_size=10,
            train_ratio=0.8,
            data_path=data_path,
        ),
        preprocessing_config=PreprocessingConfig(
            worker_count=1,
            memory_limit_mb=1024,
            debug=True,
            subsampling=GameSubsampling(tiers=[GameSubsampleTier(min_elo=0, rate=1.0)]),
        ),
        source_config=SourceConfig(data_path=data_path),
    )


class TestArchivePipelineDryRun:
    def test_end_to_end_local_dry_run(self, tmp_path):
        with given("a local PGN file containing two distinct games and one duplicate"):
            data_dir = tmp_path / "data"
            data_dir.mkdir()
            games = (
                make_lichess_pgn("Game One", date="2023.01.15")
                + make_lichess_pgn("Game Two", date="2023.01.16")
                + make_lichess_pgn("Game One", date="2023.01.15")
            )
            (data_dir / "test.pgn").write_bytes(games)
            output_dir = tmp_path / "out"
            pipeline = make_pipeline(str(data_dir))

        with when("the pipeline runs in dry-run mode"):
            result = pipeline.run(dry_run=True, output_dir=str(output_dir))

        with then(
            "the archive is processed with dedup, metadata and completion marker"
        ):
            assert result.processed == ["test"]
            assert result.skipped == []

            parquet_files = sorted(output_dir.glob("*/*/*.parquet"))
            meta_files = sorted(output_dir.glob("*/*/*.meta.json"))
            assert len(parquet_files) >= 1
            assert len(parquet_files) == len(meta_files)
            assert (output_dir / "archives" / "test" / "_COMPLETE.json").exists()

            metas = [json.loads(p.read_text()) for p in meta_files]
            assert sum(m["stats"]["num_games"] for m in metas) == 2
            assert sum(m["stats"]["duplicates_dropped"] for m in metas) == 1
            dates = [m["range"]["date_min"] for m in metas] + [
                m["range"]["date_max"] for m in metas
            ]
            assert "2023-01-15" in dates
            assert "2023-01-16" in dates

    def test_rerun_skips_completed_archive(self, tmp_path):
        with given("a pipeline that has already processed its archive in dry-run mode"):
            data_dir = tmp_path / "data"
            data_dir.mkdir()
            (data_dir / "test.pgn").write_bytes(make_lichess_pgn("Game One"))
            output_dir = tmp_path / "out"
            pipeline = make_pipeline(str(data_dir))
            pipeline.run(dry_run=True, output_dir=str(output_dir))
            files_after_first_run = sorted(
                p.relative_to(output_dir) for p in output_dir.rglob("*")
            )

        with when("the pipeline runs again with the same output"):
            result = pipeline.run(dry_run=True, output_dir=str(output_dir))

        with then("the archive is skipped and no files change"):
            assert result.skipped == ["test"]
            assert result.processed == []
            files_after_second_run = sorted(
                p.relative_to(output_dir) for p in output_dir.rglob("*")
            )
            assert files_after_second_run == files_after_first_run

    def test_parquet_output_is_deterministic(self, tmp_path):
        with given("the same input processed into two different output directories"):
            data_dir = tmp_path / "data"
            data_dir.mkdir()
            games = make_lichess_pgn("Game One") + make_lichess_pgn("Game Two")
            (data_dir / "test.pgn").write_bytes(games)
            out_a, out_b = tmp_path / "out_a", tmp_path / "out_b"
            make_pipeline(str(data_dir)).run(dry_run=True, output_dir=str(out_a))
            make_pipeline(str(data_dir)).run(dry_run=True, output_dir=str(out_b))

        with when("the parquet outputs are compared"):
            files_a = read_all(out_a)
            files_b = read_all(out_b)

        with then("the parquet bytes are identical"):
            assert set(files_a) == set(files_b)
            assert files_a == files_b
