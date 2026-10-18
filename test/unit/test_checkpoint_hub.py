import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from bdd import given, then, when
from src.training import checkpoint_hub
from src.training.checkpoint_hub import (
    build_metadata,
    collect_data_provenance,
    upload_checkpoint,
)


@pytest.fixture
def uploads(monkeypatch):
    """Stub huggingface_hub.batch_bucket_files; records upload calls."""
    calls: list[dict] = []

    def fake_batch_bucket_files(bucket_id, add=None, remove=None, **kwargs):
        calls.append({"bucket_id": bucket_id, "add": list(add or [])})
        return calls[-1]

    monkeypatch.setattr(checkpoint_hub, "batch_bucket_files", fake_batch_bucket_files)
    return calls


@pytest.fixture
def checkpoint_file(tmp_path):
    path = tmp_path / "last.pt"
    path.write_bytes(b"checkpoint-bytes")
    return path


class TestUploadCheckpoint:
    def test_uploads_checkpoint_and_metadata(self, uploads, checkpoint_file):
        with given("a local checkpoint file and run metadata"):
            metadata = {"step": 100, "best_val_loss": 6.65}

        with when("the checkpoint is uploaded as last.pt"):
            upload_checkpoint(
                checkpoint_file,
                bucket_id="patrickfrank1/chesspos-checkpoints",
                remote_name="last",
                metadata=metadata,
            )

        with then("the .pt file and its .meta.json sidecar are uploaded together"):
            call = uploads[0]
            assert call["bucket_id"] == "patrickfrank1/chesspos-checkpoints"
            local, remote = call["add"][0]
            assert local == checkpoint_file
            assert remote == "checkpoints/last.pt"
            meta_source, meta_remote = call["add"][1]
            assert meta_remote == "checkpoints/last.meta.json"
            assert json.loads(meta_source) == metadata
        return

    def test_custom_prefix(self, uploads, checkpoint_file):
        with given("a checkpoint and a custom upload prefix"):
            pass

        with when("the checkpoint is uploaded with prefix runs/exp1"):
            upload_checkpoint(
                checkpoint_file,
                bucket_id="patrickfrank1/chesspos-checkpoints",
                remote_name="best_step2000",
                metadata={},
                prefix="runs/exp1",
            )

        with then("both files land under the custom prefix"):
            assert uploads[0]["add"][0][1] == "runs/exp1/best_step2000.pt"
            assert uploads[0]["add"][1][1] == "runs/exp1/best_step2000.meta.json"
        return

    def test_versioned_best_names_are_distinct(self, uploads, checkpoint_file):
        with given("two new best checkpoints at different steps"):
            pass

        with when("each is uploaded with its step-versioned name"):
            upload_checkpoint(
                checkpoint_file,
                bucket_id="b",
                remote_name="best_step1000",
                metadata={},
            )
            upload_checkpoint(
                checkpoint_file,
                bucket_id="b",
                remote_name="best_step2000",
                metadata={},
            )

        with then("neither upload overwrites the other"):
            remotes = {entry[1] for call in uploads for entry in call["add"]}
            assert remotes == {
                "checkpoints/best_step1000.pt",
                "checkpoints/best_step1000.meta.json",
                "checkpoints/best_step2000.pt",
                "checkpoints/best_step2000.meta.json",
            }
        return


def write_shard(directory: Path, archive: str, seeds: tuple[int, ...]):
    shard_dir = directory / archive
    shard_dir.mkdir(parents=True)
    table = pa.table(
        {
            "packed": pa.array(
                [[3, 4, 2, 5, 2, 6]] * len(seeds), type=pa.list_(pa.int64())
            )
        }
    )
    pq.write_table(table, shard_dir / "batch_0000.parquet")
    (shard_dir / "batch_0000.meta.json").touch()
    return


class TestCollectDataProvenance:
    def test_describes_nested_shards_and_ignores_sidecars(self, tmp_path):
        with given("a data directory with two nested shards and meta sidecars"):
            directory = tmp_path / "train"
            write_shard(directory, "arch_a", seeds=(1, 2))
            write_shard(directory, "arch_b", seeds=(3,))

        with when("the data provenance is collected"):
            provenance = collect_data_provenance(directory)

        with then("shards, sizes and game counts are reported, sidecars ignored"):
            assert provenance["total_games"] == 3
            assert provenance["total_bytes"] > 0
            assert [shard["path"] for shard in provenance["shards"]] == [
                "arch_a/batch_0000.parquet",
                "arch_b/batch_0000.parquet",
            ]
            assert all(shard["games"] > 0 for shard in provenance["shards"])
            assert [shard["games"] for shard in provenance["shards"]] == [2, 1]
            assert all(shard["size"] > 0 for shard in provenance["shards"])
        return

    def test_empty_directory(self, tmp_path):
        with given("an empty data directory"):
            directory = tmp_path / "train"
            directory.mkdir()

        with when("the data provenance is collected"):
            provenance = collect_data_provenance(directory)

        with then("the provenance is empty but well-formed"):
            assert provenance == {"shards": [], "total_bytes": 0, "total_games": 0}
        return


class TestBuildMetadata:
    def test_contains_training_model_and_data_sections(self, tmp_path):
        with given("a training run's configuration and data provenance"):
            data_dir = tmp_path / "train"
            write_shard(data_dir, "arch_a", seeds=(1,))
            provenance = collect_data_provenance(data_dir)
            val_provenance = collect_data_provenance(tmp_path)

        with when("the checkpoint metadata is built"):
            metadata = build_metadata(
                training_config={"steps": 100, "batch_size": 8},
                model_config={"d_model": 64},
                n_params=1234,
                step=100,
                best_val_loss=6.5,
                data_repo="patrickfrank1/chess-positions",
                data_limit_gb=10,
                train_provenance=provenance,
                val_provenance=val_provenance,
                git_commit=None,
            )

        with then("training, model, step and data sections are all present"):
            assert metadata["training"] == {"steps": 100, "batch_size": 8}
            assert metadata["model"] == {"d_model": 64, "n_params": 1234}
            assert metadata["step"] == 100
            assert metadata["best_val_loss"] == 6.5
            assert metadata["data"]["repo"] == "patrickfrank1/chess-positions"
            assert metadata["data"]["limit_gb"] == 10
            assert metadata["data"]["train"] == provenance
            assert metadata["data"]["val"] == val_provenance
        return

    def test_metadata_is_json_serializable(self, tmp_path):
        with given("built metadata"):
            metadata = build_metadata(
                training_config={"steps": 100},
                model_config={"d_model": 64},
                n_params=1,
                step=100,
                best_val_loss=6.5,
                data_repo=None,
                data_limit_gb=None,
                train_provenance=collect_data_provenance(tmp_path),
                val_provenance=collect_data_provenance(tmp_path),
                git_commit="abc123",
            )

        with when("the metadata is serialized"):
            encoded = json.dumps(metadata, sort_keys=True)

        with then("it round-trips through JSON"):
            assert json.loads(encoded) == metadata
        return
