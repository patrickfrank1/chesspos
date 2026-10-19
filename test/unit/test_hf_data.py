from pathlib import Path

import pytest

from bdd import given, then, when
from src.training import hf_data


def make_snapshot(root: Path, nested: bool = False) -> Path:
    """Fabricate a snapshot directory matching the Hub repo layout."""
    for split in ("train", "test"):
        if nested:
            shard_dir = root / split / "archive_2020-01"
        else:
            shard_dir = root / split
        shard_dir.mkdir(parents=True, exist_ok=True)
        (shard_dir / "batch_0000_0001.parquet").touch()
        (shard_dir / "batch_0000_0001.meta.json").touch()
    return root


@pytest.fixture
def hub_calls(monkeypatch, tmp_path):
    """Stub huggingface_hub.snapshot_download; records calls, builds layout."""
    calls: list[dict] = []

    def fake_snapshot_download(
        repo_id, repo_type=None, allow_patterns=None, cache_dir=None, **kwargs
    ):
        calls.append(
            {
                "repo_id": repo_id,
                "repo_type": repo_type,
                "allow_patterns": allow_patterns,
                "cache_dir": cache_dir,
            }
        )
        return str(make_snapshot(tmp_path / "snapshot"))

    monkeypatch.setattr(hf_data, "snapshot_download", fake_snapshot_download)
    return calls


class TestDownloadSplit:
    def test_returns_split_directory(self, hub_calls, tmp_path):
        with given("a hub dataset snapshot with train and test splits"):
            pass

        with when("the train split is downloaded"):
            path = hf_data.download_split("user/chesspos", "train")

        with then("the returned directory is the snapshot's train directory"):
            assert path == tmp_path / "snapshot" / "train"
            assert path.is_dir()
            assert list(path.glob("*.parquet"))
        return

    def test_requests_dataset_repo_and_split_pattern(self, hub_calls):
        with given("a hub dataset download request"):
            pass

        with when("the train split is downloaded"):
            hf_data.download_split("user/chesspos", "train")

        with then(
            "snapshot_download is called with repo_type=dataset and a "
            "split-scoped allow pattern that covers nested shards"
        ):
            assert hub_calls == [
                {
                    "repo_id": "user/chesspos",
                    "repo_type": "dataset",
                    "allow_patterns": ["train/**"],
                    "cache_dir": None,
                }
            ]
        return

    def test_nested_archive_layout_is_found(self, monkeypatch, tmp_path):
        with given("a snapshot using the nested archive layout"):
            root = tmp_path / "snapshot"

            def fake_snapshot_download(**kwargs):
                return str(make_snapshot(root, nested=True))

            monkeypatch.setattr(hf_data, "snapshot_download", fake_snapshot_download)

        with when("the train split is downloaded"):
            path = hf_data.download_split("user/chesspos", "train")

        with then("the nested parquet shards are visible under the returned path"):
            shards = list(path.rglob("*.parquet"))
            assert len(shards) == 1
            assert shards[0].parent == root / "train" / "archive_2020-01"
        return

    def test_cache_dir_is_forwarded(self, hub_calls, tmp_path):
        with given("a caller with an explicit local cache directory"):
            cache = tmp_path / "hf-cache"

        with when("the test split is downloaded"):
            hf_data.download_split("user/chesspos", "test", cache_dir=str(cache))

        with then("the cache directory is passed through to snapshot_download"):
            assert hub_calls[0]["cache_dir"] == str(cache)
        return


class TestDownloadSplitLogging:
    def test_logs_split_download_progress(self, hub_calls, tmp_path):
        with given("a hub dataset download request and a log collector"):
            messages: list[str] = []

        with when("the train split is downloaded with a log callback"):
            hf_data.download_split("user/chesspos", "train", log=messages.append)

        with then(
            "progress messages mention the repo split before and the "
            "destination path after the download"
        ):
            assert any(
                "user/chesspos" in message and "train" in message
                for message in messages
            )
            assert any(
                str(tmp_path / "snapshot" / "train") in message for message in messages
            )
        return

    def test_logs_bounded_shard_selection(self, monkeypatch, tmp_path):
        with given("a repo with three equal train shards and a log collector"):
            files = [
                FakeRepoFile("train/a.parquet", 100),
                FakeRepoFile("train/b.parquet", 100),
                FakeRepoFile("train/c.parquet", 100),
                FakeRepoFile("test/t.parquet", 100),
            ]
            fake_api = FakeApi(files)
            monkeypatch.setattr(hf_data, "HfApi", lambda **kwargs: fake_api)
            monkeypatch.setattr(
                hf_data,
                "snapshot_download",
                lambda repo_id, **kwargs: str(tmp_path / "snapshot"),
            )
            messages: list[str] = []

        with when("the train split is downloaded with a 200-byte limit"):
            hf_data.download_split(
                "user/chesspos", "train", max_bytes=200, log=messages.append
            )

        with then("the log reports the shard inventory and the subset size"):
            assert any("3 parquet shards" in message for message in messages)
            assert any("2 of 3" in message for message in messages)
        return


def shard(path: str, size: int) -> tuple[str, int]:
    return (path, size)


class TestSelectShardSubset:
    def test_all_kept_when_total_fits_budget(self):
        with given("three shards totalling 300 bytes and a 1000-byte budget"):
            shards = [
                shard("s/a.parquet", 100),
                shard("s/b.parquet", 100),
                shard("s/c.parquet", 100),
            ]

        with when("a subset is selected"):
            selected = hf_data.select_shard_subset(shards, max_bytes=1000)

        with then("all shards are kept"):
            assert selected == ["s/a.parquet", "s/b.parquet", "s/c.parquet"]
        return

    def test_evenly_spaced_when_over_budget(self):
        with given("ten equal shards and a budget for four of them"):
            shards = [shard(f"s/{i:02d}.parquet", 100) for i in range(10)]

        with when("a subset is selected"):
            selected = hf_data.select_shard_subset(shards, max_bytes=400)

        with then("four shards are taken at evenly spaced indices across the repo"):
            assert selected == [
                "s/00.parquet",
                "s/02.parquet",
                "s/05.parquet",
                "s/07.parquet",
            ]
        return

    def test_budget_is_respected_with_uneven_sizes(self):
        with given("alternating large and small shards and a tight budget"):
            sizes = [50, 1, 50, 1, 50, 1, 50, 1, 50, 1]
            shards = [shard(f"s/{i:02d}.parquet", size) for i, size in enumerate(sizes)]

        with when("a subset is selected"):
            selected = hf_data.select_shard_subset(shards, max_bytes=150)

        with then("trailing selections are dropped until the budget holds"):
            assert selected == ["s/00.parquet", "s/02.parquet", "s/04.parquet"]
        return

    def test_selection_is_deterministic(self):
        with given("a shard list and a budget"):
            shards = [shard(f"s/{i:02d}.parquet", 100 + i) for i in range(20)]

        with when("the subset is selected twice"):
            first = hf_data.select_shard_subset(shards, max_bytes=700)
            second = hf_data.select_shard_subset(shards, max_bytes=700)

        with then("both selections are identical"):
            assert first == second
        return


class FakeRepoFile:
    def __init__(self, path: str, size: int) -> None:
        self.path = path
        self.size = size
        self.type = "file"
        return


class FakeApi:
    def __init__(self, files: list[FakeRepoFile]) -> None:
        self._files = files
        self.calls: list[dict] = []
        return

    def list_repo_tree(
        self, repo_id, repo_type=None, path_in_repo=None, recursive=False
    ):
        self.calls.append(
            {
                "repo_id": repo_id,
                "repo_type": repo_type,
                "path_in_repo": path_in_repo,
                "recursive": recursive,
            }
        )
        for file in self._files:
            if path_in_repo is None or file.path.startswith(path_in_repo):
                yield file
        return


class TestDownloadSplitWithLimit:
    def test_bounded_download_selects_shards(self, monkeypatch, tmp_path):
        with given("a repo with three equal train shards and a budget for two"):
            files = [
                FakeRepoFile("train/a.parquet", 100),
                FakeRepoFile("train/b.parquet", 100),
                FakeRepoFile("train/c.parquet", 100),
                FakeRepoFile("test/t.parquet", 100),
            ]
            fake_api = FakeApi(files)
            monkeypatch.setattr(hf_data, "HfApi", lambda **kwargs: fake_api)
            recorded: dict = {}

            def fake_snapshot_download(
                repo_id, repo_type=None, allow_patterns=None, **kwargs
            ):
                recorded["repo_type"] = repo_type
                recorded["allow_patterns"] = allow_patterns
                for pattern in allow_patterns:
                    target = tmp_path / "snapshot" / pattern
                    target.parent.mkdir(parents=True, exist_ok=True)
                    target.touch()
                return str(tmp_path / "snapshot")

            monkeypatch.setattr(hf_data, "snapshot_download", fake_snapshot_download)

        with when("the train split is downloaded with a 200-byte limit"):
            path = hf_data.download_split("user/chesspos", "train", max_bytes=200)

        with then("only the first two train shards are requested and present on disk"):
            assert recorded["repo_type"] == "dataset"
            assert recorded["allow_patterns"] == [
                "train/a.parquet",
                "train/b.parquet",
            ]
            assert sorted(p.name for p in path.glob("*.parquet")) == [
                "a.parquet",
                "b.parquet",
            ]
        return

    def test_unbounded_download_keeps_split_pattern(self, hub_calls, tmp_path):
        with given("a repo download without a size limit"):
            pass

        with when("the train split is downloaded"):
            hf_data.download_split("user/chesspos", "train")

        with then("the split-wide allow pattern is used, no shard listing happens"):
            assert hub_calls[0]["allow_patterns"] == ["train/**"]
        return
