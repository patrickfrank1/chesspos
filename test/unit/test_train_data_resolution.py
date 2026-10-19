from argparse import Namespace
from pathlib import Path

from bdd import given, then, when
from src.run import train_masked_transformer as train


def make_args(**overrides) -> Namespace:
    defaults = dict(
        train_dir="data/processed/train",
        val_dir="data/processed/test",
        data_repo=None,
        data_cache=None,
        data_limit_gb=None,
    )
    defaults.update(overrides)
    return Namespace(**defaults)


class TestResolveDataDirs:
    def test_local_dirs_without_repo(self, monkeypatch, tmp_path):
        with given("training args with local data directories and no hub repo"):
            args = make_args(
                train_dir=str(tmp_path / "train"), val_dir=str(tmp_path / "test")
            )

            def fail_download(*args, **kwargs):
                raise AssertionError("download_split must not be called")

            monkeypatch.setattr(train, "download_split", fail_download)

        with when("the data directories are resolved"):
            train_dir, val_dir = train.resolve_data_dirs(args)

        with then("the local directories are returned unchanged"):
            assert train_dir == Path(tmp_path / "train")
            assert val_dir == Path(tmp_path / "test")
        return

    def test_repo_resolves_both_splits(self, monkeypatch, tmp_path):
        with given("training args pointing at a hub dataset repo"):
            args = make_args(data_repo="user/chesspos")
            downloads: list[tuple] = []

            def fake_download_split(
                repo_id, split, cache_dir=None, max_bytes=None, log=None
            ):
                downloads.append((repo_id, split, cache_dir, max_bytes))
                path = tmp_path / "snapshot" / split
                path.mkdir(parents=True, exist_ok=True)
                return path

            monkeypatch.setattr(train, "download_split", fake_download_split)

        with when("the data directories are resolved"):
            train_dir, val_dir = train.resolve_data_dirs(args)

        with then(
            "train resolves to the downloaded train split and val to the test split"
        ):
            assert downloads == [
                ("user/chesspos", "train", None, None),
                ("user/chesspos", "test", None, None),
            ]
            assert train_dir == tmp_path / "snapshot" / "train"
            assert val_dir == tmp_path / "snapshot" / "test"
        return

    def test_limit_gb_applies_to_train_only(self, monkeypatch, tmp_path):
        with given("training args with a 2 GB data limit"):
            args = make_args(data_repo="user/chesspos", data_limit_gb=2)
            downloads: list[tuple] = []

            def fake_download_split(
                repo_id, split, cache_dir=None, max_bytes=None, log=None
            ):
                downloads.append((split, max_bytes))
                path = tmp_path / "snapshot" / split
                path.mkdir(parents=True, exist_ok=True)
                return path

            monkeypatch.setattr(train, "download_split", fake_download_split)

        with when("the data directories are resolved"):
            train.resolve_data_dirs(args)

        with then(
            "the train split is bounded by 2 GB in bytes and the test split is unbounded"
        ):
            assert downloads == [
                ("train", 2 * 1024**3),
                ("test", None),
            ]
        return

    def test_repo_takes_precedence_over_dirs(self, monkeypatch, tmp_path):
        with given("training args with both a hub repo and local directories"):
            args = make_args(
                data_repo="user/chesspos",
                train_dir="/nonexistent/train",
                val_dir="/nonexistent/test",
            )

            def fake_download_split(
                repo_id, split, cache_dir=None, max_bytes=None, log=None
            ):
                path = tmp_path / split
                path.mkdir(parents=True, exist_ok=True)
                return path

            monkeypatch.setattr(train, "download_split", fake_download_split)

        with when("the data directories are resolved"):
            train_dir, val_dir = train.resolve_data_dirs(args)

        with then("the hub splits win and the local dirs are ignored"):
            assert train_dir == tmp_path / "train"
            assert val_dir == tmp_path / "test"
        return

    def test_cache_dir_is_forwarded(self, monkeypatch, tmp_path):
        with given("training args with a hub repo and an explicit cache directory"):
            args = make_args(data_repo="user/chesspos", data_cache=str(tmp_path / "c"))
            seen: list[str | None] = []

            def fake_download_split(
                repo_id, split, cache_dir=None, max_bytes=None, log=None
            ):
                seen.append(cache_dir)
                path = tmp_path / split
                path.mkdir(parents=True, exist_ok=True)
                return path

            monkeypatch.setattr(train, "download_split", fake_download_split)

        with when("the data directories are resolved"):
            train.resolve_data_dirs(args)

        with then("the cache directory is passed to both split downloads"):
            assert seen == [str(tmp_path / "c"), str(tmp_path / "c")]
        return
