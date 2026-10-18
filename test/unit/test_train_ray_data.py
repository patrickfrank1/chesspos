from bdd import given, then, when
from src.training.ray_train import RayTrainingConfig, resolve_data_dirs


def make_config(**overrides) -> RayTrainingConfig:
    values = dict(data_repo=None, data_cache=None, data_limit_gb=None)
    values.update(overrides)
    return RayTrainingConfig(**values)


class TestResolveDataDirs:
    def test_repo_downloads_train_and_test_splits(self, monkeypatch, tmp_path):
        with given("a config pointing at a hub dataset repo"):
            config = make_config(data_repo="user/chesspos")
            downloads: list[tuple] = []

            def fake_download_split(repo_id, split, cache_dir=None, max_bytes=None):
                downloads.append((repo_id, split, cache_dir, max_bytes))
                path = tmp_path / "snapshot" / split
                path.mkdir(parents=True, exist_ok=True)
                return path

            monkeypatch.setattr(
                "src.training.ray_train.download_split", fake_download_split
            )

        with when("the data directories are resolved"):
            train_dir, val_dir = resolve_data_dirs(config)

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

    def test_limit_applies_to_train_only(self, monkeypatch, tmp_path):
        with given("a config with a 2 GB data limit"):
            config = make_config(data_repo="user/chesspos", data_limit_gb=2)
            downloads: list[tuple] = []

            def fake_download_split(repo_id, split, cache_dir=None, max_bytes=None):
                downloads.append((split, max_bytes))
                path = tmp_path / "snapshot" / split
                path.mkdir(parents=True, exist_ok=True)
                return path

            monkeypatch.setattr(
                "src.training.ray_train.download_split", fake_download_split
            )

        with when("the data directories are resolved"):
            resolve_data_dirs(config)

        with then(
            "the train split is bounded by 2 GB in bytes and the test split is unbounded"
        ):
            assert downloads == [
                ("train", 2 * 1024**3),
                ("test", None),
            ]
        return

    def test_cache_dir_is_forwarded(self, monkeypatch, tmp_path):
        with given("a config with a hub repo and an explicit cache directory"):
            config = make_config(
                data_repo="user/chesspos", data_cache=str(tmp_path / "c")
            )
            seen: list[str | None] = []

            def fake_download_split(repo_id, split, cache_dir=None, max_bytes=None):
                seen.append(cache_dir)
                path = tmp_path / "snapshot" / split
                path.mkdir(parents=True, exist_ok=True)
                return path

            monkeypatch.setattr(
                "src.training.ray_train.download_split", fake_download_split
            )

        with when("the data directories are resolved"):
            resolve_data_dirs(config)

        with then("the cache directory is passed to both split downloads"):
            assert seen == [str(tmp_path / "c"), str(tmp_path / "c")]
        return

    def test_repo_ignores_local_dirs(self, monkeypatch, tmp_path):
        with given("a config with both a hub repo and local directories"):
            config = make_config(
                data_repo="user/chesspos",
                train_dir="/nonexistent/train",
                val_dir="/nonexistent/test",
            )

            def fake_download_split(repo_id, split, cache_dir=None, max_bytes=None):
                path = tmp_path / "snapshot" / split
                path.mkdir(parents=True, exist_ok=True)
                return path

            monkeypatch.setattr(
                "src.training.ray_train.download_split", fake_download_split
            )

        with when("the data directories are resolved"):
            train_dir, val_dir = resolve_data_dirs(config)

        with then("the hub splits win and the local dirs are ignored"):
            assert train_dir == tmp_path / "snapshot" / "train"
            assert val_dir == tmp_path / "snapshot" / "test"
        return

    def test_local_dirs_without_repo(self, monkeypatch, tmp_path):
        with given("a config with local directories and no hub repo"):
            train_path = tmp_path / "train"
            val_path = tmp_path / "test"
            train_path.mkdir()
            val_path.mkdir()
            config = make_config(train_dir=str(train_path), val_dir=str(val_path))

            def fail_download(*args, **kwargs):
                raise AssertionError("download_split must not be called")

            monkeypatch.setattr("src.training.ray_train.download_split", fail_download)

        with when("the data directories are resolved"):
            resolved_train, resolved_val = resolve_data_dirs(config)

        with then("the local directories are returned unchanged"):
            assert resolved_train == train_path
            assert resolved_val == val_path
        return

    def test_missing_data_source_raises(self):
        with given("a config with neither a hub repo nor local directories"):
            config = make_config()

        with when("the data directories are resolved"):
            try:
                resolve_data_dirs(config)
                raised = False
            except ValueError:
                raised = True

        with then("a ValueError is raised"):
            assert raised
        return

    def test_nonexistent_local_dirs_raise(self):
        with given("a config with local directories that do not exist"):
            config = make_config(
                train_dir="/nonexistent/train", val_dir="/nonexistent/test"
            )

        with when("the data directories are resolved"):
            try:
                resolve_data_dirs(config)
                raised = False
            except ValueError:
                raised = True

        with then("a ValueError is raised"):
            assert raised
        return
