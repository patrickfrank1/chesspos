import pytest

from bdd import given, then, when
from src.training.preflight import check_dependencies


class TestCheckDependencies:
    def test_importable_modules_pass(self):
        with given("a list of importable standard-library modules"):
            modules = ["json", "math"]

        with when("the dependency check runs"):
            result = check_dependencies(modules)

        with then("the check passes"):
            assert result is None
        return

    def test_missing_module_is_reported(self):
        with given("a dependency that cannot be imported"):
            modules = ["missing_dependency_xyz"]

        with when("the dependency check runs"):
            with pytest.raises(RuntimeError) as excinfo:
                check_dependencies(modules)

        with then("the error names the missing module"):
            assert "missing_dependency_xyz" in str(excinfo.value)
        return

    def test_all_missing_modules_are_reported(self):
        with given("two dependencies that cannot be imported"):
            modules = ["missing_dependency_xyz", "another_missing_dependency"]

        with when("the dependency check runs"):
            with pytest.raises(RuntimeError) as excinfo:
                check_dependencies(modules)

        with then("the error names both modules"):
            message = str(excinfo.value)
            assert "missing_dependency_xyz" in message
            assert "another_missing_dependency" in message
        return


class TestTrainingPreflight:
    def test_main_checks_dependencies_before_logging(self, monkeypatch):
        import src.run.train_masked_transformer as trainer
        from types import SimpleNamespace

        with given("a dependency check that fails immediately"):
            monkeypatch.setattr(
                trainer, "parse_args", lambda: SimpleNamespace(log_file=None)
            )
            calls: list[str] = []

            def fail_check() -> None:
                calls.append("check")
                raise RuntimeError("missing dependency")

            def record_logger(path: str | None):
                calls.append("logger")
                return lambda message: None

            monkeypatch.setattr(trainer, "check_dependencies", fail_check)
            monkeypatch.setattr(trainer, "make_logger", record_logger)

        with when("the training entry point starts"):
            with pytest.raises(RuntimeError, match="missing dependency"):
                trainer.main()

        with then("the dependency check runs before the logger is opened"):
            assert calls == ["check"]
        return
