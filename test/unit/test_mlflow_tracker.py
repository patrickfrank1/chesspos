import pytest

from bdd import given, then, when
from src.training.mlflow_tracker import MLflowTracker, resolve_tracking_uri


@pytest.fixture
def fake_mlflow(monkeypatch):
    """Stub the mlflow module inside mlflow_tracker; records calls."""
    import types

    calls: list[tuple] = []

    module = types.ModuleType("mlflow")
    module.log_params = lambda params: calls.append(("log_params", params))
    module.log_metrics = lambda metrics, step=None: calls.append(
        ("log_metrics", metrics, step)
    )
    module.log_artifact = lambda local_path: calls.append(
        ("log_artifact", str(local_path))
    )
    module.end_run = lambda: calls.append(("end_run",))
    module.set_tracking_uri = lambda uri: calls.append(("set_tracking_uri", uri))
    module.set_experiment = lambda name: calls.append(("set_experiment", name))
    module.start_run = lambda **kwargs: calls.append(("start_run", kwargs))
    monkeypatch.setattr("src.training.mlflow_tracker.mlflow", module, raising=False)
    return calls


class TestResolveTrackingUri:
    def test_flag_takes_precedence(self, monkeypatch):
        with given("both a CLI flag and a MLFLOW_TRACKING_URI env var"):
            monkeypatch.setenv("MLFLOW_TRACKING_URI", "http://env:5000")

        with when("the tracking URI is resolved"):
            uri = resolve_tracking_uri("http://flag:5000")

        with then("the flag wins"):
            assert uri == "http://flag:5000"
        return

    def test_env_fallback(self, monkeypatch):
        with given("no CLI flag but a MLFLOW_TRACKING_URI env var"):
            monkeypatch.setenv("MLFLOW_TRACKING_URI", "http://env:5000")

        with when("the tracking URI is resolved"):
            uri = resolve_tracking_uri(None)

        with then("the env var is used"):
            assert uri == "http://env:5000"
        return

    def test_none_when_unset(self, monkeypatch):
        with given("neither a CLI flag nor an env var"):
            monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)

        with when("the tracking URI is resolved"):
            uri = resolve_tracking_uri(None)

        with then("tracking stays disabled"):
            assert uri is None
        return


class TestMLflowTracker:
    def test_disabled_tracker_is_a_no_op(self):
        with given("a disabled tracker"):
            tracker = MLflowTracker(enabled=False)

        with when("all tracking methods are called"):
            tracker.log_params({"a": 1})
            tracker.log_metrics({"loss": 1.0}, step=1)
            tracker.log_artifact("/tmp/x.pt")
            tracker.end()

        with then("no error is raised and nothing is tracked"):
            assert tracker.enabled is False
        return

    def test_start_run_sets_uri_experiment_and_name(self, fake_mlflow):
        with given("an enabled tracker with explicit settings"):
            pass

        with when("the tracker is created and started"):
            MLflowTracker(
                enabled=True,
                tracking_uri="http://flag:5000",
                experiment="chesspos",
                run_name="smoke",
            )

        with then("mlflow tracking uri, experiment and run are configured"):
            assert ("set_tracking_uri", "http://flag:5000") in fake_mlflow
            assert ("set_experiment", "chesspos") in fake_mlflow
            start = [entry for entry in fake_mlflow if entry[0] == "start_run"]
            assert start and start[0][1] == {"run_name": "smoke"}
        return

    def test_logs_params_metrics_and_artifacts(self, fake_mlflow):
        with given("an enabled tracker"):
            tracker = MLflowTracker(enabled=True)

        with when("params, metrics and an artifact are logged"):
            tracker.log_params({"steps": 100})
            tracker.log_metrics({"loss": 6.7, "lr": 1e-4}, step=10)
            tracker.log_artifact("/tmp/checkpoint.pt")

        with then("each call is forwarded to mlflow with the step attached"):
            assert ("log_params", {"steps": 100}) in fake_mlflow
            assert ("log_metrics", {"loss": 6.7, "lr": 1e-4}, 10) in fake_mlflow
            assert ("log_artifact", "/tmp/checkpoint.pt") in fake_mlflow
        return

    def test_end_closes_the_run(self, fake_mlflow):
        with given("an enabled tracker with an open run"):
            tracker = MLflowTracker(enabled=True)

        with when("the run is ended"):
            tracker.end()

        with then("mlflow.end_run is called exactly once"):
            assert fake_mlflow.count(("end_run",)) == 1
        return
