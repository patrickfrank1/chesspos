import torch

from bdd import given, then, when
from src.training.loop import (
    cpu_load_average,
    format_system_snapshot,
    gpu_snapshot,
    rss_gigabytes,
    system_snapshot,
)


class TestRssGigabytes:
    def test_returns_positive_value(self):
        with given("a running python process"):
            pass

        with when("the resident set size is sampled"):
            rss = rss_gigabytes()

        with then("the value is a positive number of sane magnitude"):
            assert 0.0 < rss < 1000.0
        return


class TestCpuLoadAverage:
    def test_returns_non_negative_value(self):
        with given("a running python process"):
            pass

        with when("the one-minute load average is sampled"):
            load = cpu_load_average()

        with then("the value is non-negative"):
            assert load >= 0.0
        return


class TestGpuSnapshot:
    def test_none_without_cuda(self, monkeypatch):
        with given("a machine without CUDA"):
            monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

        with when("a gpu snapshot is taken"):
            snapshot = gpu_snapshot()

        with then("no gpu metrics are reported"):
            assert snapshot is None
        return

    def test_reports_vram_and_utilization_when_available(self, monkeypatch):
        with given("a machine with mocked CUDA"):
            monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
            monkeypatch.setattr(
                torch.cuda, "mem_get_info", lambda: (2 * 1024**3, 8 * 1024**3)
            )
            monkeypatch.setattr(
                torch.cuda, "memory_allocated", lambda *a, **k: 1 * 1024**3
            )
            monkeypatch.setattr(torch.cuda, "utilization", lambda *a, **k: 97)

        with when("a gpu snapshot is taken"):
            snapshot = gpu_snapshot()

        with then("vram usage and gpu utilization are reported"):
            assert snapshot["vram_used_gb"] == 6.0
            assert snapshot["vram_alloc_gb"] == 1.0
            assert snapshot["vram_total_gb"] == 8.0
            assert snapshot["gpu_util_pct"] == 97.0
        return


class TestSystemSnapshot:
    def test_includes_cpu_metrics_and_gpu_when_present(self, monkeypatch):
        with given("a machine with mocked CUDA"):
            monkeypatch.setattr(
                "src.training.loop.gpu_snapshot",
                lambda: {"vram_used_gb": 6.0, "gpu_util_pct": 97.0},
            )

        with when("a full system snapshot is taken"):
            snapshot = system_snapshot()

        with then("cpu metrics and gpu metrics are merged"):
            assert snapshot["rss_gb"] > 0.0
            assert snapshot["cpu_load"] >= 0.0
            assert snapshot["vram_used_gb"] == 6.0
            assert snapshot["gpu_util_pct"] == 97.0
        return

    def test_excludes_gpu_keys_without_cuda(self, monkeypatch):
        with given("a machine without CUDA"):
            monkeypatch.setattr("src.training.loop.gpu_snapshot", lambda: None)

        with when("a full system snapshot is taken"):
            snapshot = system_snapshot()

        with then("only cpu metrics are reported"):
            assert "rss_gb" in snapshot
            assert "gpu_util_pct" not in snapshot
        return


class TestFormatSystemSnapshot:
    def test_formats_all_metrics_on_one_line(self):
        with given("a system snapshot with cpu and gpu metrics"):
            snapshot = {
                "rss_gb": 1.5,
                "cpu_load": 2.0,
                "vram_used_gb": 6.0,
                "gpu_util_pct": 97.0,
            }

        with when("the snapshot is formatted"):
            line = format_system_snapshot(snapshot)

        with then("every metric appears as key=value"):
            assert (
                line == "rss_gb=1.50 cpu_load=2.00 vram_used_gb=6.00 gpu_util_pct=97.00"
            )
        return
