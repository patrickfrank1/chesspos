from bdd import given, then, when
from http_support import RangedHttpServer
from src.dataset.download import download_ranged

PAYLOAD = bytes(range(256)) * 160  # 40,960 bytes


def _assert_ranges_tile_payload(ranges: list[tuple[int, int]]) -> None:
    assert sorted(ranges) == sorted(set(ranges))
    covered = 0
    for start, end in sorted(ranges):
        assert start == covered
        covered = end + 1
    assert covered == len(PAYLOAD)


class TestDownloadRanged:
    def test_single_connection_downloads_exact_bytes(self, tmp_path):
        with given("a ranged HTTP server and a single-connection downloader"):
            server = RangedHttpServer(PAYLOAD)
            dest = tmp_path / "a.pgn.zst"

        with when("the file is downloaded"):
            with server:
                download_ranged(server.url, dest, num_connections=1)

        with then("the file matches the served bytes"):
            assert dest.read_bytes() == PAYLOAD
        return

    def test_parallel_connections_partition_the_file(self, tmp_path):
        with given("a ranged HTTP server and a 4-connection downloader"):
            server = RangedHttpServer(PAYLOAD)
            dest = tmp_path / "a.pgn.zst"

        with when("the file is downloaded with 4 parallel connections"):
            with server:
                download_ranged(server.url, dest, num_connections=4)

        with then("the file matches and the range requests tile the payload"):
            assert dest.read_bytes() == PAYLOAD
            _assert_ranges_tile_payload(server.range_requests)
        return

    def test_failed_range_request_is_retried(self, tmp_path):
        with given(
            "a server that drops the connection on the first request for one range"
        ):
            server = RangedHttpServer(PAYLOAD)
            server.truncate_once(len(PAYLOAD) // 2)
            dest = tmp_path / "a.pgn.zst"

        with when("the file is downloaded with 2 parallel connections"):
            with server:
                download_ranged(server.url, dest, num_connections=2)

        with then("the dropped range is retried and the file is complete"):
            assert dest.read_bytes() == PAYLOAD
            dropped = [r for r in server.range_requests if r[0] == len(PAYLOAD) // 2]
            assert len(dropped) >= 2
        return

    def test_server_without_range_support_falls_back_to_single_stream(self, tmp_path):
        with given("a server that ignores Range headers"):
            server = RangedHttpServer(PAYLOAD, support_ranges=False)
            dest = tmp_path / "a.pgn.zst"

        with when("the file is downloaded"):
            with server:
                download_ranged(server.url, dest, num_connections=8)

        with then("the file is downloaded in one full-body request"):
            assert dest.read_bytes() == PAYLOAD
            assert server.range_requests == []
        return
