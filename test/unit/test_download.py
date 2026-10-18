from io import BytesIO
from unittest.mock import patch

from bdd import given, then, when
from src.dataset.download import download_file

PAYLOAD = b"fake-pgn-zst-bytes"


class TestDownloadFile:
    def test_downloads_response_body_to_disk(self, tmp_path):
        with given("a URL whose response body is the archive"):
            dest = tmp_path / "a.pgn.zst"

        with when("the file is downloaded"):
            with patch(
                "src.dataset.download.urllib.request.urlopen",
                return_value=BytesIO(PAYLOAD),
            ):
                download_file("https://x/a.pgn.zst", dest)

        with then("the response body is written to the destination"):
            assert dest.read_bytes() == PAYLOAD
        return

    def test_retries_on_transient_failure(self, tmp_path):
        with given("a URL that fails once then serves the archive"):
            dest = tmp_path / "a.pgn.zst"
            calls = {"n": 0}

            class FlakyResp:
                def read(self, n=-1):
                    return b""

            def flaky_urlopen(url):
                calls["n"] += 1
                if calls["n"] == 1:
                    raise OSError("connection reset")
                import io

                return io.BytesIO(PAYLOAD)

        with when("the file is downloaded"):
            with patch(
                "src.dataset.download.urllib.request.urlopen",
                side_effect=flaky_urlopen,
            ):
                with patch("src.dataset.download.time.sleep"):
                    download_file("https://x/a.pgn.zst", dest)

        with then("the retry succeeds and the file is complete"):
            assert dest.read_bytes() == PAYLOAD
            assert calls["n"] == 2
        return
