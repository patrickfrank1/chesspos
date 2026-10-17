"""Threaded local HTTP server that mimics the Lichess archive CDN.

Serves one fixed byte payload. Supports HTTP range requests (``bytes=a-b``),
can be configured to ignore them (to test single-stream fallback), and can
truncate the body of the first request for a given range start (to test
per-range retry).
"""

import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


class RangedHttpServer:
    """Serve a fixed payload on a random local port, with Range support."""

    SERVED_PATH = "/test-archive.pgn.zst"

    def __init__(self, payload: bytes, support_ranges: bool = True):
        self.payload = payload
        self.support_ranges = support_ranges
        self.range_requests: list[tuple[int, int]] = []
        self.truncate_once_starts: set[int] = set()
        self._server = None
        self._thread = None

    @property
    def url(self) -> str:
        port = self._server.server_address[1]
        return f"http://127.0.0.1:{port}{self.SERVED_PATH}"

    def truncate_once(self, range_start: int) -> None:
        """Drop the connection mid-body the next time this range is requested."""
        self.truncate_once_starts.add(range_start)
        return

    def start(self) -> None:
        server = self

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def _serve(self, want_body: bool, record: bool = True) -> None:
                range_header = self.headers.get("Range")
                rng = None
                if range_header and range_header.startswith("bytes="):
                    try:
                        start_s, end_s = range_header[len("bytes=") :].split("-", 1)
                        rng = (int(start_s), int(end_s))
                    except ValueError:
                        rng = None
                if rng is not None and server.support_ranges:
                    start, end = rng
                    end = min(end, len(server.payload) - 1)
                    if record:
                        server.range_requests.append((start, end))
                    body = server.payload[start : end + 1]
                    self.send_response(206)
                    self.send_header(
                        "Content-Range", f"bytes {start}-{end}/{len(server.payload)}"
                    )
                else:
                    start = 0
                    body = server.payload
                    self.send_response(200)
                if server.support_ranges:
                    self.send_header("Accept-Ranges", "bytes")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                if not want_body:
                    return
                if start in server.truncate_once_starts:
                    server.truncate_once_starts.discard(start)
                    self.wfile.write(body[: len(body) // 2])
                    self.close_connection = True
                    return
                self.wfile.write(body)
                return

            def do_GET(self) -> None:
                if self.path != server.SERVED_PATH:
                    self.send_response(404)
                    self.send_header("Content-Length", "0")
                    self.end_headers()
                    return
                self._serve(want_body=True)
                return

            def do_HEAD(self) -> None:
                self._serve(want_body=False, record=False)
                return

            def log_message(self, format: str, *args) -> None:
                return

        self._server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()
        return

    def stop(self) -> None:
        if self._server is not None:
            self._server.shutdown()
            self._server.server_close()
            self._thread.join(timeout=5)
        return

    def __enter__(self) -> "RangedHttpServer":
        self.start()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.stop()
        return
