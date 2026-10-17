"""Parallel ranged downloader for remote PGN archives.

Downloads a URL to a local file using multiple concurrent HTTP range
requests (the archive CDN supports ``Accept-Ranges: bytes``), writing each
range directly into its offset in a preallocated file. Servers without
range support fall back to a single full-body stream.
"""

import concurrent.futures
import http.client
import os
import urllib.request
from pathlib import Path

_ATTEMPTS_PER_RANGE = 3
_READ_SIZE = 1024 * 1024


def _probe(url: str) -> tuple[int | None, bool]:
    """Return (total_size, supports_ranges) for the URL."""
    request = urllib.request.Request(url, headers={"Range": "bytes=0-0"}, method="HEAD")
    with urllib.request.urlopen(request) as response:
        status = response.status
        if status == 206:
            content_range = response.headers.get("Content-Range", "")
            total = int(content_range.rsplit("/", 1)[-1])
            return total, True
        content_length = response.headers.get("Content-Length")
        total = int(content_length) if content_length else None
        return total, False


def _split_ranges(total: int, num_connections: int) -> list[tuple[int, int]]:
    """Split [0, total) into contiguous, non-overlapping ranges."""
    num_connections = max(1, min(num_connections, total))
    base, remainder = divmod(total, num_connections)
    ranges = []
    start = 0
    for i in range(num_connections):
        size = base + (1 if i < remainder else 0)
        ranges.append((start, start + size - 1))
        start += size
    return ranges


def _download_range(
    url: str, start: int, end: int, fd: int, attempts: int = _ATTEMPTS_PER_RANGE
) -> None:
    """Download bytes=[start,end] and pwrite them at their file offset."""
    last_error: Exception | None = None
    for _ in range(attempts):
        try:
            request = urllib.request.Request(
                url, headers={"Range": f"bytes={start}-{end}"}
            )
            with urllib.request.urlopen(request) as response:
                offset = start
                while True:
                    chunk = response.read(_READ_SIZE)
                    if not chunk:
                        break
                    os.pwrite(fd, chunk, offset)
                    offset += len(chunk)
            if offset == end + 1:
                return
            last_error = http.client.IncompleteRead(
                f"incomplete range {start}-{end}: got {offset - start} of {end - start + 1}"
            )
        except (OSError, http.client.HTTPException) as exc:
            last_error = exc
    raise last_error


def _download_single_stream(url: str, dest: Path) -> None:
    with urllib.request.urlopen(url) as response, open(dest, "wb") as f:
        while True:
            chunk = response.read(_READ_SIZE)
            if not chunk:
                break
            f.write(chunk)
    return


def download_ranged(url: str, dest: Path, num_connections: int = 8) -> Path:
    """Download ``url`` to ``dest`` using parallel range requests.

    Falls back to a single stream when the server does not advertise
    range support. Existing files at ``dest`` are overwritten.
    """
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    total, supports_ranges = _probe(url)
    if not supports_ranges or total is None:
        _download_single_stream(url, dest)
        return dest
    ranges = _split_ranges(total, num_connections)
    with open(dest, "wb") as f:
        f.truncate(total)
    fd = os.open(dest, os.O_WRONLY)
    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=len(ranges)) as pool:
            futures = [
                pool.submit(_download_range, url, start, end, fd)
                for start, end in ranges
            ]
            for future in futures:
                future.result()
    finally:
        os.close(fd)
    return dest
