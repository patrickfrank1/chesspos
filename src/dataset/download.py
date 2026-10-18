"""Simple single-stream file downloader used for archive prefetching."""

import shutil
import time
import urllib.request
from pathlib import Path

_ATTEMPTS = 3


def download_file(url: str, dest: Path, attempts: int = _ATTEMPTS) -> Path:
    """Stream ``url`` to ``dest`` with a few retries on transient errors."""
    last_error: Exception | None = None
    for attempt in range(attempts):
        try:
            with urllib.request.urlopen(url) as response, open(dest, "wb") as f:
                shutil.copyfileobj(response, f)
            return dest
        except OSError as exc:
            last_error = exc
            time.sleep(2**attempt)
    raise last_error
