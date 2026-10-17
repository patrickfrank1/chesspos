"""Streaming download of remote PGN archives."""

from __future__ import annotations

import shutil
import urllib.request
from pathlib import Path


def download_archive(url: str, dest_dir: str) -> Path:
    """Download ``url`` into ``dest_dir`` and return the local file path."""
    dest = Path(dest_dir) / url.rsplit("/", 1)[-1]
    with urllib.request.urlopen(url) as response, open(dest, "wb") as fh:
        shutil.copyfileobj(response, fh)
    return dest
