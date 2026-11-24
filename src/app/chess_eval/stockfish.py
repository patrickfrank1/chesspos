from __future__ import annotations

import hashlib
import shutil
import threading
import urllib.request
from pathlib import Path

from src.app.chess_eval.backend import REPO_ROOT

STOCKFISH_NAME = "Stockfish 19"
STOCKFISH_WEB_VERSION = "0.6.1"
STOCKFISH_NNUE_FILENAME = "nn-1a298aa575a0.nnue"
STOCKFISH_NNUE_URL = (
    f"https://tests.stockfishchess.org/api/nn/{STOCKFISH_NNUE_FILENAME}"
)
DEFAULT_NNUE_CACHE_DIR = REPO_ROOT / "data/models/stockfish"

_NNUE_LOCK = threading.Lock()


def ensure_nnue(cache_dir: str | Path | None = None) -> Path:
    """Return the cached Stockfish 19 NNUE, downloading it if necessary."""
    resolved = Path(cache_dir) if cache_dir is not None else DEFAULT_NNUE_CACHE_DIR
    path = resolved / STOCKFISH_NNUE_FILENAME
    if path.exists() and path.stat().st_size > 0:
        return path

    with _NNUE_LOCK:
        if path.exists() and path.stat().st_size > 0:
            return path
        resolved.mkdir(parents=True, exist_ok=True)
        part = path.with_name(path.name + ".part")
        request = urllib.request.Request(
            STOCKFISH_NNUE_URL,
            headers={"User-Agent": "chesspos-eval/0.1"},
        )
        try:
            with urllib.request.urlopen(request, timeout=300) as response:
                with part.open("wb") as handle:
                    shutil.copyfileobj(response, handle)
            digest = hashlib.sha256(part.read_bytes()).hexdigest()[:12]
            expected = STOCKFISH_NNUE_FILENAME[3:15]
            if digest != expected:
                raise RuntimeError(
                    f"NNUE checksum mismatch: expected {expected}, got {digest}"
                )
            part.replace(path)
        except Exception:
            part.unlink(missing_ok=True)
            raise
    return path


def nnue_info(cache_dir: str | Path | None = None) -> dict:
    resolved = Path(cache_dir) if cache_dir is not None else DEFAULT_NNUE_CACHE_DIR
    path = resolved / STOCKFISH_NNUE_FILENAME
    return {
        "engine": STOCKFISH_NAME,
        "stockfish_web": STOCKFISH_WEB_VERSION,
        "nnue": STOCKFISH_NNUE_FILENAME,
        "nnue_url": STOCKFISH_NNUE_URL,
        "cached": path.exists() and path.stat().st_size > 0,
    }
