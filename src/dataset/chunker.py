"""Game-aligned chunking of PGN byte streams.

Archives are decompressed as streams and cut into chunks of roughly
``chunk_target_bytes`` at game boundaries. Chunks are yielded lazily as
they are completed, so downstream processing can start while the rest of
the archive is still streaming in (bounded scratch usage).

Game boundary rule: a new game starts at the byte sequence
``\\n\\n[Event`` (blank line followed by the first mandatory header).
Cutting only at these boundaries keeps every chunk parseable and makes
``b"".join(chunks) == original stream`` hold exactly.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import BinaryIO

SEPARATOR = b"\n\n[Event"
_MIN_READ_BLOCK = 256
# Read block scales with the chunk target (1/1024 of it) so large archives
# do not pay per-byte read overhead, but stays small enough that the first
# chunk of a small stream is yielded before the stream is fully consumed.
_READ_BLOCK_RATIO = 1024


class PgnChunker:
    """Split a PGN byte stream into game-aligned chunk files."""

    def __init__(self, chunk_target_bytes: int):
        if chunk_target_bytes <= 0:
            raise ValueError("chunk_target_bytes must be positive")
        self.chunk_target_bytes = chunk_target_bytes

    @property
    def read_block_bytes(self) -> int:
        return max(_MIN_READ_BLOCK, self.chunk_target_bytes // _READ_BLOCK_RATIO)

    def iter_chunks(self, stream: BinaryIO, out_dir: str) -> Iterator[Path]:
        """Write game-aligned chunks to ``out_dir`` and yield their paths.

        A chunk is emitted as soon as it reaches ``chunk_target_bytes``
        and the next game boundary is seen; the final partial chunk is
        emitted at end of stream. An oversized single game becomes one
        complete oversized chunk.
        """
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        index = 0
        buf = bytearray()
        block = self.read_block_bytes
        while True:
            data = stream.read(block)
            if not data:
                break
            buf.extend(data)
            # Cut at every boundary that closes a >= target chunk.
            search_from = 0
            while True:
                pos = buf.find(SEPARATOR, search_from)
                if pos == -1:
                    break
                search_from = pos + 1
                cut = pos + 2  # keep the blank line with the previous chunk
                if cut >= self.chunk_target_bytes:
                    path = out / f"chunk_{index:04d}.pgn"
                    path.write_bytes(bytes(buf[:cut]))
                    yield path
                    index += 1
                    del buf[:cut]
                    search_from = 0
        if buf:
            path = out / f"chunk_{index:04d}.pgn"
            path.write_bytes(bytes(buf))
            yield path


def decompress_zst(path: str | Path) -> Iterator[bytes]:
    """Stream-decompress a ``.pgn.zst`` archive, yielding byte blocks."""
    import zstandard

    dctx = zstandard.ZstdDecompressor()
    with open(path, "rb") as fh:
        reader = dctx.stream_reader(fh)
        while True:
            block = reader.read(_MIN_READ_BLOCK * 1024)
            if not block:
                break
            yield block
