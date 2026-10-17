import io

import chess.pgn
import pytest

from bdd import given, then, when
from etl_support import SAMPLE_PGN, make_lichess_pgn
from src.dataset.chunker import PgnChunker

GAME_A = SAMPLE_PGN
GAME_B = make_lichess_pgn("Game B", date="2023.01.16")
GAME_C = make_lichess_pgn("Game C", date="2023.01.17")


def read_games_from_file(path):
    games = []
    with open(path, "rb") as f:
        while True:
            game = chess.pgn.read_game(f)
            if game is None:
                break
            games.append(game)
    return games


class TestPgnChunker:
    def test_split_preserves_all_games_and_bytes(self):
        with given("a PGN stream of three games and a small chunk target"):
            stream = io.BytesIO(GAME_A + GAME_B + GAME_C)
            chunker = PgnChunker(chunk_target_bytes=600)

        with when("the stream is split into game-aligned chunks"):
            import tempfile
            from pathlib import Path

            with tempfile.TemporaryDirectory() as tmpdir:
                chunk_paths = list(chunker.iter_chunks(stream, out_dir=tmpdir))
                chunk_bytes = [Path(p).read_bytes() for p in chunk_paths]

        with then("every chunk parses cleanly and no bytes are lost"):
            assert len(chunk_paths) >= 2
            games = [g for p in chunk_paths for g in read_games_from_file(p)]
            assert len(games) == 3
            assert b"".join(chunk_bytes) == GAME_A + GAME_B + GAME_C

    def test_no_game_spans_two_chunks(self):
        with given("a PGN stream of three games and a small chunk target"):
            stream = io.BytesIO(GAME_A + GAME_B + GAME_C)
            chunker = PgnChunker(chunk_target_bytes=600)

        with when("the stream is split"):
            import tempfile

            with tempfile.TemporaryDirectory() as tmpdir:
                chunk_paths = list(chunker.iter_chunks(stream, out_dir=tmpdir))

        with then("each chunk contains only whole games"):
            for path in chunk_paths:
                games = read_games_from_file(path)
                assert 1 <= len(games) <= 3
                for game in games:
                    assert game.headers["Event"] in {"Test", "Game B", "Game C"}

    def test_split_is_deterministic(self):
        with given("the same PGN stream split twice"):
            import tempfile

            chunker = PgnChunker(chunk_target_bytes=600)
            runs = []
            for _ in range(2):
                with tempfile.TemporaryDirectory() as tmpdir:
                    paths = list(
                        chunker.iter_chunks(
                            io.BytesIO(GAME_A + GAME_B + GAME_C), out_dir=tmpdir
                        )
                    )
                    runs.append([open(p, "rb").read() for p in paths])

        with when("the two runs are compared"):
            first, second = runs

        with then("chunk boundaries and contents are identical"):
            assert first == second

    def test_oversized_game_gets_own_chunk(self):
        with given("a single game larger than the chunk target"):
            big_game = GAME_A * 20
            stream = io.BytesIO(big_game)
            chunker = PgnChunker(chunk_target_bytes=100)

        with when("the stream is split"):
            import tempfile

            with tempfile.TemporaryDirectory() as tmpdir:
                chunk_paths = list(chunker.iter_chunks(stream, out_dir=tmpdir))

        with then("the game is emitted as one complete oversized chunk"):
            assert len(chunk_paths) == 1
            games = read_games_from_file(chunk_paths[0])
            assert len(games) == 1

    def test_chunks_are_yielded_lazily(self):
        with given("a stream that tracks how much has been consumed"):
            payload = GAME_A + GAME_B + GAME_C

            class TrackingStream(io.BytesIO):
                consumed_while_yielding = []

                def read(self, *args):
                    return super().read(*args)

            stream = TrackingStream(payload)
            chunker = PgnChunker(chunk_target_bytes=600)

        with when("chunks are consumed one at a time"):
            import tempfile

            with tempfile.TemporaryDirectory() as tmpdir:
                iterator = chunker.iter_chunks(stream, out_dir=tmpdir)
                first_chunk = next(iterator)
                stream.consumed_while_yielding.append(stream.tell())
                remaining = list(iterator)

        with then("the first chunk is available before the stream is fully consumed"):
            assert first_chunk is not None
            assert len(remaining) >= 1
            assert stream.consumed_while_yielding[0] < len(payload)


class TestZstdDecompression:
    def test_decompress_zst_roundtrip(self):
        with given("a zstd-compressed PGN file"):
            zstandard = pytest.importorskip("zstandard")
            import tempfile
            from pathlib import Path

            payload = GAME_A + GAME_B
            with tempfile.TemporaryDirectory() as tmpdir:
                zst_path = Path(tmpdir) / "archive.pgn.zst"
                cctx = zstandard.ZstdCompressor()
                zst_path.write_bytes(cctx.compress(payload))

        with when("the archive is streamed back through decompression"):
            from src.dataset.chunker import decompress_zst

            decompressed = b"".join(decompress_zst(zst_path))

        with then("the original PGN bytes are recovered"):
            assert decompressed == payload
