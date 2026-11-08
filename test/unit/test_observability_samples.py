import chess
import pyarrow as pa
import pyarrow.parquet as pq

from bdd import given, then, when
from src.dataset.token_stream import TokenStreamEncoder, pack_stream
from src.evaluation.observability.samples import (
    load_puzzle_samples,
    load_test_samples,
    make_sample,
)


def write_test_parquet(path, rows):
    table = pa.table(
        {
            "packed": pa.array([row["packed"] for row in rows], pa.list_(pa.int16())),
            "n_positions": pa.array([row["n_positions"] for row in rows], pa.int32()),
            "game_id": pa.array([row["game_id"] for row in rows], pa.string()),
        }
    )
    pq.write_table(table, path)
    return


def write_puzzle_parquet(path, rows, themes):
    columns = {
        "puzzle_id": pa.array([row["puzzle_id"] for row in rows], pa.string()),
        "packed": pa.array([row["packed"] for row in rows], pa.list_(pa.int16())),
        "rating": pa.array([row["rating"] for row in rows], pa.int32()),
    }
    for theme in themes:
        columns[theme] = pa.array(
            [row["themes"].get(theme, 0) for row in rows], pa.int8()
        )
    pq.write_table(pa.table(columns), path)
    return


def test_make_sample_preserves_board_and_surrogates():
    with given("the starting chess position"):
        board = chess.Board()
        tokens = TokenStreamEncoder().encode(board)

    with when("a position sample is created from its tokens"):
        sample = make_sample(tokens, "test", game_id="g1", ply=0)

    with then("the FEN and structural surrogates are correct"):
        assert sample.fen == board.fen()
        assert sample.surrogates["material"] == 0
        assert sample.surrogates["piece_count"] == 32
        assert sample.surrogates["open_files"] == 0
        assert sample.surrogates["isolated_pawns"] == 0
        assert sample.surrogates["doubled_pawns"] == 0
        assert sample.surrogates["bishop_pair"] == 1
        assert sample.surrogates["mobility"] == 20
    return


def test_load_test_samples_preserves_provenance_and_caps_games(tmp_path):
    with given("a tiny test shard with two games"):
        encoder = TokenStreamEncoder()
        board = chess.Board()
        first = encoder.encode(board)
        board.push_san("e4")
        second = encoder.encode(board)
        rows = [
            {
                "packed": pack_stream([first, second]).tolist(),
                "n_positions": 2,
                "game_id": "game-a",
            },
            {
                "packed": pack_stream([first, second]).tolist(),
                "n_positions": 2,
                "game_id": "game-b",
            },
        ]
        path = tmp_path / "test.parquet"
        write_test_parquet(path, rows)

    with when("two samples are loaded with a per-game cap of one"):
        samples = load_test_samples(tmp_path, n=2, per_game_cap=1, seed=0)

    with then("each game contributes exactly one position and provenance is kept"):
        assert len(samples) == 2
        assert {sample.game_id for sample in samples} == {"game-a", "game-b"}
        assert all(sample.source == "test" for sample in samples)
        assert all(sample.ply is not None for sample in samples)
        assert all(sample.fen for sample in samples)
        assert [sample.sample_id for sample in samples] == [0, 1]
    return


def test_load_test_samples_is_deterministic(tmp_path):
    with given("a tiny test shard"):
        encoder = TokenStreamEncoder()
        board = chess.Board()
        rows = [
            {
                "packed": pack_stream([encoder.encode(board)]).tolist(),
                "n_positions": 1,
                "game_id": "game-a",
            }
        ]
        path = tmp_path / "test.parquet"
        write_test_parquet(path, rows)

    with when("the same load is repeated"):
        first = load_test_samples(tmp_path, n=1, per_game_cap=1, seed=7)
        second = load_test_samples(tmp_path, n=1, per_game_cap=1, seed=7)

    with then("the sampled positions are identical"):
        assert [sample.tokens for sample in first] == [
            sample.tokens for sample in second
        ]
    return


def test_load_puzzle_samples_uses_first_position_and_filters_themes(tmp_path):
    with given("a puzzle shard with fork and pin puzzles"):
        encoder = TokenStreamEncoder()
        board = chess.Board()
        first = encoder.encode(board)
        board.push_san("e4")
        second = encoder.encode(board)
        rows = [
            {
                "puzzle_id": "p-fork",
                "packed": pack_stream([first, second]).tolist(),
                "rating": 1200,
                "themes": {"fork": 1, "pin": 0},
            },
            {
                "puzzle_id": "p-pin",
                "packed": pack_stream([first, second]).tolist(),
                "rating": 1300,
                "themes": {"fork": 0, "pin": 1},
            },
        ]
        path = tmp_path / "puzzles.parquet"
        write_puzzle_parquet(path, rows, ("fork", "pin"))

    with when("fork puzzles are loaded and the first position is requested"):
        samples = load_puzzle_samples(
            tmp_path, n=10, themes=("fork",), position="first", seed=0
        )

    with then("only the fork puzzle is returned at its first packed segment"):
        assert len(samples) == 1
        sample = samples[0]
        assert sample.puzzle_id == "p-fork"
        assert sample.themes == ("fork",)
        assert tuple(sample.tokens) == tuple(int(token) for token in first)
        assert sample.source == "puzzle"
        assert sample.rating == 1200
    return


def test_load_puzzle_samples_balances_requested_themes(tmp_path):
    with given("a puzzle shard with overlapping multi-label themes"):
        encoder = TokenStreamEncoder()
        board = chess.Board()
        segment = encoder.encode(board)
        rows = [
            {
                "puzzle_id": f"p{index}",
                "packed": pack_stream([segment]).tolist(),
                "rating": 1000 + index,
                "themes": {"fork": 1, "pin": 1},
            }
            for index in range(4)
        ]
        path = tmp_path / "puzzles.parquet"
        write_puzzle_parquet(path, rows, ("fork", "pin"))

    with when("both themes are requested with a two-sample budget"):
        samples = load_puzzle_samples(
            tmp_path, n=2, themes=("fork", "pin"), position="first", seed=0
        )

    with then("both themes are represented when possible"):
        assert len(samples) == 2
        union = {theme for sample in samples for theme in sample.themes}
        assert union == {"fork", "pin"}
    return


def test_load_test_samples_selects_contiguous_positions(tmp_path):
    with given("a game with eight consecutive positions"):
        encoder = TokenStreamEncoder()
        board = chess.Board()
        segments = [encoder.encode(board)]
        for san in ("e4", "e5", "Nf3", "Nc6", "Bc4", "Nf6", "d3"):
            board.push_san(san)
            segments.append(encoder.encode(board))
        rows = [
            {
                "packed": pack_stream(segments).tolist(),
                "n_positions": len(segments),
                "game_id": "game-a",
            }
        ]
        path = tmp_path / "test.parquet"
        write_test_parquet(path, rows)

    with when("three positions are sampled from the game"):
        samples = load_test_samples(tmp_path, n=3, per_game_cap=3, seed=0)

    with then("the selected plies form one contiguous run"):
        assert len(samples) == 3
        plies = sorted(sample.ply for sample in samples)
        assert plies == list(range(plies[0], plies[0] + 3))
    return
