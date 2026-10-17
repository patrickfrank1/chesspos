"""Shared helpers for ETL tests (unit and integration)."""

import chess
import pyarrow as pa

from src.dataset.token_stream import TokenStreamEncoder, pack_stream

SAMPLE_PGN = b"""[Event "Test"]
[White "Player1"]
[Black "Player2"]
[Result "1-0"]
[WhiteElo "2200"]
[BlackElo "2100"]

1. e4 e5 2. Nf3 Nc6 3. Bb5 a6 4. Ba4 Nf6 5. O-O Be7 6. Re1 b5 7. Bb3 d6 8. c3 O-O 9. h3 Na5 10. Bc2 c5 11. d4 Qc7 1-0

"""


LICHESS_PGN_TEMPLATE = """[Event "{event}"]
[Site "https://lichess.org/test"]
[Date "{date}"]
[UTCDate "{date}"]
[UTCTime "12:00:00"]
[White "White Player"]
[Black "Black Player"]
[Result "{result}"]
[WhiteElo "{white_elo}"]
[BlackElo "{black_elo}"]
[TimeControl "600+0"]

1. e4 e5 2. Nf3 Nc6 3. Bb5 a6 {result}"""


def make_lichess_pgn(
    event: str,
    date: str = "2023.01.15",
    white_elo: str = "2200",
    black_elo: str = "2100",
    result: str = "1-0",
) -> bytes:
    """One lichess-style game with UTCDate header, ending with a blank line."""
    return (
        LICHESS_PGN_TEMPLATE.format(
            event=event,
            date=date,
            result=result,
            white_elo=white_elo,
            black_elo=black_elo,
        )
        + "\n\n"
    ).encode("utf-8")


def make_batch() -> pa.Table:
    boards = [chess.Board(), chess.Board()]
    boards[1].push_san("e4")
    encoder = TokenStreamEncoder()
    segments = [encoder.encode(board) for board in boards]
    packed = pack_stream(segments, add_cls=True).tolist()
    return pa.table(
        {
            "packed": pa.array([packed], type=pa.list_(pa.int64())),
            "n_positions": pa.array([2], type=pa.int32()),
            "ply": pa.array([1], type=pa.int32()),
            "game_id": pa.array(["deadbeefdeadbeef"]),
            "white_elo": pa.array([2000], type=pa.int32()),
            "black_elo": pa.array([2000], type=pa.int32()),
            "result": pa.array(["1-0"]),
        }
    )


def make_batch_with_split(game_ids: list[str]) -> pa.Table:
    n = len(game_ids)
    batch = make_batch()
    columns = {}
    for name in batch.column_names:
        if name == "packed":
            continue
        columns[name] = batch.column(name).to_pylist() * n
    packed = batch.column("packed").to_pylist()[0]
    return pa.table(
        {
            **columns,
            "packed": pa.array([packed] * n, type=pa.list_(pa.int16())),
            "game_id": pa.array(game_ids, type=pa.string()),
            "split": pa.array(["train"] * n, type=pa.string()),
        }
    )
