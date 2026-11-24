"""Built-in example positions for the web UI."""

from __future__ import annotations

import io
from dataclasses import dataclass

import chess
import chess.pgn


@dataclass(frozen=True)
class Example:
    """A selectable query position."""

    label: str
    fen: str


def _from_pgn(pgn: str, *, ply: int | None = None) -> str:
    """Return the FEN after ``ply`` half-moves of a PGN."""
    game = chess.pgn.read_game(io.StringIO(pgn))
    if game is None:
        raise ValueError("could not parse PGN")
    board = game.board()
    for number, move in enumerate(game.mainline_moves(), start=1):
        if ply is not None and number > ply:
            break
        board.push(move)
    return board.fen()


EXAMPLES: tuple[Example, ...] = (
    Example(
        "Ruy Lopez middlegame (Chigorin)",
        _from_pgn(
            """
            1. e4 e5 2. Nf3 Nc6 3. Bb5 a6 4. Ba4 Nf6 5. O-O Be7 6. Re1 b5
            7. Bb3 d6 8. c3 O-O 9. h3 Na5 10. Bc2 c5 11. d4 Qc7 12. Nbd2 cxd4
            13. cxd4 Nc6
            """,
            ply=26,
        ),
    ),
    Example(
        "Najdorf middlegame (English Attack)",
        _from_pgn(
            """
            1. e4 c5 2. Nf3 d6 3. d4 cxd4 4. Nxd4 Nf6 5. Nc3 a6 6. Be3 e5
            7. Nb3 Be6 8. f3 Be7 9. Qd2 O-O 10. O-O-O Nbd7 11. g4 b5 12. g5 b4
            13. Ne2 Ne8
            """,
            ply=26,
        ),
    ),
    Example(
        "King's Indian middlegame (Mar del Plata)",
        _from_pgn(
            """
            1. d4 Nf6 2. c4 g6 3. Nc3 Bg7 4. e4 d6 5. Nf3 O-O 6. Be2 e5
            7. O-O Nc6 8. d5 Ne7 9. Ne1 Nd7 10. Nd3 f5 11. Bd2 Nf6 12. f3 f4
            """,
            ply=24,
        ),
    ),
    Example(
        "Queen's Gambit middlegame (Tartakower)",
        _from_pgn(
            """
            1. d4 d5 2. c4 e6 3. Nc3 Nf6 4. Bg5 Be7 5. e3 O-O 6. Nf3 h6
            7. Bh4 b6 8. cxd5 Nxd5 9. Bxe7 Qxe7 10. Nxd5 exd5 11. Rc1 Be6
            12. Qa4 c5
            """,
            ply=24,
        ),
    ),
    Example(
        "French Winawer middlegame (Poisoned Pawn)",
        _from_pgn(
            """
            1. e4 e6 2. d4 d5 3. Nc3 Bb4 4. e5 c5 5. a3 Bxc3+ 6. bxc3 Ne7
            7. Qg4 Qc7 8. Qxg7 Rg8 9. Qxh7 cxd4 10. Ne2 Nbc6 11. f4 dxc3
            """,
            ply=22,
        ),
    ),
    Example(
        "Opera Game middlegame (Morphy, 1858)",
        _from_pgn(
            """
            1. e4 e5 2. Nf3 d6 3. d4 Bg4 4. dxe5 Bxf3 5. Qxf3 dxe5 6. Bc4 Nf6
            7. Qb3 Qe7 8. Nc3 c6 9. Bg5 b5 10. Nxb5 cxb5 11. Bxb5+ Nbd7
            12. O-O-O Rd8
            """,
            ply=24,
        ),
    ),
    Example(
        "Lucena rook endgame",
        "1K1k4/1P6/8/8/8/8/r7/2R5 w - - 0 1",
    ),
)
