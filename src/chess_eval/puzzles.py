from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import chess

from src.chess_eval.samples import DEFAULT_PUZZLE_DIR, load_puzzle_samples


@dataclass(frozen=True)
class SolvedPuzzle:
    """A sacrifice puzzle with its full solution and the piece it gives up."""

    puzzle_id: str
    boards: tuple[chess.Board, ...]
    sacrificed_square: int | None
    rating: int


def sacrificed_square(boards: Sequence[chess.Board]) -> int | None:
    """Square of the first piece the solver sacrifices, in the start position.

    A sacrifice is a solver move whose piece is captured on its destination
    square by the immediate reply. The returned square is where that piece
    stood in ``boards[0]``, tracked through any earlier moves.
    """
    if len(boards) < 3:
        return None
    start = boards[0]
    solver = not start.turn
    ids = {square: index for index, square in enumerate(start.piece_map())}
    start_of = {index: square for square, index in ids.items()}

    for index in range(len(boards) - 1):
        board = boards[index]
        move = _move_between(boards[index], boards[index + 1])
        if move is None:
            return None
        piece = ids.get(move.from_square)
        sacrificed = False
        if board.turn == solver and index + 2 < len(boards):
            reply = _move_between(boards[index + 1], boards[index + 2])
            if (
                reply is not None
                and reply.to_square == move.to_square
                and boards[index + 1].is_capture(reply)
            ):
                sacrificed = True
        _advance_ids(ids, board, move)
        if sacrificed and piece is not None:
            return start_of.get(piece)
    return None


def load_solved_puzzles(
    data_dir: str | Path | None = None,
    max_puzzles: int = 30,
    seed: int = 17,
    shard_limit: int = 5,
) -> list[SolvedPuzzle]:
    """Load sacrifice puzzles as board sequences with their sacrificed piece."""
    samples = load_puzzle_samples(
        data_dir if data_dir is not None else DEFAULT_PUZZLE_DIR,
        max_puzzles=max_puzzles,
        seed=seed,
        shard_limit=shard_limit,
    )
    groups: dict[str, list] = {}
    for sample in samples:
        if sample.sacrifice != 1:
            continue
        groups.setdefault(sample.puzzle_id, []).append(sample)

    puzzles: list[SolvedPuzzle] = []
    for puzzle_id, group in groups.items():
        group.sort(key=lambda sample: sample.position_index)
        boards = tuple(chess.Board(sample.fen) for sample in group)
        puzzles.append(
            SolvedPuzzle(
                puzzle_id=puzzle_id,
                boards=boards,
                sacrificed_square=sacrificed_square(boards),
                rating=group[0].rating,
            )
        )
    return sorted(puzzles, key=lambda puzzle: puzzle.puzzle_id)


def background_boards(
    data_dir: str | Path | None = None,
    count: int = 4,
    seed: int = 17,
    shard_limit: int = 5,
) -> tuple[chess.Board, ...]:
    """Real start positions used as the interventional SHAP background."""
    if count < 0:
        raise ValueError("count must be non-negative")
    if count == 0:
        return ()
    samples = load_puzzle_samples(
        data_dir if data_dir is not None else DEFAULT_PUZZLE_DIR,
        max_puzzles=max(12, count * 3),
        seed=seed,
        shard_limit=shard_limit,
    )
    starts: dict[str, str] = {}
    for sample in samples:
        if sample.position_index == 0:
            starts.setdefault(sample.puzzle_id, sample.fen)
    fens = list(starts.values())
    if not fens:
        return ()
    if len(fens) <= count:
        chosen = fens
    else:
        step = len(fens) / count
        chosen = [fens[min(len(fens) - 1, int(index * step))] for index in range(count)]
    return tuple(chess.Board(fen) for fen in chosen)


def _move_between(before: chess.Board, after: chess.Board) -> chess.Move | None:
    for move in before.legal_moves:
        probe = before.copy()
        probe.push(move)
        if probe.board_fen() == after.board_fen():
            return move
    return None


def _advance_ids(ids: dict[int, int], board: chess.Board, move: chess.Move) -> None:
    if move.from_square in ids:
        ids[move.to_square] = ids.pop(move.from_square)
    if board.is_en_passant(move):
        captured = chess.square(
            chess.square_file(move.to_square), chess.square_rank(move.from_square)
        )
        ids.pop(captured, None)
    else:
        ids.pop(move.to_square, None)
    return
