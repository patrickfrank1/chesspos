from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

import chess
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from src.dataset.puzzle_variants import _split_stream
from src.dataset.token_stream import SEGMENT_LEN, TokenStreamEncoder

_PIECE_VALUES = {
    chess.PAWN: 1,
    chess.KNIGHT: 3,
    chess.BISHOP: 3,
    chess.ROOK: 5,
    chess.QUEEN: 9,
    chess.KING: 0,
}
_THEME_COLUMNS_REQUIRED = {"puzzle_id", "packed", "rating", "previous_position"}

SURROGATE_KEYS = (
    "material",
    "piece_count",
    "phase",
    "open_files",
    "semi_open_files",
    "rook_on_open_file",
    "pawn_islands",
    "isolated_pawns",
    "doubled_pawns",
    "passed_pawns",
    "mobility",
    "king_safety",
    "center_control",
    "bishop_pair",
    "is_check",
    "king_attack",
    "knights_on_edge",
    "inactive_bishops",
    "smothered_king",
)


@dataclass(frozen=True)
class PositionSample:
    sample_id: int
    tokens: tuple[int, ...]
    fen: str
    source: str
    game_id: str | None = None
    puzzle_id: str | None = None
    ply: int | None = None
    themes: tuple[str, ...] = ()
    rating: int | None = None
    surrogates: dict[str, float | int] = field(default_factory=dict)


def _pawns(board: chess.Board, color: chess.Color) -> dict[int, list[int]]:
    files: dict[int, list[int]] = {}
    for square in board.pieces(chess.PAWN, color):
        files.setdefault(chess.square_file(square), []).append(
            chess.square_rank(square)
        )
    return files


def _isolated_pawns(board: chess.Board, color: chess.Color) -> int:
    files = _pawns(board, color)
    return sum(
        1
        for file, ranks in files.items()
        for _ in ranks
        if not files.get(file - 1) and not files.get(file + 1)
    )


def _doubled_pawns(board: chess.Board, color: chess.Color) -> int:
    files = _pawns(board, color)
    return sum(max(0, len(ranks) - 1) for ranks in files.values())


def _passed_pawns(board: chess.Board, color: chess.Color) -> int:
    own = _pawns(board, color)
    enemy = _pawns(board, not color)
    direction = 1 if color == chess.WHITE else -1
    count = 0
    for file, ranks in own.items():
        for rank in ranks:
            blocked = False
            for adjacent in (file - 1, file, file + 1):
                for enemy_rank in enemy.get(adjacent, []):
                    ahead = (enemy_rank - rank) * direction > 0
                    if ahead:
                        blocked = True
                        break
                if blocked:
                    break
            if not blocked:
                count += 1
    return count


def _pawn_islands(board: chess.Board, color: chess.Color) -> int:
    files = sorted(_pawns(board, color))
    if not files:
        return 0
    islands = 1
    for previous, current in zip(files, files[1:]):
        if current - previous > 1:
            islands += 1
    return islands


def _king_shield(board: chess.Board, color: chess.Color) -> int:
    king = board.king(color)
    if king is None:
        return 0
    file = chess.square_file(king)
    rank = chess.square_rank(king)
    direction = 1 if color == chess.WHITE else -1
    shield = 0
    for adjacent in (file - 1, file, file + 1):
        if not 0 <= adjacent <= 7:
            continue
        for distance in (1, 2):
            target_rank = rank + direction * distance
            if not 0 <= target_rank <= 7:
                continue
            if board.piece_at(chess.square(adjacent, target_rank)) == chess.Piece(
                chess.PAWN, color
            ):
                shield += 1
                break
    return shield


def surrogate_matrix(
    samples: Sequence[PositionSample], keys: Sequence[str] = SURROGATE_KEYS
) -> np.ndarray:
    """Numeric surrogate features for negative-pair matching."""
    return np.array(
        [
            [float(sample.surrogates.get(key, 0.0)) for key in keys]
            for sample in samples
        ],
        dtype=np.float64,
    )


def _king_attack_pressure(board: chess.Board) -> int:
    pressure = 0
    for color in (chess.WHITE, chess.BLACK):
        king = board.king(color)
        if king is None:
            continue
        enemy = not color
        attacked = sum(
            1
            for square in chess.SquareSet(chess.BB_KING_ATTACKS[king])
            if board.attackers(enemy, square)
        )
        pressure = max(pressure, attacked)
    return pressure


def _knights_on_edge(board: chess.Board) -> int:
    return sum(
        1
        for color in (chess.WHITE, chess.BLACK)
        for square in board.pieces(chess.KNIGHT, color)
        if chess.square_file(square) in (0, 7)
    )


def _inactive_bishops(board: chess.Board) -> int:
    bishops = {
        square: 0
        for color in (chess.WHITE, chess.BLACK)
        for square in board.pieces(chess.BISHOP, color)
    }
    for move in board.pseudo_legal_moves:
        piece = board.piece_at(move.from_square)
        if move.from_square in bishops and piece is not None:
            bishops[move.from_square] += 1
    return sum(1 for count in bishops.values() if count <= 2)


def _smothered_kings(board: chess.Board) -> int:
    if not board.is_check():
        return 0
    king = board.king(board.turn)
    if king is None:
        return 0
    king_moves = [move for move in board.legal_moves if move.from_square == king]
    return 0 if king_moves else 1


def board_surrogates(board: chess.Board) -> dict[str, float | int]:
    """Structural, model-independent descriptions of a board."""
    material = 0
    for piece in board.piece_map().values():
        value = _PIECE_VALUES[piece.piece_type]
        material += value if piece.color == chess.WHITE else -value
    piece_count = len(board.piece_map())

    pawn_files = {
        color: set(_pawns(board, color)) for color in (chess.WHITE, chess.BLACK)
    }
    all_pawn_files = pawn_files[chess.WHITE] | pawn_files[chess.BLACK]
    open_files = sum(1 for file in range(8) if file not in all_pawn_files)
    semi_open_files = sum(
        1
        for file in range(8)
        if file not in pawn_files[chess.WHITE] or file not in pawn_files[chess.BLACK]
    )
    rook_on_open_file = sum(
        1
        for square in board.pieces(chess.ROOK, chess.WHITE)
        | board.pieces(chess.ROOK, chess.BLACK)
        if chess.square_file(square) not in all_pawn_files
    )
    center = [chess.D4, chess.E4, chess.D5, chess.E5]
    center_control = sum(
        1
        for square in center
        if board.attackers(chess.WHITE, square) or board.attackers(chess.BLACK, square)
    )
    bishop_pair = int(
        any(
            len(board.pieces(chess.BISHOP, color)) >= 2
            for color in (chess.WHITE, chess.BLACK)
        )
    )
    king_safety = min(
        _king_shield(board, chess.WHITE), _king_shield(board, chess.BLACK)
    )
    phase = 0 if piece_count >= 24 else 1 if piece_count >= 14 else 2

    return {
        "material": int(material),
        "piece_count": int(piece_count),
        "phase": int(phase),
        "turn": int(board.turn == chess.WHITE),
        "castling_rights": len(board.castling_xfen().strip("-")),
        "en_passant": int(board.ep_square is not None),
        "isolated_pawns": int(
            _isolated_pawns(board, chess.WHITE) + _isolated_pawns(board, chess.BLACK)
        ),
        "doubled_pawns": int(
            _doubled_pawns(board, chess.WHITE) + _doubled_pawns(board, chess.BLACK)
        ),
        "passed_pawns": int(
            _passed_pawns(board, chess.WHITE) + _passed_pawns(board, chess.BLACK)
        ),
        "pawn_islands": int(
            _pawn_islands(board, chess.WHITE) + _pawn_islands(board, chess.BLACK)
        ),
        "open_files": int(open_files),
        "semi_open_files": int(semi_open_files),
        "rook_on_open_file": int(rook_on_open_file),
        "center_control": int(center_control),
        "bishop_pair": int(bishop_pair),
        "king_safety": int(king_safety),
        "mobility": int(board.legal_moves.count()),
        "is_check": int(board.is_check()),
        "king_attack": int(_king_attack_pressure(board)),
        "knights_on_edge": int(_knights_on_edge(board)),
        "inactive_bishops": int(_inactive_bishops(board)),
        "smothered_king": int(_smothered_kings(board)),
    }


def make_sample(
    tokens: Sequence[int] | np.ndarray,
    source: str,
    *,
    game_id: str | None = None,
    puzzle_id: str | None = None,
    ply: int | None = None,
    themes: Sequence[str] = (),
    rating: int | None = None,
    sample_id: int = -1,
) -> PositionSample:
    values = np.asarray(tokens, dtype=np.int16).ravel()
    if values.size != SEGMENT_LEN:
        raise ValueError(
            f"Expected a {SEGMENT_LEN}-token segment, got {values.size} tokens"
        )
    board = TokenStreamEncoder().decode(values)
    return PositionSample(
        sample_id=sample_id,
        tokens=tuple(int(token) for token in values),
        fen=board.fen(),
        source=source,
        game_id=game_id,
        puzzle_id=puzzle_id,
        ply=ply,
        themes=tuple(str(theme) for theme in themes),
        rating=rating,
        surrogates=board_surrogates(board),
    )


class _Reservoir:
    def __init__(self, size: int, rng: np.random.Generator):
        self.size = max(0, size)
        self.rng = rng
        self.items: list = []
        self.seen = 0
        return

    def add(self, item) -> None:
        self.seen += 1
        if len(self.items) < self.size:
            self.items.append(item)
            return
        if self.size == 0:
            return
        slot = int(self.rng.integers(0, self.seen))
        if slot < self.size:
            self.items[slot] = item
        return


def _parquet_paths(data_dir: str | Path) -> list[Path]:
    paths = sorted(Path(data_dir).rglob("*.parquet"))
    if not paths:
        raise ValueError(f"No parquet shards found under {data_dir}")
    return paths


def _streams_from_column(column) -> tuple[np.ndarray, np.ndarray]:
    combined = column.combine_chunks()
    offsets = np.asarray(combined.offsets)
    values = combined.flatten().to_numpy(zero_copy_only=False).astype(np.int16)
    return offsets, values


def load_test_samples(
    data_dir: str | Path,
    n: int,
    *,
    per_game_cap: int = 1,
    seed: int = 0,
) -> list[PositionSample]:
    """Deterministically sample contiguous position windows from the test split."""
    if n <= 0:
        raise ValueError("n must be positive")
    if per_game_cap <= 0:
        raise ValueError("per_game_cap must be positive")

    rng = np.random.default_rng(seed)
    max_windows = max(1, (n + per_game_cap - 1) // per_game_cap)
    reservoir = _Reservoir(max_windows, rng)
    for path in _parquet_paths(data_dir):
        table = pq.read_table(path, columns=["packed", "game_id"])
        game_ids = table.column("game_id").to_pylist()
        offsets, values = _streams_from_column(table.column("packed"))
        for row, game_id in enumerate(game_ids):
            stream = values[offsets[row] : offsets[row + 1]]
            segments = [
                segment
                for segment in _split_stream(stream)
                if segment.size == SEGMENT_LEN
            ]
            if not segments:
                continue
            window_size = min(per_game_cap, len(segments))
            if len(segments) > window_size:
                start = int(rng.integers(0, len(segments) - window_size + 1))
            else:
                start = 0
            window = tuple(
                (start + offset, segments[start + offset].copy())
                for offset in range(window_size)
            )
            reservoir.add((str(game_id), window))
        continue

    if not reservoir.items:
        raise ValueError(
            f"No {SEGMENT_LEN}-token segments found under {data_dir}; "
            "the shard may use a legacy token encoding"
        )
    positions = [
        (game_id, ply, segment)
        for game_id, window in reservoir.items
        for ply, segment in window
    ][:n]
    rows = sorted(
        positions,
        key=lambda item: (item[0], item[1], tuple(int(token) for token in item[2])),
    )
    return [
        make_sample(
            segment,
            "test",
            game_id=game_id,
            ply=ply,
            sample_id=index,
        )
        for index, (game_id, ply, segment) in enumerate(rows)
    ]


def _scalar_columns(path: Path) -> set[str]:
    schema = pq.read_schema(path)
    return {field.name for field in schema if not pa.types.is_list(field.type)}


def load_puzzle_samples(
    data_dir: str | Path,
    n: int,
    *,
    themes: Sequence[str] = (),
    position: str = "first",
    seed: int = 0,
) -> list[PositionSample]:
    """Load one position per puzzle, labelled with the puzzle's themes."""
    if n <= 0:
        raise ValueError("n must be positive")
    if position not in ("first", "last", "random"):
        raise ValueError("position must be 'first', 'last', or 'random'")

    requested = tuple(str(theme) for theme in themes)
    rng = np.random.default_rng(seed)
    global_reservoir = _Reservoir(n, rng)
    quota = max(1, n // len(requested)) if requested else n
    theme_reservoirs = {theme: _Reservoir(quota, rng) for theme in requested}

    for path in _parquet_paths(data_dir):
        names = _scalar_columns(path)
        available_themes = sorted(names - _THEME_COLUMNS_REQUIRED)
        selected_themes = [
            theme for theme in requested if theme in available_themes
        ] or (requested and [])
        if requested and not selected_themes:
            continue
        columns = ["puzzle_id", "packed", "rating", *selected_themes]
        table = pq.read_table(path, columns=columns)
        puzzle_ids = table.column("puzzle_id").to_pylist()
        ratings = table.column("rating").to_pylist()
        theme_values = {
            theme: table.column(theme).to_pylist() for theme in selected_themes
        }
        offsets, values = _streams_from_column(table.column("packed"))

        for row, puzzle_id in enumerate(puzzle_ids):
            segments = [
                segment
                for segment in _split_stream(values[offsets[row] : offsets[row + 1]])
                if segment.size == SEGMENT_LEN
            ]
            if not segments:
                continue
            present = tuple(
                theme for theme in selected_themes if theme_values[theme][row] == 1
            )
            if requested and not present:
                continue
            if position == "first":
                index = 0
            elif position == "last":
                index = len(segments) - 1
            else:
                index = int(rng.integers(len(segments)))
            candidate = (
                str(puzzle_id),
                index,
                segments[index].copy(),
                present,
                ratings[row],
            )
            global_reservoir.add(candidate)
            for theme in present:
                if theme in theme_reservoirs:
                    theme_reservoirs[theme].add(candidate)
        continue

    chosen = []
    seen: set[tuple[str, int]] = set()
    if requested:
        for theme in requested:
            for candidate in theme_reservoirs.get(theme, _Reservoir(0, rng)).items:
                key = (candidate[0], candidate[1])
                if key not in seen:
                    seen.add(key)
                    chosen.append(candidate)
    for candidate in global_reservoir.items:
        key = (candidate[0], candidate[1])
        if key not in seen:
            seen.add(key)
            chosen.append(candidate)

    chosen = chosen[:n]
    chosen.sort(key=lambda item: (item[0], item[1]))
    if not chosen:
        raise ValueError(
            f"No {SEGMENT_LEN}-token puzzle segments found under {data_dir}; "
            "the shard may use a legacy token encoding"
        )
    return [
        make_sample(
            segment,
            "puzzle",
            puzzle_id=puzzle_id,
            ply=index,
            themes=present,
            rating=rating,
            sample_id=sample_index,
        )
        for sample_index, (puzzle_id, index, segment, present, rating) in enumerate(
            chosen
        )
    ]
