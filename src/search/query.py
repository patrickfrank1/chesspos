"""Group position hits into games and run end-to-end queries."""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from typing import Sequence

import numpy as np

from src.search.codec import decode_board
from src.search.schema import game_id_to_hex
from src.search.store import PositionFilter, PositionHit, PositionStore


@dataclass(frozen=True)
class GameHit:
    """One game ranked by its best matching position."""

    game_id: int
    score: float
    best_ply: int
    best_position: bytes
    match_count: int
    white_elo: int
    black_elo: int
    utc_date: dt.date
    year: int
    result: str
    split: str
    n_pieces: int

    def to_dict(self) -> dict:
        return {
            "game_id": game_id_to_hex(self.game_id),
            "score": self.score,
            "ply": self.best_ply,
            "match_count": self.match_count,
            "fen": decode_board(self.best_position).fen(),
            "white_elo": self.white_elo,
            "black_elo": self.black_elo,
            "utc_date": self.utc_date.isoformat(),
            "year": self.year,
            "result": self.result,
            "split": self.split,
            "n_pieces": self.n_pieces,
        }


def _game_hit(game_id: int, hits: Sequence[PositionHit]) -> GameHit:
    ordered = sorted(hits, key=lambda hit: (hit.distance, hit.ply))
    best = ordered[0]
    return GameHit(
        game_id=game_id,
        score=best.score,
        best_ply=best.ply,
        best_position=best.packed_position,
        match_count=len(hits),
        white_elo=best.white_elo,
        black_elo=best.black_elo,
        utc_date=best.utc_date,
        year=best.year,
        result=best.result,
        split=best.split,
        n_pieces=best.n_pieces,
    )


def group_position_hits(
    hits: Sequence[PositionHit],
    *,
    n_games: int = 10,
    exclude_game_id: int | None = None,
) -> list[GameHit]:
    """Collapse position hits into one ranked entry per game."""
    if n_games <= 0:
        return []
    grouped: dict[int, list[PositionHit]] = {}
    for hit in hits:
        if exclude_game_id is not None and hit.game_id == exclude_game_id:
            continue
        grouped.setdefault(hit.game_id, []).append(hit)
    games = [_game_hit(game_id, group) for game_id, group in grouped.items()]
    games.sort(key=lambda game: (-game.score, game.game_id))
    return games[:n_games]


def search_games(
    *,
    store: PositionStore,
    embedder,
    segment: np.ndarray,
    n_games: int = 10,
    candidate_limit: int = 500,
    filter: PositionFilter | None = None,
    nprobes: int | None = None,
    refine_factor: int | None = None,
    exact: bool = False,
) -> list[GameHit]:
    """Embed a query position, search, and group the hits by game."""
    vectors = np.asarray(embedder.embed([segment]), dtype=np.float32)
    if vectors.shape != (1, embedder.dim):
        raise ValueError(
            f"embedder returned shape {vectors.shape}, expected {(1, embedder.dim)}"
        )
    hits = store.search(
        vectors[0],
        limit=candidate_limit,
        filter=filter,
        nprobes=nprobes,
        refine_factor=refine_factor,
        exact=exact,
    )
    return group_position_hits(
        hits,
        n_games=n_games,
        exclude_game_id=None if filter is None else filter.exclude_game_id,
    )
