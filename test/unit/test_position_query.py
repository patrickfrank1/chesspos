import datetime as dt

import numpy as np
import pytest

from bdd import given, then, when
from src.search.codec import decode_position
from src.search.memory_store import MemoryPositionStore
from src.search.query import group_position_hits, search_games
from src.search.store import PositionHit
from position_support import FakeEmbedder, make_position_row


def _hit(
    *,
    game_id: int,
    ply: int,
    distance: float,
    result: str = "1-0",
    split: str = "train",
    year: int = 2024,
) -> PositionHit:
    row = make_position_row(
        game_id=game_id,
        ply=ply,
        result=result,
        split=split,
        year=year,
        utc_date=dt.date(year, 1, 1),
    )
    return PositionHit(
        game_id=row.game_id,
        ply=row.ply,
        packed_position=row.packed_position,
        distance=distance,
        white_elo=row.white_elo,
        black_elo=row.black_elo,
        utc_date=row.utc_date,
        year=row.year,
        result=row.result,
        split=row.split,
        n_pieces=row.n_pieces,
    )


def test_grouping_keeps_the_best_hit_per_game():
    with given("position hits from three games, one with two matches"):
        hits = [
            _hit(game_id=1, ply=10, distance=0.10),
            _hit(game_id=1, ply=11, distance=0.20),
            _hit(game_id=2, ply=20, distance=0.15),
            _hit(game_id=3, ply=30, distance=0.30),
        ]

    with when("the hits are grouped into games"):
        games = group_position_hits(hits, n_games=2)

    with then("games are ordered by their best score and keep match counts"):
        assert [game.game_id for game in games] == [1, 2]
        assert games[0].score == pytest.approx(0.90)
        assert games[0].best_ply == 10
        assert games[0].match_count == 2
        assert games[1].score == pytest.approx(0.85)
        assert games[1].best_ply == 20
        assert games[1].match_count == 1
    return


def test_grouping_can_exclude_a_game():
    with given("hits from three games"):
        hits = [
            _hit(game_id=1, ply=10, distance=0.10),
            _hit(game_id=2, ply=20, distance=0.15),
            _hit(game_id=3, ply=30, distance=0.30),
        ]

    with when("the query excludes game 2"):
        games = group_position_hits(hits, n_games=3, exclude_game_id=2)

    with then("the excluded game is absent"):
        assert [game.game_id for game in games] == [1, 3]
    return


def test_game_hit_serialises_to_a_rendered_position():
    with given("a grouped game hit"):
        hit = _hit(game_id=0x0123456789ABCDEF, ply=5, distance=0.12)
        game = group_position_hits([hit], n_games=1)[0]

    with when("the hit is serialised"):
        payload = game.to_dict()

    with then("the payload has a hex id, score, ply and FEN"):
        assert payload["game_id"] == "0123456789abcdef"
        assert payload["score"] == pytest.approx(0.88)
        assert payload["ply"] == 5
        assert payload["match_count"] == 1
        assert (
            payload["fen"].split()[0] == "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR"
        )
    return


def test_empty_hits_produce_no_games():
    with given("no position hits"):
        hits: list[PositionHit] = []

    with when("the hits are grouped"):
        games = group_position_hits(hits, n_games=10)

    with then("the result is empty"):
        assert games == []
    return


def test_search_games_embeds_and_groups():
    with given("a store containing four games"):
        rows = []
        for index in range(4):
            vector = np.zeros(8, dtype=np.float32)
            vector[index] = 1.0
            rows.append(make_position_row(vector=vector, game_id=index, ply=1))
        store = MemoryPositionStore(dim=8)
        store.add(rows)

    with when("a query position is embedded and searched"):
        games = search_games(
            store=store,
            embedder=FakeEmbedder(dim=8),
            segment=decode_position(rows[0].packed_position),
            n_games=2,
            candidate_limit=10,
        )

    with then("the result is grouped into distinct games"):
        assert len(games) == 2
        assert len({game.game_id for game in games}) == 2
        assert all(game.match_count >= 1 for game in games)
    return
