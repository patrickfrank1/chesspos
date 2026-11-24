"""BDD tests for the position-search web UI."""

from __future__ import annotations

import chess
from fastapi.testclient import TestClient

from bdd import given, then, when
from position_support import FakeEmbedder, make_position_row
from src.search.memory_store import MemoryPositionStore
from src.run.serve_position_search import parse_args as parse_serve_args
from src.web.app import create_app
from src.web.examples import EXAMPLES


def _client() -> tuple[TestClient, MemoryPositionStore]:
    store = MemoryPositionStore(dim=8)
    store.add(
        [
            make_position_row(game_id=1, ply=1),
            make_position_row(game_id=2, ply=2),
            make_position_row(game_id=3, ply=3),
        ]
    )
    app = create_app(store=store, embedder=FakeEmbedder(dim=8))
    return TestClient(app), store


def test_examples_are_legal_positions():
    with given("the built-in example positions"):
        examples = EXAMPLES

    with when("each example FEN is parsed"):
        boards = [chess.Board(example.fen) for example in examples]

    with then("every example is legal, labelled, and unique"):
        assert len(examples) >= 4
        assert all(board.is_valid() for board in boards)
        assert all(example.label.strip() for example in examples)
        assert len({example.fen for example in examples}) == len(examples)
    return


def test_examples_are_middlegames_not_openings():
    with given("the built-in example positions"):
        examples = EXAMPLES
        boards = [chess.Board(example.fen) for example in examples]

    with when("positions after move 10 with 20 or more pieces count as middlegames"):
        middlegames = [
            board
            for board in boards
            if board.fullmove_number >= 10 and len(board.piece_map()) >= 20
        ]

    with then("the dropdown offers several middlegames and no bare start position"):
        assert len(middlegames) >= 5
        assert chess.STARTING_FEN not in {example.fen for example in examples}
    return


def test_index_page_serves_the_search_form():
    with given("the web app"):
        client, _ = _client()

    with when("the index page is requested"):
        response = client.get("/")

    with then("the page contains the FEN input, the example selector and the API call"):
        assert response.status_code == 200
        assert "text/html" in response.headers["content-type"]
        html = response.text
        assert 'id="fen"' in html
        assert 'id="examples"' in html
        assert "/api/search" in html
    return


def test_examples_endpoint_returns_selectable_positions():
    with given("the web app"):
        client, _ = _client()

    with when("the example list is requested"):
        response = client.get("/api/examples")

    with then("it returns every example with an id and a label"):
        assert response.status_code == 200
        examples = response.json()["examples"]
        assert [example["fen"] for example in examples] == [
            example.fen for example in EXAMPLES
        ]
        assert all(example["label"] for example in examples)
    return


def test_search_endpoint_returns_grouped_games_with_boards():
    with given("a store with three games"):
        client, _ = _client()

    with when("the starting position is searched for two games"):
        response = client.post(
            "/api/search",
            json={"fen": chess.STARTING_FEN, "n_games": 2, "k": 10},
        )

    with then("the response carries the query board and ranked game boards"):
        assert response.status_code == 200
        payload = response.json()
        assert payload["query"]["fen"] == chess.STARTING_FEN
        assert payload["query"]["svg"].startswith("<svg")
        games = payload["games"]
        assert len(games) == 2
        assert games[0]["score"] >= games[1]["score"]
        assert games[0]["svg"].startswith("<svg")
        assert {
            "game_id",
            "score",
            "ply",
            "match_count",
            "fen",
            "svg",
            "white_elo",
            "black_elo",
            "result",
            "year",
        } <= set(games[0])
    return


def test_search_endpoint_honours_the_game_budget():
    with given("a store with three games"):
        client, _ = _client()

    with when("only one game is requested"):
        response = client.post(
            "/api/search",
            json={"fen": chess.STARTING_FEN, "n_games": 1, "k": 10},
        )

    with then("exactly one game is returned"):
        assert response.status_code == 200
        assert len(response.json()["games"]) == 1
    return


def test_search_endpoint_rejects_an_invalid_fen():
    with given("the web app"):
        client, _ = _client()

    with when("a malformed FEN is searched"):
        response = client.post("/api/search", json={"fen": "not a fen"})

    with then("the API answers with a readable 400 instead of crashing"):
        assert response.status_code == 400
        assert "FEN" in response.json()["detail"]
    return


def test_search_endpoint_requires_a_fen():
    with given("the web app"):
        client, _ = _client()

    with when("a request without a FEN is sent"):
        response = client.post("/api/search", json={})

    with then("request validation rejects it"):
        assert response.status_code == 422
    return


def test_info_endpoint_reports_the_index_size():
    with given("a store with three rows"):
        client, _ = _client()

    with when("the index info is requested"):
        response = client.get("/api/info")

    with then("the row count and dimension are reported"):
        assert response.status_code == 200
        payload = response.json()
        assert payload["rows"] == 3
        assert payload["dim"] == 8
    return


def test_serve_cli_defaults_to_localhost():
    with given("the required serve arguments"):
        argv = ["--index", "data/vector/ae-20k", "--checkpoint", "run/last.pt"]

    with when("the arguments are parsed"):
        args = parse_serve_args(argv)

    with then("the server binds to localhost and the default table"):
        assert args.index == "data/vector/ae-20k"
        assert args.checkpoint == "run/last.pt"
        assert args.host == "127.0.0.1"
        assert args.port == 8000
        assert args.table == "positions"
    return
