import json

import chess
import numpy as np

from bdd import given, then, when
from src.run.build_position_index import parse_args as parse_build_args
from src.run.search_positions import (
    format_text,
    parse_args as parse_search_args,
    run_search,
)
from src.search.memory_store import MemoryPositionStore
from position_support import FakeEmbedder, make_position_row


def test_build_index_parser_has_safe_defaults():
    with given("the required build arguments"):
        argv = [
            "--data-dir",
            "data/processed/hub/train",
            "--checkpoint",
            "data/models/run/last.pt",
            "--output",
            "data/vector",
        ]

    with when("the arguments are parsed"):
        args = parse_build_args(argv)

    with then("the CLI defaults to a batched IVF-PQ build"):
        assert args.data_dir == "data/processed/hub/train"
        assert args.repo is None
        assert args.split == "train"
        assert args.checkpoint == "data/models/run/last.pt"
        assert args.output == "data/vector"
        assert args.batch_games == 32
        assert args.index_type == "ivf_pq"
        assert args.limit_games is None
        assert args.no_index is False
        assert args.overwrite is False
    return


def test_build_index_parser_accepts_a_hub_repo_source():
    with given("a Hub repo and a shard budget"):
        argv = [
            "--repo",
            "patrickfrank1/chess-positions",
            "--split",
            "train",
            "--max-bytes",
            "200000000",
            "--checkpoint",
            "data/models/run/last.pt",
            "--output",
            "data/vector",
        ]

    with when("the arguments are parsed"):
        args = parse_build_args(argv)

    with then("the repo source and budget are carried on the namespace"):
        assert args.repo == "patrickfrank1/chess-positions"
        assert args.split == "train"
        assert args.max_bytes == 200_000_000
        assert args.data_dir is None
    return


def test_search_parser_builds_a_metadata_filter():
    with given("a search query with filters"):
        argv = [
            "--index",
            "data/vector",
            "--checkpoint",
            "data/models/run/last.pt",
            "--fen",
            chess.STARTING_FEN,
            "--k",
            "50",
            "--n-games",
            "5",
            "--min-elo",
            "2400",
            "--min-year",
            "2020",
            "--result",
            "1-0",
            "--split",
            "test",
        ]

    with when("the arguments are parsed"):
        args = parse_search_args(argv)

    with then("the filter values are carried on the namespace"):
        assert args.index == "data/vector"
        assert args.fen == chess.STARTING_FEN
        assert args.k == 50
        assert args.n_games == 5
        assert args.min_elo == 2400
        assert args.min_year == 2020
        assert args.result == "1-0"
        assert args.split == "test"
        assert args.by == "game"
    return


def test_run_search_returns_grouped_games():
    with given("a store with two games and a FEN query"):
        store = MemoryPositionStore(dim=8)
        rows = []
        for index in range(2):
            vector = np.zeros(8, dtype=np.float32)
            vector[index] = 1.0
            rows.append(make_position_row(vector=vector, game_id=index, ply=1))
        store.add(rows)
        args = parse_search_args(
            [
                "--index",
                "unused",
                "--checkpoint",
                "unused",
                "--fen",
                chess.STARTING_FEN,
                "--k",
                "10",
                "--n-games",
                "2",
            ]
        )

    with when("the search is executed with a deterministic embedder"):
        payload = run_search(args, store=store, embedder=FakeEmbedder(dim=8))

    with then("the payload contains the query and grouped games"):
        assert payload["query"]["fen"] == chess.STARTING_FEN
        assert len(payload["games"]) == 2
        assert set(payload["games"][0]) >= {
            "game_id",
            "score",
            "ply",
            "match_count",
            "fen",
        }
    return


def test_text_output_contains_the_game_and_position():
    with given("a search payload with one game"):
        payload = {
            "query": {"fen": chess.STARTING_FEN},
            "games": [
                {
                    "game_id": "0123456789abcdef",
                    "score": 0.9,
                    "ply": 12,
                    "match_count": 2,
                    "fen": chess.STARTING_FEN,
                    "white_elo": 2500,
                    "black_elo": 2400,
                    "result": "1-0",
                    "split": "train",
                    "year": 2024,
                    "utc_date": "2024-01-01",
                    "n_pieces": 32,
                }
            ],
        }

    with when("the payload is rendered as text"):
        text = format_text(payload)

    with then("the id, score and FEN appear"):
        assert "0123456789abcdef" in text
        assert "0.900" in text
        assert chess.STARTING_FEN.split()[0] in text
    return


def test_json_output_is_parseable():
    with given("a search payload"):
        payload = {"query": {}, "games": []}

    with when("it is serialised as JSON"):
        encoded = json.dumps(payload)

    with then("the result parses back"):
        assert json.loads(encoded) == payload
    return
