"""CLI: search a position vector index for similar games."""

from __future__ import annotations

import argparse
import datetime as dt
import json
from pathlib import Path
from typing import Sequence

import chess

from src.dataset.token_stream import TokenStreamEncoder
from src.search.codec import decode_board
from src.search.query import search_games
from src.search.schema import game_id_to_hex, game_id_to_int
from src.search.store import PositionFilter, PositionHit


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Find games with positions similar to a query position"
    )
    parser.add_argument("--index", required=True)
    parser.add_argument("--checkpoint", required=True)
    query = parser.add_mutually_exclusive_group()
    query.add_argument("--fen", default=None)
    query.add_argument("--pgn", default=None)
    parser.add_argument("--moves", nargs="*", default=[])
    parser.add_argument("--ply", type=int, default=None)
    parser.add_argument("--k", type=int, default=500)
    parser.add_argument("--n-games", type=int, default=10)
    parser.add_argument("--by", choices=("game", "position"), default="game")
    parser.add_argument("--min-elo", type=int, default=None)
    parser.add_argument("--max-elo", type=int, default=None)
    parser.add_argument("--min-year", type=int, default=None)
    parser.add_argument("--max-year", type=int, default=None)
    parser.add_argument("--min-date", default=None)
    parser.add_argument("--max-date", default=None)
    parser.add_argument("--result", default=None)
    parser.add_argument("--split", default=None)
    parser.add_argument("--min-ply", type=int, default=None)
    parser.add_argument("--max-ply", type=int, default=None)
    parser.add_argument("--min-pieces", type=int, default=None)
    parser.add_argument("--max-pieces", type=int, default=None)
    parser.add_argument("--exclude-game", default=None)
    parser.add_argument("--nprobes", type=int, default=None)
    parser.add_argument("--refine-factor", type=int, default=None)
    parser.add_argument("--exact", action="store_true")
    parser.add_argument("--device", default=None)
    parser.add_argument("--table", default="positions")
    parser.add_argument("--json", action="store_true")
    return parser.parse_args(argv)


def _parse_date(value: str | None) -> dt.date | None:
    if not value:
        return None
    return dt.date.fromisoformat(value)


def _query_board(args: argparse.Namespace) -> chess.Board:
    if args.pgn:
        with open(args.pgn, encoding="utf-8") as handle:
            game = chess.pgn.read_game(handle)
        if game is None:
            raise ValueError(f"No game found in {args.pgn}")
        board = game.board()
        for number, move in enumerate(game.mainline_moves(), start=1):
            if args.ply is not None and number > args.ply:
                break
            board.push(move)
    else:
        board = chess.Board(args.fen or chess.STARTING_FEN)

    for move in args.moves:
        try:
            board.push_uci(move)
        except ValueError:
            board.push_san(move)
    return board


def _query_filter(args: argparse.Namespace) -> PositionFilter:
    exclude = game_id_to_int(args.exclude_game) if args.exclude_game else None
    return PositionFilter(
        min_elo=args.min_elo,
        max_elo=args.max_elo,
        min_year=args.min_year,
        max_year=args.max_year,
        min_date=_parse_date(args.min_date),
        max_date=_parse_date(args.max_date),
        result=args.result,
        split=args.split,
        min_ply=args.min_ply,
        max_ply=args.max_ply,
        min_pieces=args.min_pieces,
        max_pieces=args.max_pieces,
        exclude_game_id=exclude,
    )


def _position_payload(hit: PositionHit) -> dict:
    return {
        "game_id": game_id_to_hex(hit.game_id),
        "score": hit.score,
        "ply": hit.ply,
        "fen": decode_board(hit.packed_position).fen(),
        "white_elo": hit.white_elo,
        "black_elo": hit.black_elo,
        "utc_date": hit.utc_date.isoformat(),
        "year": hit.year,
        "result": hit.result,
        "split": hit.split,
        "n_pieces": hit.n_pieces,
    }


def run_search(
    args: argparse.Namespace,
    *,
    store,
    embedder,
) -> dict:
    board = _query_board(args)
    segment = TokenStreamEncoder().encode(board)
    position_filter = _query_filter(args)
    payload = {
        "query": {"fen": board.fen(), "moves": list(args.moves)},
    }
    if args.by == "position":
        vectors = embedder.embed([segment])
        hits = store.search(
            vectors[0],
            limit=args.k,
            filter=position_filter,
            nprobes=args.nprobes,
            refine_factor=args.refine_factor,
            exact=args.exact,
        )
        payload["positions"] = [_position_payload(hit) for hit in hits]
        return payload

    games = search_games(
        store=store,
        embedder=embedder,
        segment=segment,
        n_games=args.n_games,
        candidate_limit=args.k,
        filter=position_filter,
        nprobes=args.nprobes,
        refine_factor=args.refine_factor,
        exact=args.exact,
    )
    payload["games"] = [game.to_dict() for game in games]
    return payload


def format_text(payload: dict) -> str:
    query = payload.get("query", {})
    lines = [f"query: {query.get('fen', '')}"]
    if "games" in payload:
        games = payload["games"]
        if not games:
            lines.append("no matching games")
        for rank, game in enumerate(games, start=1):
            lines.append(
                f"{rank:>3}. {game['game_id']}  score={game['score']:.3f}  "
                f"ply={game['ply']}  "
                f"elo={game['white_elo']}/{game['black_elo']}  "
                f"{game['result']}  {game['split']}  {game['year']}  "
                f"{game['fen']}"
            )
        return "\n".join(lines)
    positions = payload.get("positions", [])
    if not positions:
        lines.append("no matching positions")
    for rank, position in enumerate(positions, start=1):
        lines.append(
            f"{rank:>3}. {position['game_id']}  score={position['score']:.3f}  "
            f"ply={position['ply']}  {position['fen']}"
        )
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    from src.search.embedder import CheckpointEmbedder
    from src.search.lancedb_store import LancePositionStore

    embedder = CheckpointEmbedder(args.checkpoint, device=args.device)
    store = LancePositionStore(
        Path(args.index),
        dim=embedder.dim,
        create=False,
        table_name=args.table,
    )
    payload = run_search(args, store=store, embedder=embedder)
    store.close()
    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        print(format_text(payload))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
