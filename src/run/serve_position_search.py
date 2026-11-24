"""CLI: serve the position-search web UI."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Serve a local web UI for searching a position index"
    )
    parser.add_argument("--index", required=True, help="Position index directory")
    parser.add_argument("--checkpoint", required=True, help="Frozen encoder checkpoint")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--table", default="positions")
    parser.add_argument("--device", default=None)
    return parser.parse_args(argv)


def build_app(args: argparse.Namespace):
    from src.search.embedder import CheckpointEmbedder
    from src.search.factory import open_position_store
    from src.web.app import create_app

    embedder = CheckpointEmbedder(args.checkpoint, device=args.device)
    store = open_position_store(
        Path(args.index),
        dim=embedder.dim,
        table_name=args.table,
    )
    return create_app(store=store, embedder=embedder)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    import uvicorn

    uvicorn.run(build_app(args), host=args.host, port=args.port)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
