from __future__ import annotations

import argparse
from typing import Sequence

from src.chess_eval.app import create_app


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Serve the interactive chesspos position evaluator"
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument(
        "--model-root",
        action="append",
        default=None,
        help="Checkpoint root to scan; repeat for multiple roots",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Torch device, e.g. cpu or cuda (default: auto)",
    )
    parser.add_argument(
        "--nnue-cache-dir",
        default=None,
        help="Where to cache the Stockfish NNUE file",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    app = create_app(
        args.model_root,
        device=args.device,
        nnue_cache_dir=args.nnue_cache_dir,
    )
    import uvicorn

    print(
        f"chesspos evaluator: http://{args.host}:{args.port} "
        f"({len(app.state.catalog)} checkpoints, device={app.state.device})",
        flush=True,
    )
    uvicorn.run(app, host=args.host, port=args.port)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
