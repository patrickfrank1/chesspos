from __future__ import annotations

import argparse
import json
from typing import Sequence

from src.chess_eval.attribution import (
    DEFAULT_SHAP_SAMPLES,
    METHODS,
    OCCLUSION_MODES,
    attribute_position,
)
from src.chess_eval.backend import load_checkpoint_handle, replay
from src.chess_eval.puzzles import background_boards
from src.chess_eval.samples import DEFAULT_PUZZLE_DIR


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Explain a chesspos score or sacrifice prediction per square"
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--fen", default=None, help="Position to explain")
    parser.add_argument(
        "--moves",
        default="",
        help="Comma-separated UCI moves applied to --fen before explaining",
    )
    parser.add_argument(
        "--task",
        choices=("score", "sacrifice"),
        default=None,
        help="Task of the checkpoint (default: infer from the checkpoint)",
    )
    parser.add_argument("--method", choices=METHODS, default="occlusion")
    parser.add_argument("--occlusion", choices=OCCLUSION_MODES, default="mask")
    parser.add_argument("--samples", type=int, default=DEFAULT_SHAP_SAMPLES)
    parser.add_argument(
        "--background-dir",
        default=str(DEFAULT_PUZZLE_DIR),
        help="Puzzle shards providing the interventional SHAP background",
    )
    parser.add_argument(
        "--background-count",
        type=int,
        default=4,
        help="Real positions used as the SHAP background (0 disables it)",
    )
    parser.add_argument("--top", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default=None)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.fen is None:
        print("Provide --fen")
        return 1
    handle = load_checkpoint_handle(args.checkpoint, task=args.task, device=args.device)
    moves = [move.strip() for move in args.moves.split(",") if move.strip()]
    boards, _ = replay(args.fen, moves)
    try:
        background = background_boards(
            args.background_dir, count=args.background_count, seed=args.seed
        )
    except Exception:
        background = ()
    result = attribute_position(
        handle,
        boards,
        device=args.device,
        method=args.method,
        occlusion=args.occlusion,
        samples=args.samples,
        seed=args.seed,
        background=background,
    )
    print(json.dumps(result.to_dict(top=args.top)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
