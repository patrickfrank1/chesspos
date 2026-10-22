from __future__ import annotations

import argparse
import json
from pathlib import Path

import chess
import chess.pgn
import torch

from src.dataset.puzzle_variants import collate_variants, pack_segments
from src.dataset.token_stream import TokenStreamEncoder
from src.modeling.puzzle_classifier import (
    FrozenEncoderClassifier,
    PoolingClassifier,
    PoolingClassifierConfig,
)
from src.modeling.transformer import MaskedStreamTransformer, TransformerConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Score positions with a trained puzzle-theme classifier"
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--fen", default=None, help="Score a single FEN position")
    parser.add_argument(
        "--pgn", default=None, help="Score every position in a PGN game"
    )
    parser.add_argument(
        "--window",
        type=int,
        default=1,
        help="Number of preceding positions to include (1 = single position)",
    )
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--device", default=None)
    return parser.parse_args()


def load_classifier(path: str, device: torch.device) -> FrozenEncoderClassifier:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    encoder = MaskedStreamTransformer(TransformerConfig(**checkpoint["encoder_config"]))
    encoder.load_state_dict(checkpoint["encoder_state"])
    head = PoolingClassifier(PoolingClassifierConfig(**checkpoint["head_config"]))
    head.load_state_dict(checkpoint["head_state"])
    model = FrozenEncoderClassifier(encoder, head)
    return model.to(device).eval()


def _windows(boards: list[chess.Board], window: int) -> list[list[chess.Board]]:
    if window < 1:
        raise ValueError("window must be at least 1")
    return [
        boards[max(0, index - window + 1) : index + 1] for index in range(len(boards))
    ]


@torch.no_grad()
def score_windows(
    model: FrozenEncoderClassifier,
    windows: list[list[chess.Board]],
    device: torch.device,
) -> list[float]:
    encoder = TokenStreamEncoder()
    samples = []
    for boards in windows:
        segments = [encoder.encode(board) for board in boards]
        tokens, segment_ids = pack_segments(segments)
        samples.append(
            {
                "tokens": tokens,
                "segment_ids": segment_ids,
                "label": 0,
                "puzzle_id": "",
            }
        )
    batch = collate_variants(samples)
    logits = model(
        batch["tokens"].to(device),
        batch["segment_ids"].to(device),
        batch["padding_mask"].to(device),
    )
    return torch.sigmoid(logits).cpu().tolist()


def score_positions(
    model: FrozenEncoderClassifier,
    boards: list[chess.Board],
    device: torch.device,
    window: int = 1,
) -> list[float]:
    return score_windows(model, _windows(boards, window), device)


def classify_fen(
    model: FrozenEncoderClassifier, fen: str, device: torch.device
) -> float:
    return score_positions(model, [chess.Board(fen)], device)[0]


def game_positions(game: chess.pgn.Game) -> list[tuple[int, str | None, chess.Board]]:
    board = game.board()
    positions = [(0, None, board.copy())]
    for ply, move in enumerate(game.mainline_moves(), start=1):
        board.push(move)
        positions.append((ply, move.uci(), board.copy()))
    return positions


def classify_pgn(
    model: FrozenEncoderClassifier,
    path: str | Path,
    device: torch.device,
    window: int = 1,
) -> list[dict]:
    with open(path) as handle:
        game = chess.pgn.read_game(handle)
    if game is None:
        raise ValueError(f"No game found in {path}")
    positions = game_positions(game)
    scores = score_positions(
        model, [board for _, _, board in positions], device, window=window
    )
    return [
        {
            "ply": ply,
            "move": move,
            "fen": board.fen(),
            "probability": score,
        }
        for (ply, move, board), score in zip(positions, scores)
    ]


def _emit(payload: dict, threshold: float | None) -> None:
    if threshold is not None:
        payload = {**payload, "is_theme": payload["probability"] >= threshold}
    print(json.dumps(payload))
    return


def main() -> int:
    args = parse_args()
    if args.fen is None and args.pgn is None:
        print("Provide --fen or --pgn")
        return 1
    device = torch.device(
        args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    )
    model = load_classifier(args.checkpoint, device)

    if args.fen is not None:
        _emit(
            {"fen": args.fen, "probability": classify_fen(model, args.fen, device)},
            args.threshold,
        )
    if args.pgn is not None:
        for payload in classify_pgn(model, args.pgn, device, window=args.window):
            _emit(payload, args.threshold)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
