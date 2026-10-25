from __future__ import annotations

import argparse
import json

import torch

from src.evaluation.retrieval import evaluate_retrieval, load_benchmark
from src.run.train_puzzle_classifier import load_frozen_encoder


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate an encoder on the frozen retrieval benchmark"
    )
    parser.add_argument("--benchmark", required=True)
    parser.add_argument("--revision", default=None)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--device", default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    device = torch.device(
        args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    )
    benchmark = load_benchmark(args.benchmark, revision=args.revision)
    encoder = load_frozen_encoder(args.checkpoint, device)
    metrics = evaluate_retrieval(encoder, benchmark, device, batch_size=args.batch_size)
    print(json.dumps(metrics, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
