from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

import numpy as np

from src.chess_eval.attribution import (
    METHODS,
    OCCLUSION_MODES,
    attribute_position,
    deletion_curve,
    occlusion_replacement,
)
from src.chess_eval.backend import (
    ModelHandle,
    forward_tokens,
    load_checkpoint_handle,
    model_batch,
    replay,
)
from src.chess_eval.samples import DEFAULT_PUZZLE_DIR, load_puzzle_samples
from src.dataset.token_stream import EMPTY


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare attribution methods with deletion curves"
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--task", choices=("score", "sacrifice"), default=None)
    parser.add_argument(
        "--puzzle-data-dir", default=str(DEFAULT_PUZZLE_DIR), help="Puzzle shards"
    )
    parser.add_argument("--limit", type=int, default=12, help="Positions to explain")
    parser.add_argument("--samples", type=int, default=64, help="KernelSHAP samples")
    parser.add_argument("--random-trials", type=int, default=2)
    parser.add_argument(
        "--steps",
        type=int,
        default=16,
        help="Deletion steps scored per curve (an early prefix)",
    )
    parser.add_argument("--occlusion", choices=OCCLUSION_MODES, default="mask")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default=None)
    parser.add_argument("--output", default=None)
    return parser.parse_args(argv)


def _normalised_impact(curve: Sequence[float], steps: int = 16) -> float:
    """Mean absolute output change over the first ``steps`` deletions.

    Only an early prefix is scored: once most squares are masked the model
    saturates, and both ranked and random deletion converge on that limit.
    """
    values = np.asarray(curve, dtype=np.float64)
    base = float(values[0])
    scale = max(abs(base), 1e-3)
    upto = min(values.size - 1, max(1, steps))
    return float(np.mean(np.abs(values[1 : upto + 1] - base)) / scale)


def _position_impact(
    handle: ModelHandle,
    fen: str,
    method: str,
    device: str | None,
    samples: int,
    random_trials: int,
    seed: int,
    occlusion: str,
    steps: int,
) -> tuple[float, float]:
    boards, _ = replay(fen, [])
    batch = model_batch(handle, boards)

    def score_batch(rows: np.ndarray) -> np.ndarray:
        return forward_tokens(handle, batch, rows, device)

    result = attribute_position(
        handle,
        boards,
        device=device,
        method=method,
        occlusion=occlusion,
        samples=samples,
        seed=seed,
    )
    positions = batch.square_positions[-1]
    order = np.argsort(-np.abs(np.asarray(result.squares))).tolist()
    # Each method is scored under its own counterfactual: occlusion uses the
    # requested replacement, KernelSHAP uses the empty board it is defined on.
    replacement = EMPTY if method == "shap" else occlusion_replacement(occlusion)
    curve = deletion_curve(
        score_batch, batch.tokens, positions, order, replacement=replacement
    )

    rng = np.random.default_rng(seed)
    random_impacts: list[float] = []
    for _ in range(max(0, random_trials)):
        permutation = rng.permutation(positions.size).tolist()
        random_curve = deletion_curve(
            score_batch, batch.tokens, positions, permutation, replacement=replacement
        )
        random_impacts.append(_normalised_impact(random_curve, steps))
    random_impact = float(np.mean(random_impacts)) if random_impacts else 0.0
    return _normalised_impact(curve, steps), random_impact


def attribution_report(
    handle: ModelHandle,
    fens: Sequence[str],
    device: str | None = None,
    samples: int = 64,
    random_trials: int = 2,
    seed: int = 0,
    occlusion: str = "mask",
    steps: int = 16,
) -> dict:
    """Measure deletion-curve faithfulness of every method on the same positions."""
    methods: dict[str, dict] = {}
    for method in METHODS:
        attributed: list[float] = []
        random: list[float] = []
        ratios: list[float] = []
        for index, fen in enumerate(fens):
            attributed_impact, random_impact = _position_impact(
                handle,
                fen,
                method,
                device,
                samples,
                random_trials,
                seed + index,
                occlusion,
                steps,
            )
            attributed.append(attributed_impact)
            random.append(random_impact)
            if random_impact > 0:
                ratios.append(attributed_impact / random_impact)
        methods[method] = {
            "faithfulness": float(np.mean(attributed)) if attributed else 0.0,
            "random": float(np.mean(random)) if random else 0.0,
            # Per-position ratios keep one high-variance board from deciding
            # the comparison.
            "lift": float(np.mean(ratios)) if ratios else None,
            "median_lift": float(np.median(ratios)) if ratios else None,
        }
    return {
        "task": handle.info.task,
        "checkpoint": str(handle.info.path),
        "positions": len(fens),
        "occlusion": occlusion,
        "samples": samples,
        "random_trials": random_trials,
        "steps": steps,
        "seed": seed,
        "methods": methods,
    }


def select_positions(data_dir: str | Path, limit: int, seed: int) -> list[str]:
    samples = load_puzzle_samples(data_dir, max_puzzles=max(limit, 2), seed=seed)
    if not samples:
        return []
    if len(samples) <= limit:
        return [sample.fen for sample in samples]
    step = max(1, len(samples) // limit)
    return [samples[index].fen for index in range(0, len(samples), step)][:limit]


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    handle = load_checkpoint_handle(args.checkpoint, task=args.task, device=args.device)
    fens = select_positions(args.puzzle_data_dir, args.limit, args.seed)
    if not fens:
        raise ValueError(f"no puzzle positions found in {args.puzzle_data_dir}")
    report = attribution_report(
        handle,
        fens,
        device=args.device,
        samples=args.samples,
        random_trials=args.random_trials,
        seed=args.seed,
        occlusion=args.occlusion,
        steps=args.steps,
    )
    for method, entry in report["methods"].items():
        lift = entry["lift"]
        lift_text = f"{lift:.2f}" if lift is not None else "n/a"
        print(
            f"{method:>9}: faithfulness={entry['faithfulness']:.4f} "
            f"random={entry['random']:.4f} lift={lift_text} "
            f"median_lift={entry['median_lift']:.2f}"
        )
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, indent=2) + "\n")
        print(f"wrote {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
