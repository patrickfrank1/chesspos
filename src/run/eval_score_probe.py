from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

import torch

from src.dataset.game_outcomes import (
    GameOutcome,
    ScoreDataset,
    ScorePlan,
    build_score_plan,
    load_game_records,
)
from src.modeling.factory import build_model, model_config_from_dict
from src.modeling.score_head import (
    FrozenEncoderScore,
    ScoreHead,
    ScoreHeadConfig,
)
from src.run.train_score_probe import (
    build_score_loader,
    load_remote_game_records,
    predict,
    save_predictions,
)
from src.training.score_metrics import score_metrics


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a trained frozen-encoder score probe"
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data-dir", default="data/processed/games")
    parser.add_argument("--remote-repo", default=None)
    parser.add_argument("--remote-split", default="test")
    parser.add_argument(
        "--remote-shard-patterns",
        default=None,
        help="Comma-separated fnmatch patterns, each receiving an equal share",
    )
    parser.add_argument("--remote-cache-dir", default="data/hf-shards")
    parser.add_argument("--output-dir", default="data/eval/score_probe")
    parser.add_argument("--window", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--amp", choices=("none", "bf16"), default="none")
    parser.add_argument("--device", default=None)
    parser.add_argument("--seed", type=int, default=17)
    return parser.parse_args(argv)


def load_score_probe(path: str | Path, device: torch.device) -> FrozenEncoderScore:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"probe checkpoint not found: {path}")
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    encoder = build_model(model_config_from_dict(checkpoint["encoder_config"]))
    encoder.load_state_dict(checkpoint["encoder_state"])
    head = ScoreHead(ScoreHeadConfig(**checkpoint["head_config"]))
    head.load_state_dict(checkpoint["head_state"])
    model = FrozenEncoderScore(
        encoder,
        head,
        feature=checkpoint.get("feature", "native"),
        unfreeze_last_block=checkpoint.get("unfreeze_last_block", False),
    )
    return model.to(device).eval()


def _remote_records(args: argparse.Namespace) -> list[GameOutcome]:
    patterns = (
        [pattern.strip() for pattern in args.remote_shard_patterns.split(",")]
        if args.remote_shard_patterns
        else [None]
    )
    per_pattern = None if args.limit is None else max(1, args.limit // len(patterns))
    records: list[GameOutcome] = []
    for index, pattern in enumerate(patterns):
        remaining = None if args.limit is None else args.limit - len(records)
        if remaining is not None and remaining <= 0:
            break
        limit = remaining if per_pattern is None else min(per_pattern, remaining)
        records.extend(
            load_remote_game_records(
                args.remote_repo,
                args.remote_split,
                limit=limit,
                seed=args.seed + index,
                cache_dir=args.remote_cache_dir,
                shard_pattern=pattern,
            )
        )
    return records


def resolve_records(args: argparse.Namespace) -> list[GameOutcome]:
    if args.remote_repo is not None:
        return _remote_records(args)
    return load_game_records(args.data_dir, limit=args.limit)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    device = torch.device(
        args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    )
    model = load_score_probe(args.checkpoint, device)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    window = args.window or int(checkpoint.get("window", 1))
    include_state_features = bool(checkpoint.get("include_state_features", False))

    records = resolve_records(args)
    if not records:
        raise ValueError("no games loaded for evaluation")
    dataset = ScoreDataset(
        records,
        _all_positions(records),
        window=window,
        include_state_features=include_state_features,
    )
    loader = build_score_loader(
        dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        device=device,
    )
    outputs = predict(model, loader, device, amp=args.amp)
    metrics = score_metrics(outputs["predictions"], outputs["targets"])

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    save_predictions(output_dir / "predictions.npz", outputs)
    (output_dir / "metrics.json").write_text(
        json.dumps(
            {
                "checkpoint": str(args.checkpoint),
                "feature": model.feature,
                "window": window,
                "include_state_features": include_state_features,
                "source": args.remote_repo or args.data_dir,
                "split": args.remote_split if args.remote_repo else "local",
                "games": len(records),
                "positions": int(outputs["predictions"].size),
                "metrics": metrics,
            },
            indent=2,
        )
        + "\n"
    )
    print(
        f"positions={outputs['predictions'].size} mse={metrics['mse']:.4f} "
        f"r2={metrics['r2']:.4f} mae={metrics['mae']:.4f} "
        f"auroc_decisive={metrics['auroc_decisive']:.4f}"
    )
    return 0


def _all_positions(records: Sequence[GameOutcome]) -> ScorePlan:
    max_ply = max((len(record.segments) for record in records), default=0)
    return build_score_plan(
        records, positions_per_game=None, max_ply=max(1, max_ply), seed=0
    )


if __name__ == "__main__":
    raise SystemExit(main())
