"""P4: zero-ablation (layer pruning) probe for an encoder-depth checkpoint.

Skip block ``i`` (the residual stream carries through) and measure the causal
drop in masked-token accuracy and in the material probe. Unlike the drift
probe, this measures what the block actually contributes to the product.

Zero-ablation is a distribution shift rather than a trained ablation, so treat
it as a screen, not proof. See ``docs/plan_depth_probing.md`` (P4).
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch

from src.evaluation.eval_suite import (
    board_metadata,
    material_probe_ci,
    sample_segments,
)
from src.evaluation.probes import (
    count_parameters,
    depth_accuracy,
    embed_segments_at_depth,
    load_checkpoint_model,
    select_window_batches,
)
from src.training.window_dataset import load_game_pool


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--windows", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--pool", type=int, default=2048)
    parser.add_argument("--games", type=int, default=8192)
    parser.add_argument("--prefetch-shards", type=int, default=2)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--mode", default="board")
    parser.add_argument("--device", default="cpu", choices=("cpu", "cuda", "auto"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    torch.set_num_threads(min(8, os.cpu_count() or 1))
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    checkpoint, model = load_checkpoint_model(args.checkpoint, device)
    print(
        f"step={checkpoint['step']} blocks={model.config.n_layers} "
        f"params={count_parameters(model) / 1e6:.1f}M device={device}",
        flush=True,
    )

    games = load_game_pool(
        args.data_dir, args.games, prefetch_shards=args.prefetch_shards
    )
    print(f"games={len(games)}", flush=True)
    batches = select_window_batches(
        games,
        windows=args.windows,
        batch_size=args.batch_size,
        seed=args.seed,
        mode=args.mode,
        device=device,
    )
    segments = sample_segments(games, args.pool, args.seed)
    _, _, materials = board_metadata(segments)

    baseline = depth_accuracy(model, batches)
    print(json.dumps({"block": "none", **baseline}), flush=True)
    rows = []
    for index in range(model.config.n_layers):
        metrics = depth_accuracy(model, batches, skip=index)
        embeds = embed_segments_at_depth(
            model, segments, device, skip=index, batch_size=64, bf16=False
        )
        material = material_probe_ci(embeds, materials, args.seed, n_seeds=5)
        row = {
            "block": index,
            **metrics,
            "acc_drop": baseline["masked_acc"] - metrics["masked_acc"],
            "test_r2_material": material["test_r2_material"],
            "r2_ci_low": material["test_r2_material_ci_low"],
            "r2_ci_high": material["test_r2_material_ci_high"],
        }
        rows.append(row)
        print(json.dumps(row), flush=True)

    payload = {
        "probe": "layer_pruning",
        "checkpoint": args.checkpoint,
        "step": int(checkpoint["step"]),
        "n_layers": int(model.config.n_layers),
        "params": count_parameters(model),
        "windows": args.windows,
        "pool": args.pool,
        "seed": args.seed,
        "mode": args.mode,
        "baseline": baseline,
        "rows": rows,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(payload, indent=2))
    return


if __name__ == "__main__":
    main()
