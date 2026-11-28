"""P2: early-exit accuracy curve for an encoder-depth checkpoint.

Truncate the encoder after block ``k`` (1..L), apply ``ln_final``, pool the
window through the trained bottleneck and decode it. Every ``k`` scores the
same masked tokens, so the curve is paired: the first ``k`` within noise of
the full model is the depth at which decodable information stops growing.

See ``docs/plan_depth_probing.md`` (P2).
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch

from src.evaluation.probes import (
    count_parameters,
    depth_accuracy,
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
    parser.add_argument("--games", type=int, default=8192)
    parser.add_argument("--prefetch-shards", type=int, default=2)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--mode", default="board")
    parser.add_argument("--max-seq-len", type=int, default=2048)
    parser.add_argument("--max-segments", type=int, default=30)
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
        max_seq_len=args.max_seq_len,
        max_segments=args.max_segments,
        device=device,
    )

    rows = []
    for depth in range(1, model.config.n_layers + 1):
        metrics = depth_accuracy(model, batches, depth=depth)
        rows.append({"depth": depth, **metrics})
        print(json.dumps(rows[-1]), flush=True)

    full = depth_accuracy(model, batches)
    payload = {
        "probe": "early_exit",
        "checkpoint": args.checkpoint,
        "step": int(checkpoint["step"]),
        "n_layers": int(model.config.n_layers),
        "d_model": int(model.config.d_model),
        "params": count_parameters(model),
        "windows": args.windows,
        "seed": args.seed,
        "mode": args.mode,
        "rows": rows,
        "full_model": full,
        "best_depth_within_0_3pt": next(
            row["depth"]
            for row in rows
            if full["masked_acc"] - row["masked_acc"] <= 0.003
        ),
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(payload, indent=2))
    return


if __name__ == "__main__":
    main()
