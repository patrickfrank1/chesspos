"""P1: consecutive-block residual drift for an encoder-depth checkpoint.

For every encoder block, report the per-window relative change
``||h_l - h_{l-1}|| / ||h_{l-1}||`` and ``cos(h_l, h_{l-1})`` over the same
paired window pool. Near-identity blocks show ``rel_delta -> 0`` and
``cos -> 1``: depth that burns compute without changing the representation.

See ``docs/plan_depth_probing.md`` (P1).
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch

from src.evaluation.probes import (
    count_parameters,
    load_checkpoint_model,
    residual_drift,
    select_window_batches,
)
from src.training.window_dataset import load_game_pool


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--windows", type=int, default=32)
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
    rows = residual_drift(model, batches)
    for row in rows:
        print(
            f"block={row['block']} rel_delta={row['rel_delta_mean']:.3f}"
            f"±{row['rel_delta_std']:.3f} cos={row['cos_mean']:.3f}",
            flush=True,
        )

    payload = {
        "probe": "residual_drift",
        "checkpoint": args.checkpoint,
        "step": int(checkpoint["step"]),
        "n_layers": int(model.config.n_layers),
        "d_model": int(model.config.d_model),
        "d_ff": int(model.config.d_ff),
        "latent_dim": int(model.config.latent_dim),
        "params": count_parameters(model),
        "windows": args.windows,
        "seed": args.seed,
        "mode": args.mode,
        "rows": rows,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(payload, indent=2))
    return


if __name__ == "__main__":
    main()
