"""Evaluate the bottleneck embeddings of an attention-autoencoder checkpoint.

Runs the training-time ``full_eval`` suite at ``--suite-pool`` (default 1,024)
and then repeats the embedding probes on a larger fixed pool (default 4,096)
for tighter confidence intervals. Works on CPU or CUDA.

Example:

    python scripts/eval_bottleneck_embeddings.py \
        --checkpoint data/models/ae_20m_100k/best_step80000.pt \
        --data-dir data/hf-cache/.../snapshots/<hash>/test \
        --out-dir data/eval/ae_20m_80000
"""

from __future__ import annotations

import argparse
import json
import os
import time
from contextlib import contextmanager
from pathlib import Path

import torch

from src.evaluation.eval_suite import (
    board_metadata,
    embed_segments,
    full_eval,
    material_probe_ci,
    neighbor_examples,
    neighbor_stats,
    phase_stats_ci,
    sample_segments,
    side_to_move_probe,
    write_neighbor_examples,
)
from src.modeling.factory import build_model, model_config_from_dict
from src.training.window_dataset import load_game_pool


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"))
    parser.add_argument("--games", type=int, default=8192)
    parser.add_argument("--suite-pool", type=int, default=1024)
    parser.add_argument("--probe-pool", type=int, default=4096)
    parser.add_argument("--windows", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--counterfactual-n", type=int, default=256)
    parser.add_argument("--probe-seeds", type=int, default=10)
    parser.add_argument("--bootstraps", type=int, default=50)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--skip-suite", action="store_true")
    return parser.parse_args()


@contextmanager
def stage(name: str, timings: dict[str, float]):
    start = time.perf_counter()
    yield
    timings[name] = round(time.perf_counter() - start, 2)
    print(f"{name:34s} {timings[name]:8.2f}s", flush=True)


def as_jsonable(metrics: dict) -> dict:
    return {
        key: float(value) if hasattr(value, "item") else value
        for key, value in metrics.items()
    }


def load_model(path: str, device: torch.device):
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    model = build_model(model_config_from_dict(checkpoint["model_config"]))
    model.load_state_dict(checkpoint["model_state"])
    return checkpoint, model.to(device).eval()


def main() -> None:
    args = parse_args()
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    if device.type == "cpu":
        torch.set_num_threads(min(8, os.cpu_count() or 1))
    bf16 = device.type == "cuda"
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    timings: dict[str, float] = {}

    with stage("load checkpoint", timings):
        checkpoint, model = load_model(args.checkpoint, device)
    print(f"step={checkpoint['step']} device={device.type}", flush=True)

    with stage(f"load_game_pool({args.games})", timings):
        games = load_game_pool(args.data_dir, args.games, prefetch_shards=2)
    print(f"val games={len(games)}", flush=True)

    if not args.skip_suite:
        with stage(f"full_eval pool={args.suite_pool}", timings):
            metrics = full_eval(
                model,
                games,
                device,
                windows=args.windows,
                batch_size=args.batch_size,
                max_seq_len=2048,
                max_segments=30,
                pool_size=args.suite_pool,
                counterfactual_n=args.counterfactual_n,
                seed=args.seed,
                plot_dir=str(out),
                examples_dir=str(out),
                bf16=bf16,
            )
        (out / "full_eval_metrics.json").write_text(
            json.dumps(as_jsonable(metrics), indent=2, sort_keys=True)
        )

    with stage(f"embed {args.probe_pool} positions", timings):
        segments = sample_segments(games, args.probe_pool, args.seed)
        embeds = embed_segments(model, segments, device, batch_size=64, bf16=bf16)
        boards, piece_counts, materials = board_metadata(segments)

    probes: dict[str, float] = {}
    with stage("neighbor_stats", timings):
        probes.update(neighbor_stats(embeds, piece_counts, materials, args.seed))
    with stage(f"material_probe_ci({args.probe_seeds})", timings):
        probes.update(
            material_probe_ci(embeds, materials, args.seed, n_seeds=args.probe_seeds)
        )
    with stage("side_to_move_probe", timings):
        probes.update(side_to_move_probe(embeds, boards, args.seed))
    with stage(f"phase_stats_ci({args.bootstraps})", timings):
        probes.update(
            phase_stats_ci(
                embeds,
                piece_counts,
                args.seed,
                n_bootstrap=args.bootstraps,
                device=device,
            )
        )
    with stage("neighbor_examples", timings):
        examples = neighbor_examples(embeds, boards, materials, args.seed)
        write_neighbor_examples(examples, str(out / "large_pool"))

    probes = as_jsonable(probes)
    (out / "large_pool_metrics.json").write_text(
        json.dumps(probes, indent=2, sort_keys=True)
    )
    (out / "timings.json").write_text(json.dumps(timings, indent=2))
    print(json.dumps(probes, indent=2, sort_keys=True), flush=True)
    print(f"wrote {out}", flush=True)
    return


if __name__ == "__main__":
    main()
