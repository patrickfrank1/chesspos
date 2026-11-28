"""P3: latent richness vs encoder depth (PCA-truncation probe).

At every early-exit depth, encode a paired window pool and a board pool, then
measure how much of each of the two products survives PCA truncation:

* reconstruction accuracy when the decoder only sees the top-``k`` window
  latent components;
* the ridge material probe when the board embedding is truncated the same way;
* effective rank at 99 % / 99.9 % explained variance.

A deeper encoder that leaves the truncated curves and the effective rank
unchanged has not enriched the embedding downstream tasks consume.

See ``docs/plan_depth_probing.md`` (P3).
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
    effective_rank,
    embed_segments_at_depth,
    encoder_states,
    latent_from_state,
    load_checkpoint_model,
    pca_basis,
    reconstruction_accuracy,
    select_window_batches,
    truncate,
)
from src.training.window_dataset import load_game_pool


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--basis-windows", type=int, default=1024)
    parser.add_argument("--accuracy-windows", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--pool", type=int, default=4096)
    parser.add_argument("--games", type=int, default=8192)
    parser.add_argument("--prefetch-shards", type=int, default=2)
    parser.add_argument("--ks", default="")
    parser.add_argument("--depths", default="")
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
    width = int(model.config.latent_dim)
    default_ks = [width, 512, 384, 256, 192, 128, 96, 64, 48, 32]
    ks = (
        [int(value) for value in args.ks.split(",") if value] if args.ks else default_ks
    )
    ks = sorted({k for k in ks if 0 < k <= width}, reverse=True)
    depths = (
        [int(value) for value in args.depths.split(",") if value]
        if args.depths
        else list(range(1, model.config.n_layers + 1))
    )
    print(
        f"step={checkpoint['step']} blocks={model.config.n_layers} latent={width} "
        f"params={count_parameters(model) / 1e6:.1f}M ks={ks} depths={depths}",
        flush=True,
    )

    games = load_game_pool(
        args.data_dir, args.games, prefetch_shards=args.prefetch_shards
    )
    print(f"games={len(games)}", flush=True)
    batches = select_window_batches(
        games,
        windows=args.basis_windows,
        batch_size=args.batch_size,
        seed=args.seed,
        mode=args.mode,
        device=device,
    )
    segments = sample_segments(games, args.pool, args.seed)
    _, _, materials = board_metadata(segments)

    depth_rows = []
    for depth in depths:
        latents = []
        for batch in batches:
            states = encoder_states(
                model,
                batch["tokens"],
                batch["segment_ids"],
                batch["padding_mask"],
            )
            latents.append(
                latent_from_state(model, states[depth], batch["padding_mask"]).float()
            )
        window_matrix = torch.cat(latents, dim=0)
        mean, components, explained = pca_basis(window_matrix)
        keep = set(range(min(args.accuracy_windows, window_matrix.shape[0])))
        accuracy_batches = []
        offset = 0
        for batch, latent in zip(batches, latents):
            mask = torch.tensor(
                [offset + i in keep for i in range(latent.shape[0])], dtype=torch.bool
            )
            if mask.any():
                accuracy_batches.append((batch, latent[mask]))
            offset += latent.shape[0]

        embeds = embed_segments_at_depth(
            model, segments, device, depth=depth, batch_size=64, bf16=False
        )
        embed_mean, embed_components, embed_explained = pca_basis(embeds)

        rows = []
        for k in ks:
            counts = {}
            for batch, latent in accuracy_batches:
                batch_counts = reconstruction_accuracy(
                    model, batch, truncate(latent, mean, components, k)
                )
                for key, value in batch_counts.items():
                    counts[key] = counts.get(key, 0) + value
            material = material_probe_ci(
                truncate(embeds, embed_mean, embed_components, k).float(),
                materials,
                args.seed,
                n_seeds=5,
            )
            row = {
                "k": k,
                "window_var_explained": float(explained[:k].sum()),
                "embed_var_explained": float(embed_explained[:k].sum()),
                "mlm_board_acc": counts.get("correct", 0)
                / max(1, counts.get("total", 0)),
                "test_r2_material": material["test_r2_material"],
                "r2_ci_low": material["test_r2_material_ci_low"],
                "r2_ci_high": material["test_r2_material_ci_high"],
            }
            rows.append(row)
            print(json.dumps({"depth": depth, **row}), flush=True)

        depth_rows.append(
            {
                "depth": depth,
                "effective_rank_99": effective_rank(explained, 0.99),
                "effective_rank_99_9": effective_rank(explained, 0.999),
                "embed_effective_rank_99": effective_rank(embed_explained, 0.99),
                "embed_effective_rank_99_9": effective_rank(embed_explained, 0.999),
                "rows": rows,
            }
        )

    payload = {
        "probe": "latent_truncation",
        "checkpoint": args.checkpoint,
        "step": int(checkpoint["step"]),
        "n_layers": int(model.config.n_layers),
        "latent_dim": width,
        "params": count_parameters(model),
        "basis_windows": args.basis_windows,
        "accuracy_windows": args.accuracy_windows,
        "pool": args.pool,
        "seed": args.seed,
        "mode": args.mode,
        "depths": depth_rows,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(payload, indent=2))
    return


if __name__ == "__main__":
    main()
