from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch

from src.evaluation.eval_suite import (
    board_metadata,
    counterfactual_stats,
    embed_segments,
    material_probe,
    mlm_breakdown,
    neighbor_stats,
    phase_stats,
    piece_prior,
    sample_segments,
    side_to_move_probe,
)
from src.modeling.transformer import MaskedStreamTransformer, TransformerConfig
from src.training.window_dataset import _load_game_segments


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate masked stream transformer")
    parser.add_argument("--checkpoint", default="models/run_full/best.pt")
    parser.add_argument("--data-dir", default="data/processed/test")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--windows", type=int, default=24)
    parser.add_argument("--max-seq-len", type=int, default=2048)
    parser.add_argument("--max-segments", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--n-positions", type=int, default=3000)
    parser.add_argument("--n-queries", type=int, default=3)
    parser.add_argument("--n-counterfactual", type=int, default=300)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--out-dir", default="eval_out")
    return parser.parse_args()


def load_model(path: str, device: torch.device) -> MaskedStreamTransformer:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    model = MaskedStreamTransformer(TransformerConfig(**checkpoint["model_config"]))
    model.load_state_dict(checkpoint["model_state"])
    return model.to(device).eval()


def print_neighbor_queries(
    embeds: torch.Tensor,
    boards: list,
    materials: np.ndarray,
    n_queries: int,
    seed: int,
) -> None:
    sims = (embeds @ embeds.T).numpy()
    np.fill_diagonal(sims, -np.inf)
    rng = np.random.default_rng(seed)
    for q in rng.choice(len(boards), size=n_queries, replace=False):
        top = np.argsort(-sims[q])[:5]
        print(f"\n  query: {boards[q].fen()}  material={materials[q]:+.0f}")
        for rank, j in enumerate(top, 1):
            print(
                f"    {rank}. sim={sims[q][j]:.3f} material={materials[j]:+.0f}  {boards[j].fen()}"
            )


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)

    print(f"loading checkpoint {args.checkpoint}")
    model = load_model(args.checkpoint, device)
    print("loading games")
    games = []
    for path in sorted(Path(args.data_dir).glob("*.parquet")):
        games.extend(_load_game_segments(path))
    print(f"games={len(games)}")

    print("\n=== 1. MLM accuracy by masking mode ===")
    for mode, stats in mlm_breakdown(
        model,
        games,
        device,
        windows=args.windows,
        batch_size=args.batch_size,
        seed=args.seed,
        max_seq_len=args.max_seq_len,
        max_segments=args.max_segments,
        prior=piece_prior(games),
    ).items():
        print(
            f"  {mode:6s} acc={stats['acc']:.4f} piece={stats['piece_acc']:.4f} "
            f"(prior={stats['prior_baseline']:.4f}) turn={stats['turn_acc']:.4f} "
            f"castle={stats['castle_acc']:.4f}"
        )

    print("\n=== 2. Position embeddings ===")
    segments = sample_segments(games, args.n_positions, args.seed)
    embeds = embed_segments(model, segments, device)
    boards, piece_counts, materials = board_metadata(segments)
    print(f"embedded {len(segments)} positions (d={embeds.shape[1]})")

    print("\n-- nearest neighbours --")
    print_neighbor_queries(embeds, boards, materials, args.n_queries, args.seed)
    for name, value in neighbor_stats(
        embeds, piece_counts, materials, args.seed
    ).items():
        print(f"  {name}: {value:.4f}")

    print("\n-- counterfactual similarities --")
    for name, value in counterfactual_stats(
        model, segments, embeds, boards, device, 256, args.n_counterfactual, args.seed
    ).items():
        print(f"  {name}: {value:.4f}")

    print("\n-- linear probes (ridge) --")
    for name, value in material_probe(embeds, materials, args.seed).items():
        print(f"  {name}: {value:.4f}")
    for name, value in side_to_move_probe(embeds, boards, args.seed).items():
        print(f"  {name}: {value:.4f}")

    print("\n-- phase structure (PCA) --")
    projected, corr1, corr2 = phase_stats(embeds, piece_counts)
    print(f"  pc1_piece_corr: {corr1:.4f}")
    print(f"  pc2_piece_corr: {corr2:.4f}")
    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(
        projected[:, 0], projected[:, 1], c=piece_counts, s=6, cmap="viridis"
    )
    plt.colorbar(scatter, label="pieces on board")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.title("Position embeddings coloured by game phase")
    plt.tight_layout()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_dir / "pca_phase.png", dpi=150)
    plt.close()
    print(f"  plot saved to {out_dir / 'pca_phase.png'}")


if __name__ == "__main__":
    main()
