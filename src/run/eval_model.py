from __future__ import annotations

import argparse
from pathlib import Path

import chess
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch

from src.dataset.token_stream import (
    CLS,
    PAD,
    PIECE_SQUARE_BASE,
    SEP,
    VOCAB_SIZE,
    TokenStreamEncoder,
)
from src.modeling.masking import IGNORE_INDEX, MaskingConfig
from src.modeling.transformer import MaskedStreamTransformer, TransformerConfig
from src.training.window_dataset import (
    WindowDataset,
    WindowDatasetConfig,
    _load_game_segments,
    collate_windows,
)

MODE_CONFIGS = {
    "random": MaskingConfig(p_random=1.0, p_board=0.0, p_span=0.0),
    "board": MaskingConfig(p_random=0.0, p_board=1.0, p_span=0.0),
    "span": MaskingConfig(p_random=0.0, p_board=0.0, p_span=1.0),
}
PIECE_VALUES = {
    chess.PAWN: 1.0,
    chess.KNIGHT: 3.0,
    chess.BISHOP: 3.0,
    chess.ROOK: 5.0,
    chess.QUEEN: 9.0,
    chess.KING: 0.0,
}


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


def load_games(data_dir: str) -> list[list[np.ndarray]]:
    games: list[list[np.ndarray]] = []
    for path in sorted(Path(data_dir).glob("*.parquet")):
        games.extend(_load_game_segments(path))
    return games


def piece_prior(games: list[list[np.ndarray]]) -> np.ndarray:
    counts = np.zeros(VOCAB_SIZE, dtype=np.int64)
    for game in games:
        for segment in game:
            counts += np.bincount(segment, minlength=VOCAB_SIZE)
    total = counts[PIECE_SQUARE_BASE:].sum()
    prior = np.zeros(VOCAB_SIZE)
    prior[PIECE_SQUARE_BASE:] = counts[PIECE_SQUARE_BASE:] / max(1, total)
    return prior


@torch.no_grad()
def mlm_breakdown(
    model: MaskedStreamTransformer,
    data_dir: str,
    args: argparse.Namespace,
    device: torch.device,
    prior: np.ndarray,
) -> dict[str, dict[str, float]]:
    results = {}
    for mode, masking in MODE_CONFIGS.items():
        dataset = WindowDataset(
            WindowDatasetConfig(
                data_dir=data_dir,
                max_seq_len=args.max_seq_len,
                max_segments=args.max_segments,
                masking=masking,
                seed=args.seed,
            )
        )
        rng = np.random.default_rng(args.seed)
        indices = rng.permutation(len(dataset))[: args.windows]
        counts = {
            "total": 0,
            "correct": 0,
            "piece": 0,
            "piece_correct": 0,
            "scalar": 0,
            "scalar_correct": 0,
        }
        prior_score = 0.0
        for start in range(0, len(indices), args.batch_size):
            samples = [
                dataset[int(i)] for i in indices[start : start + args.batch_size]
            ]
            batch = {k: v.to(device) for k, v in collate_windows(samples).items()}
            output = model(
                batch["tokens"],
                batch["segment_ids"],
                padding_mask=batch["padding_mask"],
                targets=batch["targets"],
            )
            valid = batch["targets"].ne(IGNORE_INDEX)
            predictions = output["logits"].argmax(-1)
            correct = predictions[valid] == batch["targets"][valid]
            targets = batch["targets"][valid]
            piece = targets.ge(PIECE_SQUARE_BASE)
            scalar = ~piece
            counts["total"] += int(valid.sum())
            counts["correct"] += int(correct.sum())
            counts["piece"] += int(piece.sum())
            counts["piece_correct"] += int(correct[piece].sum())
            counts["scalar"] += int(scalar.sum())
            counts["scalar_correct"] += int(correct[scalar].sum())
            for token, ok in zip(targets[piece].tolist(), correct[piece].tolist()):
                prior_score += float(prior[token]) if not ok else 1.0
        results[mode] = {
            "acc": counts["correct"] / max(1, counts["total"]),
            "piece_acc": counts["piece_correct"] / max(1, counts["piece"]),
            "scalar_acc": counts["scalar_correct"] / max(1, counts["scalar"]),
            "prior_baseline": prior_score / max(1, counts["piece"]),
        }
    return results


def sample_segments(
    games: list[list[np.ndarray]], n: int, seed: int
) -> list[np.ndarray]:
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(games))
    segments: list[np.ndarray] = []
    for index in order:
        game = games[int(index)]
        segments.append(game[int(rng.integers(len(game)))])
        if len(segments) >= n:
            break
    return segments


def collate_segments(segments: list[np.ndarray]) -> dict[str, torch.Tensor]:
    lengths = [len(s) + 2 for s in segments]
    max_len = max(lengths)
    tokens = np.full((len(segments), max_len), PAD, dtype=np.int64)
    for i, segment in enumerate(segments):
        tokens[i, 0] = CLS
        tokens[i, 1 : 1 + len(segment)] = segment
        tokens[i, 1 + len(segment)] = SEP
    segment_ids = np.cumsum(tokens == SEP, axis=1)
    return {
        "tokens": torch.from_numpy(tokens),
        "segment_ids": torch.from_numpy(segment_ids),
        "padding_mask": torch.from_numpy(tokens != PAD),
    }


@torch.no_grad()
def embed_segments(
    model: MaskedStreamTransformer,
    segments: list[np.ndarray],
    device: torch.device,
    batch_size: int,
) -> torch.Tensor:
    vectors = []
    for start in range(0, len(segments), batch_size):
        chunk = segments[start : start + batch_size]
        batch = {k: v.to(device) for k, v in collate_segments(chunk).items()}
        pooled, _ = model.embed(
            batch["tokens"], batch["segment_ids"], padding_mask=batch["padding_mask"]
        )
        vectors.append(pooled[:, 0, :].float().cpu())
    vectors = torch.cat(vectors)
    return torch.nn.functional.normalize(vectors, dim=-1)


def board_metadata(
    segments: list[np.ndarray],
) -> tuple[list[chess.Board], np.ndarray, np.ndarray]:
    encoder = TokenStreamEncoder()
    boards = [encoder.decode(segment) for segment in segments]
    piece_counts = np.array([len(board.piece_map()) for board in boards])
    materials = np.array(
        [
            sum(
                PIECE_VALUES[piece.piece_type] * (1 if piece.color else -1)
                for piece in board.piece_map().values()
            )
            for board in boards
        ]
    )
    return boards, piece_counts, materials


def neighbor_report(
    embeds: torch.Tensor,
    boards: list[chess.Board],
    piece_counts: np.ndarray,
    materials: np.ndarray,
    n_queries: int,
    seed: int,
) -> dict[str, float]:
    sims = (embeds @ embeds.T).numpy()
    np.fill_diagonal(sims, -np.inf)
    rng = np.random.default_rng(seed)
    queries = rng.choice(len(boards), size=n_queries, replace=False)
    for q in queries:
        top = np.argsort(-sims[q])[:5]
        print(
            f"\n  query: {boards[q].fen()}  material={materials[q]:+.0f} pieces={piece_counts[q]}"
        )
        for rank, j in enumerate(top, 1):
            print(
                f"    {rank}. sim={sims[q][j]:.3f} material={materials[j]:+.0f} "
                f"pieces={piece_counts[j]}  {boards[j].fen()}"
            )
    k = min(10, len(boards) - 1)
    topk_mean = float(np.mean(np.sort(sims, axis=1)[:, -k:]))
    ii, jj = np.triu_indices(len(boards), k=1)
    if len(ii) > 200_000:
        pick = rng.choice(len(ii), size=200_000, replace=False)
        ii, jj = ii[pick], jj[pick]
    random_mean = float(np.mean(sims[ii, jj]))
    top5 = np.argsort(-sims, axis=1)[:, :5]
    top5_material = float(
        np.mean(
            [
                np.mean(np.abs(materials[top5[i]] - materials[i]))
                for i in range(len(boards))
            ]
        )
    )
    random_material = float(np.mean(np.abs(materials[jj] - materials[ii])))
    return {
        f"mean_top{k}_cosine": topk_mean,
        "mean_random_pair_cosine": random_mean,
        "top5_material_delta": top5_material,
        "random_pair_material_delta": random_material,
    }


def corrupt_segment(segment: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    piece_positions = np.flatnonzero(segment >= PIECE_SQUARE_BASE)
    picks = rng.choice(
        len(piece_positions), size=min(2, len(piece_positions)), replace=False
    )
    corrupted = segment.copy()
    for p in picks:
        corrupted[piece_positions[p]] = rng.integers(PIECE_SQUARE_BASE, VOCAB_SIZE)
    return corrupted


@torch.no_grad()
def counterfactual_report(
    model: MaskedStreamTransformer,
    segments: list[np.ndarray],
    embeds: torch.Tensor,
    boards: list[chess.Board],
    device: torch.device,
    batch_size: int,
    n: int,
    seed: int,
) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    picks = rng.choice(len(segments), size=min(n, len(segments)), replace=False)
    children: list[np.ndarray] = []
    corrupted: list[np.ndarray] = []
    used: list[int] = []
    for i in picks:
        board = boards[int(i)]
        moves = list(board.legal_moves)
        if not moves:
            continue
        child = board.copy()
        child.push(moves[int(rng.integers(len(moves)))])
        children.append(TokenStreamEncoder().encode(child))
        corrupted.append(corrupt_segment(segments[int(i)], rng))
        used.append(int(i))
    child_embeds = embed_segments(model, children, device, batch_size)
    corrupted_embeds = embed_segments(model, corrupted, device, batch_size)
    base = embeds[torch.tensor(used)]
    child_sim = torch.sum(base * child_embeds, dim=-1).mean().item()
    corrupted_sim = torch.sum(base * corrupted_embeds, dim=-1).mean().item()
    used_t = torch.tensor(used)
    random_j = torch.randint(0, len(segments), (len(used),))
    clash = random_j == used_t
    while bool(clash.any()):
        random_j[clash] = torch.randint(0, len(segments), (int(clash.sum()),))
        clash = random_j == used_t
    random_sim = torch.sum(base * embeds[random_j], dim=-1).mean().item()
    return {
        "cos_to_move_child": child_sim,
        "cos_to_corrupted": corrupted_sim,
        "cos_to_random_position": random_sim,
    }


def ridge_probe(
    embeds: torch.Tensor, targets: np.ndarray, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    indices = rng.permutation(len(embeds))
    split = int(0.8 * len(indices))
    train, test = indices[:split], indices[split:]
    x = embeds.numpy()
    y = targets
    x_mean, y_mean = x[train].mean(axis=0), y[train].mean()
    x_train = x[train] - x_mean
    x_test = x[test] - x_mean
    y_train = y[train] - y_mean
    gram = x_train.T @ x_train + 1e-2 * np.eye(x.shape[1])
    weights = np.linalg.solve(gram, x_train.T @ y_train)
    return x_test @ weights + y_mean, y[test]


def material_probe(
    embeds: torch.Tensor, materials: np.ndarray, seed: int
) -> dict[str, float]:
    predictions, y_test = ridge_probe(embeds, materials, seed)
    ss_res = float(np.sum((y_test - predictions) ** 2))
    ss_tot = float(np.sum((y_test - y_test.mean()) ** 2))
    return {"test_r2_material": 1.0 - ss_res / max(ss_tot, 1e-9)}


def side_to_move_probe(
    embeds: torch.Tensor, boards: list[chess.Board], seed: int
) -> dict[str, float]:
    labels = np.array([1.0 if board.turn else -1.0 for board in boards])
    predictions, y_test = ridge_probe(embeds, labels, seed)
    accuracy = float(np.mean(np.sign(predictions) == np.sign(y_test)))
    return {"test_acc_side_to_move": accuracy}


def phase_plot(
    embeds: torch.Tensor, piece_counts: np.ndarray, out_dir: Path
) -> dict[str, float]:
    x = embeds.numpy()
    x = x - x.mean(axis=0)
    _, _, vt = np.linalg.svd(x, full_matrices=False)
    projected = x @ vt[:2].T
    corr1 = float(np.corrcoef(projected[:, 0], piece_counts)[0, 1])
    corr2 = float(np.corrcoef(projected[:, 1], piece_counts)[0, 1])
    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(
        projected[:, 0], projected[:, 1], c=piece_counts, s=6, cmap="viridis"
    )
    plt.colorbar(scatter, label="pieces on board")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.title("Position embeddings coloured by game phase")
    plt.tight_layout()
    out_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_dir / "pca_phase.png", dpi=150)
    plt.close()
    return {"pc1_piece_corr": corr1, "pc2_piece_corr": corr2}


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    rng_seed = args.seed

    print(f"loading checkpoint {args.checkpoint}")
    model = load_model(args.checkpoint, device)
    print("loading games")
    games = load_games(args.data_dir)
    print(f"games={len(games)}")

    print("\n=== 1. MLM accuracy by masking mode ===")
    prior = piece_prior(games)
    for mode, stats in mlm_breakdown(model, args.data_dir, args, device, prior).items():
        print(
            f"  {mode:6s} acc={stats['acc']:.4f} piece={stats['piece_acc']:.4f} "
            f"(prior baseline={stats['prior_baseline']:.4f}) scalar={stats['scalar_acc']:.4f}"
        )

    print("\n=== 2. Position embeddings ===")
    print("building embedding pool")
    segments = sample_segments(games, args.n_positions, rng_seed)
    embeds = embed_segments(model, segments, device, batch_size=256)
    boards, piece_counts, materials = board_metadata(segments)
    print(f"embedded {len(segments)} positions (d={embeds.shape[1]})")

    print("\n-- nearest neighbours --")
    for name, value in neighbor_report(
        embeds, boards, piece_counts, materials, args.n_queries, rng_seed
    ).items():
        print(f"  {name}: {value:.4f}")

    print("\n-- counterfactual similarities --")
    for name, value in counterfactual_report(
        model, segments, embeds, boards, device, 256, args.n_counterfactual, rng_seed
    ).items():
        print(f"  {name}: {value:.4f}")

    print("\n-- linear probes (ridge) --")
    for name, value in material_probe(embeds, materials, rng_seed).items():
        print(f"  {name}: {value:.4f}")
    for name, value in side_to_move_probe(embeds, boards, rng_seed).items():
        print(f"  {name}: {value:.4f}")

    print("\n-- phase structure (PCA) --")
    for name, value in phase_plot(embeds, piece_counts, Path(args.out_dir)).items():
        print(f"  {name}: {value:.4f}")
    print(f"  plot saved to {args.out_dir}/pca_phase.png")


if __name__ == "__main__":
    main()
