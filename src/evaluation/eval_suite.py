from __future__ import annotations

import numpy as np
import torch

from src.dataset.token_stream import (
    CASTLE_BASE,
    EP_BASE,
    PIECE_SQUARE_BASE,
    TURN_WHITE,
    VOCAB_SIZE,
    TokenStreamEncoder,
)
from src.modeling.masking import IGNORE_INDEX, MaskingConfig
from src.modeling.transformer import MaskedStreamTransformer
from src.training.window_dataset import (
    CLS,
    PAD,
    SEP,
    WindowDataset,
    WindowDatasetConfig,
    collate_windows,
)

MODE_CONFIGS = {
    "random": MaskingConfig(p_random=1.0, p_board=0.0, p_span=0.0),
    "board": MaskingConfig(p_random=0.0, p_board=1.0, p_span=0.0),
    "span": MaskingConfig(p_random=0.0, p_board=0.0, p_span=1.0),
}
SCALAR_CLASS_BOUNDS = {
    "turn": (TURN_WHITE, TURN_WHITE + 2),
    "castle": (CASTLE_BASE, EP_BASE),
    "ep": (EP_BASE, PIECE_SQUARE_BASE),
}


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
    games: list[list[np.ndarray]],
    device: torch.device,
    *,
    windows: int,
    batch_size: int,
    seed: int,
    max_seq_len: int,
    max_segments: int,
    prior: np.ndarray,
) -> dict[str, dict[str, float]]:
    results = {}
    for mode, masking in MODE_CONFIGS.items():
        dataset = WindowDataset(
            WindowDatasetConfig(
                games=games,
                max_seq_len=max_seq_len,
                max_segments=max_segments,
                masking=masking,
                seed=seed,
            )
        )
        rng = np.random.default_rng(seed)
        indices = rng.permutation(len(dataset))[:windows]
        counts = {
            "total": 0,
            "correct": 0,
            "piece": 0,
            "piece_correct": 0,
            "scalar": 0,
            "scalar_correct": 0,
            "turn": 0,
            "turn_correct": 0,
            "castle": 0,
            "castle_correct": 0,
            "ep": 0,
            "ep_correct": 0,
        }
        prior_score = 0.0
        for start in range(0, len(indices), batch_size):
            samples = [dataset[int(i)] for i in indices[start : start + batch_size]]
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
            for name, (low, high) in SCALAR_CLASS_BOUNDS.items():
                member = targets.ge(low) & targets.lt(high)
                counts[name] += int(member.sum())
                counts[f"{name}_correct"] += int(correct[member].sum())
            for token, ok in zip(targets[piece].tolist(), correct[piece].tolist()):
                prior_score += float(prior[token]) if not ok else 1.0
        results[mode] = {
            "acc": counts["correct"] / max(1, counts["total"]),
            "piece_acc": counts["piece_correct"] / max(1, counts["piece"]),
            "scalar_acc": counts["scalar_correct"] / max(1, counts["scalar"]),
            "turn_acc": counts["turn_correct"] / max(1, counts["turn"]),
            "castle_acc": counts["castle_correct"] / max(1, counts["castle"]),
            "prior_baseline": prior_score / max(1, counts["piece"]),
        }
    return results


def sample_segments(
    games: list[list[np.ndarray]], n: int, seed: int
) -> list[np.ndarray]:
    rng = np.random.default_rng(seed)
    segments: list[np.ndarray] = []
    while len(segments) < n:
        for index in rng.permutation(len(games)):
            game = games[int(index)]
            segments.append(game[int(rng.integers(len(game)))])
            if len(segments) >= n:
                return segments
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
    batch_size: int = 256,
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
) -> tuple[list, np.ndarray, np.ndarray]:
    import chess

    encoder = TokenStreamEncoder()
    boards = [encoder.decode(segment) for segment in segments]
    piece_values = {
        chess.PAWN: 1.0,
        chess.KNIGHT: 3.0,
        chess.BISHOP: 3.0,
        chess.ROOK: 5.0,
        chess.QUEEN: 9.0,
        chess.KING: 0.0,
    }
    piece_counts = np.array([len(board.piece_map()) for board in boards])
    materials = np.array(
        [
            sum(
                piece_values[piece.piece_type] * (1 if piece.color else -1)
                for piece in board.piece_map().values()
            )
            for board in boards
        ]
    )
    return boards, piece_counts, materials


def neighbor_stats(
    embeds: torch.Tensor,
    piece_counts: np.ndarray,
    materials: np.ndarray,
    seed: int,
    k: int = 10,
    max_pairs: int = 200_000,
) -> dict[str, float]:
    sims = (embeds @ embeds.T).numpy()
    np.fill_diagonal(sims, -np.inf)
    topk = min(k, len(sims) - 1)
    topk_mean = float(np.mean(np.sort(sims, axis=1)[:, -topk:]))
    rng = np.random.default_rng(seed)
    ii, jj = np.triu_indices(len(sims), k=1)
    if len(ii) > max_pairs:
        pick = rng.choice(len(ii), size=max_pairs, replace=False)
        ii, jj = ii[pick], jj[pick]
    top5 = np.argsort(-sims, axis=1)[:, :5]
    top5_material = float(
        np.mean(
            [
                np.mean(np.abs(materials[top5[i]] - materials[i]))
                for i in range(len(sims))
            ]
        )
    )
    return {
        f"mean_top{topk}_cosine": topk_mean,
        "mean_random_pair_cosine": float(np.mean(sims[ii, jj])),
        "top5_material_delta": top5_material,
        "random_pair_material_delta": float(
            np.mean(np.abs(materials[jj] - materials[ii]))
        ),
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
def counterfactual_stats(
    model: MaskedStreamTransformer,
    segments: list[np.ndarray],
    embeds: torch.Tensor,
    boards: list,
    device: torch.device,
    batch_size: int,
    n: int,
    seed: int,
) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    picks = rng.choice(len(segments), size=min(n, len(segments)), replace=False)
    encoder = TokenStreamEncoder()
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
        children.append(encoder.encode(child))
        corrupted.append(corrupt_segment(segments[int(i)], rng))
        used.append(int(i))
    if not used:
        return {}
    child_embeds = embed_segments(model, children, device, batch_size)
    corrupted_embeds = embed_segments(model, corrupted, device, batch_size)
    base = embeds[torch.tensor(used)]
    used_t = torch.tensor(used)
    random_j = torch.randint(0, len(segments), (len(used),))
    clash = random_j == used_t
    while bool(clash.any()):
        random_j[clash] = torch.randint(0, len(segments), (int(clash.sum()),))
        clash = random_j == used_t
    return {
        "cos_to_move_child": torch.sum(base * child_embeds, dim=-1).mean().item(),
        "cos_to_corrupted": torch.sum(base * corrupted_embeds, dim=-1).mean().item(),
        "cos_to_random_position": torch.sum(base * embeds[random_j], dim=-1)
        .mean()
        .item(),
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
    embeds: torch.Tensor, boards: list, seed: int
) -> dict[str, float]:
    import chess

    labels = np.array([1.0 if board.turn == chess.WHITE else -1.0 for board in boards])
    predictions, y_test = ridge_probe(embeds, labels, seed)
    accuracy = float(np.mean(np.sign(predictions) == np.sign(y_test)))
    return {"test_acc_side_to_move": accuracy}


def phase_stats(
    embeds: torch.Tensor, piece_counts: np.ndarray
) -> tuple[np.ndarray, float, float]:
    x = embeds.numpy()
    x = x - x.mean(axis=0)
    _, _, vt = np.linalg.svd(x, full_matrices=False)
    projected = x @ vt[:2].T
    corr1 = float(np.corrcoef(projected[:, 0], piece_counts)[0, 1])
    corr2 = float(np.corrcoef(projected[:, 1], piece_counts)[0, 1])
    return projected, corr1, corr2


@torch.no_grad()
def full_eval(
    model: MaskedStreamTransformer,
    val_games: list[list[np.ndarray]],
    device: torch.device,
    *,
    windows: int = 20,
    batch_size: int = 8,
    max_seq_len: int = 2048,
    max_segments: int = 128,
    pool_size: int = 1024,
    counterfactual_n: int = 256,
    seed: int = 17,
) -> dict[str, float]:
    was_training = model.training
    model.eval()
    metrics: dict[str, float] = {}
    prior = piece_prior(val_games)
    for mode, stats in mlm_breakdown(
        model,
        val_games,
        device,
        windows=windows,
        batch_size=batch_size,
        seed=seed,
        max_seq_len=max_seq_len,
        max_segments=max_segments,
        prior=prior,
    ).items():
        for key, value in stats.items():
            metrics[f"{mode}_{key}"] = value
    segments = sample_segments(val_games, pool_size, seed)
    if len(segments) >= 8:
        embeds = embed_segments(model, segments, device)
        boards, piece_counts, materials = board_metadata(segments)
        metrics.update(neighbor_stats(embeds, piece_counts, materials, seed))
        if counterfactual_n > 0:
            metrics.update(
                counterfactual_stats(
                    model,
                    segments,
                    embeds,
                    boards,
                    device,
                    batch_size,
                    counterfactual_n,
                    seed,
                )
            )
        metrics.update(material_probe(embeds, materials, seed))
        metrics.update(side_to_move_probe(embeds, boards, seed))
        _, corr1, corr2 = phase_stats(embeds, piece_counts)
        metrics["pc1_piece_corr"] = corr1
        metrics["pc2_piece_corr"] = corr2
    if was_training:
        model.train()
    return metrics
