from __future__ import annotations

import argparse
import fnmatch
import json
import random
from dataclasses import asdict
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from src.dataset.game_outcomes import (
    GameOutcome,
    ScoreDataset,
    ScorePlan,
    build_score_plan,
    collate_score_samples,
    load_game_records,
    read_game_shard,
    split_for_game,
)
from src.modeling.factory import build_model, model_config_from_dict
from src.modeling.frozen import embedding_dim
from src.modeling.score_head import (
    FrozenEncoderScore,
    ScoreHead,
    ScoreHeadConfig,
)
from src.training.autocast import autocast_context
from src.training.early_stop import is_at_chance
from src.training.score_metrics import score_metrics


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a frozen-encoder position score regressor"
    )
    parser.add_argument("--data-dir", default="data/processed/games")
    parser.add_argument("--remote-repo", default=None)
    parser.add_argument("--remote-split", default="train")
    parser.add_argument(
        "--remote-shard-pattern",
        default=None,
        help="Optional fnmatch pattern applied to remote shard paths",
    )
    parser.add_argument("--remote-cache-dir", default="data/hf-shards")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", default="data/models/score_probe")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument(
        "--encoder-lr",
        type=float,
        default=1e-5,
        help="Learning rate for unfrozen encoder blocks (last-block arm)",
    )
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--head-hidden-dim", type=int, default=256)
    parser.add_argument("--pooling", choices=("mean", "attention"), default="mean")
    parser.add_argument(
        "--feature", choices=("native", "full", "native_positions"), default="native"
    )
    parser.add_argument("--loss", choices=("mse", "huber"), default="mse")
    parser.add_argument("--amp", choices=("none", "bf16"), default="none")
    parser.add_argument(
        "--abort-if-chance",
        action="store_true",
        help="Stop after an epoch whose decisive AUC is not above chance",
    )
    parser.add_argument("--chance-margin-z", type=float, default=1.645)
    parser.add_argument("--include-state-features", action="store_true")
    parser.add_argument(
        "--token-pooling",
        choices=("mean", "attention"),
        default="mean",
        help="Pool the last token layer by mean or a learned attention query",
    )
    unfreeze = parser.add_mutually_exclusive_group()
    unfreeze.add_argument("--unfreeze-last-block", action="store_true")
    unfreeze.add_argument(
        "--unfreeze-all",
        action="store_true",
        help="Fine-tune every encoder parameter (requires --feature full)",
    )
    parser.add_argument("--window", type=int, default=1)
    parser.add_argument("--positions-per-game", type=int, default=16)
    parser.add_argument("--max-ply", type=int, default=150)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--seed", type=int, default=17)
    return parser.parse_args(argv)


def load_frozen_encoder(path: str, device: torch.device) -> nn.Module:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    config = model_config_from_dict(checkpoint["model_config"])
    encoder = build_model(config)
    encoder.load_state_dict(checkpoint["model_state"])
    encoder.to(device).eval()
    for parameter in encoder.parameters():
        parameter.requires_grad = False
    return encoder


def load_remote_game_records(
    repo: str,
    split: str,
    limit: int | None = None,
    seed: int = 17,
    cache_dir: str = "data/hf-shards",
    shard_pattern: str | None = None,
) -> list[GameOutcome]:
    """Download shuffled remote shards until ``limit`` games are loaded."""
    from huggingface_hub import HfApi, hf_hub_download

    files = [
        path
        for path in HfApi().list_repo_files(repo, repo_type="dataset")
        if path.startswith(f"{split}/") and path.endswith(".parquet")
    ]
    if shard_pattern is not None:
        files = [path for path in files if fnmatch.fnmatch(path, shard_pattern)]
    if not files:
        raise ValueError(f"no parquet shards found for {repo}/{split}")
    random.Random(seed).shuffle(files)

    records: list[GameOutcome] = []
    for name in files:
        remaining = None if limit is None else limit - len(records)
        if remaining is not None and remaining <= 0:
            break
        shard = hf_hub_download(repo, name, repo_type="dataset", cache_dir=cache_dir)
        records.extend(read_game_shard(Path(shard), limit=remaining))
    return records


def resolve_records(args: argparse.Namespace) -> list[GameOutcome]:
    if args.remote_repo is not None:
        return load_remote_game_records(
            args.remote_repo,
            args.remote_split,
            limit=args.limit,
            seed=args.seed,
            cache_dir=args.remote_cache_dir,
            shard_pattern=args.remote_shard_pattern,
        )
    return load_game_records(args.data_dir, limit=args.limit)


def build_optimizer(
    model: FrozenEncoderScore,
    lr: float,
    encoder_lr: float,
    weight_decay: float,
) -> torch.optim.Optimizer:
    """Separate parameter groups: unfrozen encoder blocks train slowly."""
    encoder_params = [
        parameter
        for name, parameter in model.encoder.named_parameters()
        if parameter.requires_grad
    ]
    head_params = [
        parameter for parameter in model.head.parameters() if parameter.requires_grad
    ]
    groups: list[dict] = []
    if encoder_params:
        groups.append({"params": encoder_params, "lr": encoder_lr})
    if head_params:
        groups.append({"params": head_params, "lr": lr})
    return torch.optim.AdamW(groups, lr=lr, weight_decay=weight_decay)


def build_score_loader(
    dataset: ScoreDataset,
    batch_size: int,
    num_workers: int,
    device: torch.device,
    shuffle: bool = False,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_score_samples,
        pin_memory=device.type == "cuda",
        persistent_workers=False,
    )


def _features(batch: dict, device: torch.device) -> torch.Tensor | None:
    features = batch.get("features")
    if features is None:
        return None
    return features.to(device, non_blocking=True)


def _loss(scores: torch.Tensor, targets: torch.Tensor, loss: str) -> torch.Tensor:
    if loss == "mse":
        return nn.functional.mse_loss(scores, targets)
    if loss == "huber":
        return nn.functional.smooth_l1_loss(scores, targets)
    raise ValueError(f"unknown loss {loss!r}")


def train_epoch(
    model: FrozenEncoderScore,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    loss: str = "mse",
    epoch: int = 1,
    log_every: int = 0,
    amp: str = "none",
) -> float:
    model.train()
    total_loss = 0.0
    total_examples = 0
    total_batches = len(loader)
    trainable = [p for p in model.parameters() if p.requires_grad]
    for step, batch in enumerate(loader, start=1):
        tokens = batch["tokens"].to(device, non_blocking=True)
        segment_ids = batch["segment_ids"].to(device, non_blocking=True)
        padding_mask = batch["padding_mask"].to(device, non_blocking=True)
        targets = batch["targets"].to(device, non_blocking=True)

        with autocast_context(device, amp):
            scores = model(tokens, segment_ids, padding_mask, _features(batch, device))
            batch_loss = _loss(scores, targets, loss)
        optimizer.zero_grad(set_to_none=True)
        batch_loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        optimizer.step()

        total_loss += float(batch_loss.item()) * targets.shape[0]
        total_examples += targets.shape[0]
        if log_every > 0 and step % log_every == 0:
            print(
                f"epoch={epoch} step={step}/{total_batches} "
                f"loss={total_loss / total_examples:.4f} "
                f"lr={optimizer.param_groups[0]['lr']:.2e}",
                flush=True,
            )
    return total_loss / max(1, total_examples)


@torch.no_grad()
def predict(
    model: FrozenEncoderScore,
    loader: DataLoader,
    device: torch.device,
    amp: str = "none",
) -> dict[str, np.ndarray]:
    model.eval()
    predictions: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    plies: list[np.ndarray] = []
    materials: list[np.ndarray] = []
    game_ids: list[str] = []
    sources: list[str] = []
    for batch in loader:
        tokens = batch["tokens"].to(device, non_blocking=True)
        segment_ids = batch["segment_ids"].to(device, non_blocking=True)
        padding_mask = batch["padding_mask"].to(device, non_blocking=True)
        with autocast_context(device, amp):
            scores = model(tokens, segment_ids, padding_mask, _features(batch, device))
        predictions.append(scores.float().cpu().numpy())
        targets.append(batch["targets"].numpy())
        plies.append(batch["plies"].numpy())
        materials.append(batch["materials"].numpy())
        game_ids.extend(batch["game_ids"])
        sources.extend(batch["sources"])
    return {
        "predictions": np.concatenate(predictions).astype(np.float32),
        "targets": np.concatenate(targets).astype(np.float32),
        "plies": np.concatenate(plies).astype(np.int16),
        "materials": np.concatenate(materials).astype(np.int16),
        "game_ids": np.asarray(game_ids),
        "sources": np.asarray(sources),
    }


def evaluate(
    model: FrozenEncoderScore,
    loader: DataLoader,
    device: torch.device,
    amp: str = "none",
) -> tuple[dict[str, float], float]:
    outputs = predict(model, loader, device, amp=amp)
    metrics = score_metrics(outputs["predictions"], outputs["targets"])
    return metrics, metrics["mse"]


def save_predictions(path: Path, outputs: dict[str, np.ndarray]) -> None:
    np.savez_compressed(path, **outputs)
    return


def save_checkpoint(
    path: Path,
    model: FrozenEncoderScore,
    epoch: int,
    feature: str,
    unfreeze_last_block: bool,
    metrics: dict[str, float],
    window: int = 1,
    include_state_features: bool = False,
    token_pooling: str = "mean",
    unfreeze_all: bool = False,
) -> None:
    torch.save(
        {
            "encoder_config": asdict(model.encoder.config),
            "encoder_state": model.encoder.state_dict(),
            "head_config": asdict(model.head.config),
            "head_state": model.head.state_dict(),
            "feature": feature,
            "unfreeze_last_block": unfreeze_last_block,
            "unfreeze_all": unfreeze_all,
            "token_pooling": token_pooling,
            "token_pool_state": (
                model.token_pool.state_dict() if model.token_pool is not None else None
            ),
            "window": window,
            "include_state_features": include_state_features,
            "epoch": epoch,
            "metrics": metrics,
        },
        path,
    )
    return


def _validation_plan(records: Sequence[GameOutcome]) -> ScorePlan:
    """Score every position once, capped by the model's usable sequence length."""
    max_ply = max((len(record.segments) for record in records), default=0)
    return build_score_plan(
        records,
        positions_per_game=None,
        max_ply=max(1, max_ply),
        seed=0,
    )


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    torch.manual_seed(args.seed)
    device = torch.device(
        args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    )

    records = resolve_records(args)
    train_records = [
        record
        for record in records
        if split_for_game(record.game_id, args.val_ratio) == "train"
    ]
    val_records = [
        record
        for record in records
        if split_for_game(record.game_id, args.val_ratio) == "val"
    ]
    if not train_records:
        raise ValueError("training split is empty; increase --limit")
    if not val_records:
        raise ValueError("validation split is empty; increase --limit")

    encoder = load_frozen_encoder(args.checkpoint, device)
    head = ScoreHead(
        ScoreHeadConfig(
            d_model=embedding_dim(encoder, args.feature),
            hidden_dim=args.head_hidden_dim,
            dropout=args.dropout,
            pooling=args.pooling,
            state_features=3 if args.include_state_features else 0,
        )
    ).to(device)
    model = FrozenEncoderScore(
        encoder,
        head,
        feature=args.feature,
        unfreeze_last_block=args.unfreeze_last_block,
        token_pooling=args.token_pooling,
        unfreeze_all=args.unfreeze_all,
    ).to(device)

    optimizer = build_optimizer(
        model,
        lr=args.lr,
        encoder_lr=args.encoder_lr,
        weight_decay=args.weight_decay,
    )

    val_plan = _validation_plan(val_records)
    val_loader = build_score_loader(
        ScoreDataset(
            val_records,
            val_plan,
            window=args.window,
            include_state_features=args.include_state_features,
        ),
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        device=device,
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    best_val_loss = float("inf")
    history: list[dict] = []
    aborted = False

    print(
        f"records={len(records)} train={len(train_records)} "
        f"val={len(val_records)} val_positions={len(val_plan)} "
        f"feature={args.feature} loss={args.loss} device={device}"
    )
    for epoch in range(1, args.epochs + 1):
        plan = build_score_plan(
            train_records,
            positions_per_game=args.positions_per_game,
            max_ply=args.max_ply,
            seed=args.seed + epoch,
        )
        train_loader = build_score_loader(
            ScoreDataset(
                train_records,
                plan,
                window=args.window,
                include_state_features=args.include_state_features,
            ),
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            device=device,
            shuffle=True,
        )
        train_loss = train_epoch(
            model,
            train_loader,
            optimizer,
            device,
            loss=args.loss,
            epoch=epoch,
            log_every=args.log_every,
            amp=args.amp,
        )
        outputs = predict(model, val_loader, device, amp=args.amp)
        metrics = score_metrics(outputs["predictions"], outputs["targets"])
        val_loss = metrics["mse"]
        print(
            f"epoch={epoch} train_loss={train_loss:.4f} "
            f"val_loss={val_loss:.4f} r2={metrics['r2']:.4f} "
            f"mae={metrics['mae']:.4f} "
            f"auroc_decisive={metrics['auroc_decisive']:.4f}"
        )
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "metrics": metrics,
            }
        )
        save_checkpoint(
            output_dir / "last.pt",
            model,
            epoch,
            args.feature,
            args.unfreeze_last_block,
            metrics,
            window=args.window,
            include_state_features=args.include_state_features,
            token_pooling=args.token_pooling,
            unfreeze_all=args.unfreeze_all,
        )
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint(
                output_dir / "best.pt",
                model,
                epoch,
                args.feature,
                args.unfreeze_last_block,
                metrics,
                window=args.window,
                include_state_features=args.include_state_features,
                token_pooling=args.token_pooling,
                unfreeze_all=args.unfreeze_all,
            )
            save_predictions(output_dir / "predictions.npz", outputs)
            print(f"saved best checkpoint to {output_dir / 'best.pt'}")
        if args.abort_if_chance and epoch < args.epochs:
            decisive = outputs["targets"] != 0
            n_positive = int((outputs["targets"][decisive] > 0).sum())
            n_negative = int((outputs["targets"][decisive] < 0).sum())
            at_chance = is_at_chance(
                metrics["auroc_decisive"],
                n_positive,
                n_negative,
                args.chance_margin_z,
            )
            if at_chance:
                print(
                    f"aborted after epoch {epoch}: decisive AUC "
                    f"{metrics['auroc_decisive']:.4f} is not above chance "
                    f"(n_pos={n_positive}, n_neg={n_negative})"
                )
                aborted = True
                break

    (output_dir / "metrics.json").write_text(
        json.dumps(
            {
                "feature": args.feature,
                "unfreeze_last_block": args.unfreeze_last_block,
                "unfreeze_all": args.unfreeze_all,
                "window": args.window,
                "include_state_features": args.include_state_features,
                "loss": args.loss,
                "amp": args.amp,
                "token_pooling": args.token_pooling,
                "checkpoint": args.checkpoint,
                "train_games": len(train_records),
                "val_games": len(val_records),
                "history": history,
                "aborted": aborted,
            },
            indent=2,
        )
        + "\n"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
