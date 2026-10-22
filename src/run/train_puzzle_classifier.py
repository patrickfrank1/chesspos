from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from src.dataset.puzzle_variants import (
    EpochPlan,
    PuzzleRecord,
    PuzzleVariantDataset,
    RollingVariantPlanner,
    collate_variants,
    load_puzzle_records,
    split_for_puzzle,
)
from src.modeling.puzzle_classifier import (
    FrozenEncoderClassifier,
    PoolingClassifier,
    PoolingClassifierConfig,
)
from src.modeling.transformer import MaskedStreamTransformer, TransformerConfig
from src.training.losses import (
    FocalLossConfig,
    focal_loss_with_logits,
    resolve_focal_loss,
)
from src.training.puzzle_metrics import binary_metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a frozen-encoder binary classifier for a puzzle theme"
    )
    parser.add_argument("--data-dir", default="data/processed/puzzles")
    parser.add_argument("--theme", default="sacrifice")
    parser.add_argument(
        "--checkpoint",
        default="data/models/run_d512l8_20261021_streaming/best_step40000.pt",
    )
    parser.add_argument("--output-dir", default="data/models/puzzle_classifier")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--positive-fraction", type=float, default=0.5)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--head-hidden-dim", type=int, default=256)
    parser.add_argument("--pooling", choices=("mean", "attention"), default="attention")
    parser.add_argument("--pos-weight", type=float, default=None)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--loss", choices=("bce", "focal"), default="bce")
    parser.add_argument(
        "--focal-gamma",
        type=float,
        default=None,
        help="Focal modulating exponent; calibrated from train labels when omitted",
    )
    parser.add_argument(
        "--focal-alpha",
        type=float,
        default=None,
        help="Positive-class weight; calibrated from train labels when omitted",
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--seed", type=int, default=17)
    return parser.parse_args()


def load_frozen_encoder(path: str, device: torch.device) -> MaskedStreamTransformer:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    config = TransformerConfig(**checkpoint["model_config"])
    encoder = MaskedStreamTransformer(config)
    encoder.load_state_dict(checkpoint["model_state"])
    encoder.to(device).eval()
    for parameter in encoder.parameters():
        parameter.requires_grad = False
    return encoder


def build_validation_plan(records: list[PuzzleRecord]) -> EpochPlan:
    """Use every single position once, preserving the natural class prevalence."""
    puzzle_index: list[int] = []
    start: list[int] = []
    end: list[int] = []
    label: list[int] = []
    for index, record in enumerate(records):
        for position in range(len(record.segments)):
            puzzle_index.append(index)
            start.append(position)
            end.append(position)
            label.append(record.label)
    return EpochPlan(
        puzzle_index=np.asarray(puzzle_index, dtype=np.int32),
        start=np.asarray(start, dtype=np.int16),
        end=np.asarray(end, dtype=np.int16),
        label=np.asarray(label, dtype=np.int8),
    )


def make_loader(
    dataset: PuzzleVariantDataset,
    batch_size: int,
    num_workers: int,
    device: torch.device,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_variants,
        pin_memory=device.type == "cuda",
        persistent_workers=False,
    )


def train_epoch(
    model: FrozenEncoderClassifier,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    pos_weight: torch.Tensor | None,
    epoch: int = 1,
    log_every: int = 0,
    focal: FocalLossConfig | None = None,
) -> float:
    model.train()
    total_loss = 0.0
    total_examples = 0
    total_batches = len(loader)
    for step, batch in enumerate(loader, start=1):
        tokens = batch["tokens"].to(device, non_blocking=True)
        segment_ids = batch["segment_ids"].to(device, non_blocking=True)
        padding_mask = batch["padding_mask"].to(device, non_blocking=True)
        labels = batch["labels"].to(device, non_blocking=True)

        logits = model(tokens, segment_ids, padding_mask)
        if focal is None:
            loss = nn.functional.binary_cross_entropy_with_logits(
                logits, labels, pos_weight=pos_weight
            )
        else:
            loss = focal_loss_with_logits(
                logits, labels, gamma=focal.gamma, alpha=focal.alpha
            )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.head.parameters(), 1.0)
        optimizer.step()

        total_loss += float(loss.item()) * labels.shape[0]
        total_examples += labels.shape[0]
        if log_every > 0 and step % log_every == 0:
            print(
                f"epoch={epoch} step={step}/{total_batches} "
                f"loss={total_loss / total_examples:.4f} "
                f"lr={optimizer.param_groups[0]['lr']:.2e}",
                flush=True,
            )
    return total_loss / max(1, total_examples)


@torch.no_grad()
def evaluate(
    model: FrozenEncoderClassifier,
    loader: DataLoader,
    device: torch.device,
) -> tuple[dict[str, float], float]:
    model.eval()
    scores: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    total_loss = 0.0
    total_examples = 0
    for batch in loader:
        tokens = batch["tokens"].to(device, non_blocking=True)
        segment_ids = batch["segment_ids"].to(device, non_blocking=True)
        padding_mask = batch["padding_mask"].to(device, non_blocking=True)
        batch_labels = batch["labels"].to(device, non_blocking=True)

        logits = model(tokens, segment_ids, padding_mask)
        loss = nn.functional.binary_cross_entropy_with_logits(logits, batch_labels)
        total_loss += float(loss.item()) * batch_labels.shape[0]
        total_examples += batch_labels.shape[0]

        scores.append(torch.sigmoid(logits).cpu().numpy())
        labels.append(batch_labels.cpu().numpy())

    metrics = binary_metrics(np.concatenate(scores), np.concatenate(labels))
    return metrics, total_loss / max(1, total_examples)


def save_checkpoint(
    path: Path,
    model: FrozenEncoderClassifier,
    planner: RollingVariantPlanner,
    epoch: int,
    metrics: dict[str, float],
) -> None:
    torch.save(
        {
            "encoder_config": asdict(model.encoder.config),
            "encoder_state": model.encoder.state_dict(),
            "head_config": asdict(model.head.config),
            "head_state": model.head.state_dict(),
            "planner_state": planner.state_dict(),
            "epoch": epoch,
            "metrics": metrics,
        },
        path,
    )
    return


def main() -> int:
    args = parse_args()
    torch.manual_seed(args.seed)
    device = torch.device(
        args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    )

    records = load_puzzle_records(args.data_dir, args.theme, limit=args.limit)
    train_records = [
        record
        for record in records
        if split_for_puzzle(record.puzzle_id, args.val_ratio) == "train"
    ]
    val_records = [
        record
        for record in records
        if split_for_puzzle(record.puzzle_id, args.val_ratio) == "val"
    ]
    if not train_records:
        raise ValueError("training split is empty; increase --limit")
    if not val_records:
        raise ValueError("validation split is empty; increase --limit")

    encoder = load_frozen_encoder(args.checkpoint, device)
    head = PoolingClassifier(
        PoolingClassifierConfig(
            d_model=encoder.config.d_model,
            dropout=args.dropout,
            hidden_dim=args.head_hidden_dim,
            pooling=args.pooling,
        )
    ).to(device)
    model = FrozenEncoderClassifier(encoder, head).to(device)

    planner = RollingVariantPlanner(
        train_records,
        positive_fraction=args.positive_fraction,
        seed=args.seed,
    )
    val_plan = build_validation_plan(val_records)
    val_loader = make_loader(
        PuzzleVariantDataset(val_records, val_plan),
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        device=device,
    )

    optimizer = torch.optim.AdamW(
        model.head.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    pos_weight = (
        torch.tensor(args.pos_weight, device=device)
        if args.pos_weight is not None
        else None
    )
    class_counts = Counter(record.label for record in train_records)
    focal = resolve_focal_loss(
        args.loss, class_counts, args.focal_gamma, args.focal_alpha
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    best_pr_auc = -1.0

    print(
        f"records={len(records)} train={len(train_records)} val={len(val_records)} "
        f"theme={args.theme} device={device} loss={args.loss} "
        f"pooling={args.pooling}"
    )
    if focal is not None:
        print(f"focal_gamma={focal.gamma:.4f} focal_alpha={focal.alpha:.4f}")
    for epoch in range(1, args.epochs + 1):
        plan = planner.next_epoch()
        train_loader = make_loader(
            PuzzleVariantDataset(train_records, plan),
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            device=device,
        )
        train_loss = train_epoch(
            model,
            train_loader,
            optimizer,
            device,
            pos_weight,
            epoch=epoch,
            log_every=args.log_every,
        )
        metrics, val_loss = evaluate(model, val_loader, device)
        print(
            f"epoch={epoch} train_loss={train_loss:.4f} val_loss={val_loss:.4f} "
            f"auroc={metrics['auroc']:.4f} pr_auc={metrics['pr_auc']:.4f} "
            f"f1={metrics['f1']:.4f} threshold={metrics['threshold']:.4f}"
        )
        save_checkpoint(output_dir / "last.pt", model, planner, epoch, metrics)
        if metrics["pr_auc"] > best_pr_auc:
            best_pr_auc = metrics["pr_auc"]
            save_checkpoint(output_dir / "best.pt", model, planner, epoch, metrics)
            print(f"saved best checkpoint to {output_dir / 'best.pt'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
