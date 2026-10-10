from __future__ import annotations

import argparse
import math
import time
from dataclasses import asdict
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from src.modeling.transformer import MaskedStreamTransformer, TransformerConfig
from src.training.window_dataset import (
    WindowDataset,
    WindowDatasetConfig,
    collate_windows,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train masked stream transformer")
    parser.add_argument("--train-dir", default="data/processed/train")
    parser.add_argument("--val-dir", default="data/processed/test")
    parser.add_argument("--checkpoint-dir", default="models")
    parser.add_argument("--steps", type=int, default=10_000)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup", type=int, default=500)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--d-model", type=int, default=256)
    parser.add_argument("--n-layers", type=int, default=4)
    parser.add_argument("--n-heads", type=int, default=8)
    parser.add_argument("--d-ff", type=int, default=1024)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--max-seq-len", type=int, default=2048)
    parser.add_argument("--max-segments", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--log-every", type=int, default=20)
    parser.add_argument("--eval-every", type=int, default=1_000)
    parser.add_argument("--eval-batches", type=int, default=20)
    parser.add_argument("--seed", type=int, default=17)
    return parser.parse_args()


def lr_lambda(step: int, warmup: int, total: int):
    if step < warmup:
        return step / max(1, warmup)
    progress = (step - warmup) / max(1, total - warmup)
    return 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))


@torch.no_grad()
def evaluate(
    model: MaskedStreamTransformer,
    loader: DataLoader,
    device: torch.device,
    max_batches: int,
) -> tuple[float, float]:
    model.eval()
    losses = []
    correct = 0
    total = 0
    for i, batch in enumerate(loader):
        if i >= max_batches:
            break
        batch = {k: v.to(device) for k, v in batch.items()}
        output = model(
            batch["tokens"],
            batch["segment_ids"],
            padding_mask=batch["padding_mask"],
            targets=batch["targets"],
        )
        losses.append(output["loss"].item())
        valid = batch["targets"].ne(-100)
        predictions = output["logits"].argmax(-1)
        correct += (predictions[valid] == batch["targets"][valid]).sum().item()
        total += valid.sum().item()
    model.train()
    loss = sum(losses) / max(1, len(losses))
    accuracy = correct / max(1, total)
    return loss, accuracy


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_config = TransformerConfig(
        d_model=args.d_model,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        d_ff=args.d_ff,
        dropout=args.dropout,
        max_seq_len=args.max_seq_len,
        max_segments=args.max_segments,
    )
    model = MaskedStreamTransformer(model_config).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"device={device} params={n_params / 1e6:.2f}M")

    train_dataset = WindowDataset(
        WindowDatasetConfig(
            data_dir=args.train_dir,
            max_seq_len=args.max_seq_len,
            max_segments=args.max_segments,
        )
    )
    val_dataset = WindowDataset(
        WindowDatasetConfig(
            data_dir=args.val_dir,
            max_seq_len=args.max_seq_len,
            max_segments=args.max_segments,
            seed=args.seed,
        )
    )
    print(f"train games={len(train_dataset)} val games={len(val_dataset)}")

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_windows,
        drop_last=True,
        persistent_workers=args.num_workers > 0,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_windows,
        persistent_workers=args.num_workers > 0,
    )

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda step: lr_lambda(step, args.warmup, args.steps),
    )

    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    train_iter = iter(train_loader)
    model.train()
    running_loss = 0.0
    running_count = 0
    start = time.time()
    best_val_loss = float("inf")

    for step in range(1, args.steps + 1):
        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            batch = next(train_iter)

        batch = {k: v.to(device) for k, v in batch.items()}
        output = model(
            batch["tokens"],
            batch["segment_ids"],
            padding_mask=batch["padding_mask"],
            targets=batch["targets"],
        )
        optimizer.zero_grad()
        output["loss"].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()
        scheduler.step()

        running_loss += output["loss"].item()
        running_count += 1

        if step % args.log_every == 0:
            elapsed = time.time() - start
            tokens_per_step = args.batch_size * args.max_seq_len
            print(
                f"step={step} loss={running_loss / running_count:.4f} "
                f"lr={scheduler.get_last_lr()[0]:.2e} "
                f"tokens/s={tokens_per_step * args.log_every / elapsed:.0f}",
                flush=True,
            )
            running_loss = 0.0
            running_count = 0
            start = time.time()

        if step % args.eval_every == 0 or step == args.steps:
            val_loss, val_acc = evaluate(model, val_loader, device, args.eval_batches)
            print(
                f"step={step} val_loss={val_loss:.4f} val_acc={val_acc:.4f}", flush=True
            )
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                checkpoint = {
                    "step": step,
                    "model_state": model.state_dict(),
                    "model_config": asdict(model_config),
                    "val_loss": val_loss,
                    "val_acc": val_acc,
                }
                path = checkpoint_dir / "masked_stream_transformer.pt"
                torch.save(checkpoint, path)
                print(f"saved checkpoint to {path}", flush=True)


if __name__ == "__main__":
    main()
