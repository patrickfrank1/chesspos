from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import asdict
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from src.evaluation.eval_suite import full_eval
from src.modeling.transformer import MaskedStreamTransformer, TransformerConfig
from src.training.window_dataset import (
    WindowDataset,
    WindowDatasetConfig,
    _load_game_segments,
    collate_windows,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train masked stream transformer")
    parser.add_argument("--train-dir", default="data/processed/train")
    parser.add_argument("--val-dir", default="data/processed/test")
    parser.add_argument("--checkpoint-dir", default="models")
    parser.add_argument("--resume", default=None)
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
    parser.add_argument("--save-every", type=int, default=2_000)
    parser.add_argument(
        "--full-eval", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--eval-pool-size", type=int, default=1024)
    parser.add_argument("--eval-counterfactual", type=int, default=256)
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--log-file", default=None)
    parser.add_argument("--seed", type=int, default=17)
    return parser.parse_args()


def make_logger(path: str | None):
    file = open(path, "a", encoding="utf-8") if path else None

    def log(message: str) -> None:
        print(message, flush=True)
        if file is not None:
            file.write(f"{message}\n")
            file.flush()

    return log


def lr_lambda(step: int, warmup: int, total: int):
    if step < warmup:
        return step / max(1, warmup)
    progress = (step - warmup) / max(1, total - warmup)
    return 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))


def save_checkpoint(
    path: Path,
    model: MaskedStreamTransformer,
    model_config: TransformerConfig,
    step: int,
    extra: dict,
) -> None:
    checkpoint = {
        "step": step,
        "model_state": model.state_dict(),
        "model_config": asdict(model_config),
        **extra,
    }
    tmp = path.with_suffix(".pt.tmp")
    torch.save(checkpoint, tmp)
    tmp.replace(path)


@torch.no_grad()
def evaluate(
    model: MaskedStreamTransformer,
    loader: DataLoader,
    device: torch.device,
    max_batches: int,
    bf16: bool,
) -> tuple[float, float]:
    model.eval()
    losses = []
    correct = 0
    total = 0
    for i, batch in enumerate(loader):
        if i >= max_batches:
            break
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.autocast(
            device_type=device.type, dtype=torch.bfloat16, enabled=bf16
        ):
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


def write_eval_artifacts(checkpoint_dir: Path, payload: dict) -> None:
    (checkpoint_dir / "eval_metrics.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True)
    )
    with (checkpoint_dir / "eval_history.jsonl").open("a") as file:
        file.write(json.dumps(payload, sort_keys=True) + "\n")


def main() -> None:
    args = parse_args()
    log = make_logger(args.log_file)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.set_float32_matmul_precision("high")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    bf16 = args.bf16 and device.type == "cuda"

    start_step = 0
    best_val_loss = float("inf")
    if args.resume:
        checkpoint = torch.load(args.resume, map_location="cpu", weights_only=False)
        model_config = TransformerConfig(**checkpoint["model_config"])
        start_step = checkpoint["step"]
        best_val_loss = checkpoint.get("best_val_loss", float("inf"))
        log(
            f"resuming from {args.resume} step={start_step} "
            f"best_val_loss={best_val_loss:.4f}"
        )
    else:
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
    raw_model = model
    if args.compile:
        model = torch.compile(model)
    n_params = sum(p.numel() for p in model.parameters())
    log(
        f"device={device} params={n_params / 1e6:.2f}M bf16={bf16} compile={args.compile}"
    )

    train_dataset = WindowDataset(
        WindowDatasetConfig(
            data_dir=args.train_dir,
            max_seq_len=args.max_seq_len,
            max_segments=args.max_segments,
        )
    )
    val_games = _load_game_segments(Path(args.val_dir))
    val_dataset = WindowDataset(
        WindowDatasetConfig(
            games=val_games,
            max_seq_len=args.max_seq_len,
            max_segments=args.max_segments,
            seed=args.seed,
        )
    )
    log(f"train games={len(train_dataset)} val games={len(val_dataset)}")

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

    if args.resume:
        raw_model.load_state_dict(checkpoint["model_state"])
        if "optimizer_state" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer_state"])
        if "scheduler_state" in checkpoint:
            scheduler.load_state_dict(checkpoint["scheduler_state"])

    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def autocast():
        return torch.autocast(
            device_type=device.type, dtype=torch.bfloat16, enabled=bf16
        )

    train_iter = iter(train_loader)
    model.train()
    running_loss = 0.0
    running_count = 0
    latest_eval: dict | None = None
    start = time.time()

    for step in range(start_step + 1, args.steps + 1):
        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            batch = next(train_iter)

        batch = {k: v.to(device) for k, v in batch.items()}
        with autocast():
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
            log(
                f"step={step} loss={running_loss / running_count:.4f} "
                f"lr={scheduler.get_last_lr()[0]:.2e} "
                f"tokens/s={tokens_per_step * args.log_every / elapsed:.0f}"
            )
            running_loss = 0.0
            running_count = 0
            start = time.time()

        if step % args.eval_every == 0 or step == args.steps:
            val_loss, val_acc = evaluate(
                model, val_loader, device, args.eval_batches, bf16
            )
            log(f"step={step} val_loss={val_loss:.4f} val_acc={val_acc:.4f}")
            extras = {"val_loss": val_loss, "val_acc": val_acc}
            if args.full_eval:
                metrics = full_eval(
                    raw_model,
                    val_games,
                    device,
                    windows=args.eval_batches,
                    batch_size=args.batch_size,
                    max_seq_len=args.max_seq_len,
                    max_segments=args.max_segments,
                    pool_size=args.eval_pool_size,
                    counterfactual_n=args.eval_counterfactual,
                    seed=args.seed,
                )
                payload = {"step": step, **metrics}
                write_eval_artifacts(checkpoint_dir, payload)
                latest_eval = payload
                watch = [
                    "board_turn_acc",
                    "board_castle_acc",
                    "board_piece_acc",
                    "board_prior_baseline",
                    "random_piece_acc",
                    "span_piece_acc",
                    "cos_to_move_child",
                    "cos_to_corrupted",
                    "test_r2_material",
                    "pc1_piece_corr",
                ]
                log(
                    f"step={step} "
                    + " ".join(
                        f"{key}={metrics[key]:.4f}" for key in watch if key in metrics
                    )
                )
                extras["full_eval"] = metrics
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                path = checkpoint_dir / "best.pt"
                save_checkpoint(
                    path,
                    raw_model,
                    model_config,
                    step,
                    extras,
                )
                log(f"saved best checkpoint to {path} (val_loss={val_loss:.4f})")

        if step % args.save_every == 0 or step == args.steps:
            path = checkpoint_dir / "last.pt"
            save_extra = {
                "optimizer_state": optimizer.state_dict(),
                "scheduler_state": scheduler.state_dict(),
                "best_val_loss": best_val_loss,
            }
            if latest_eval is not None:
                save_extra["full_eval"] = {
                    key: value for key, value in latest_eval.items() if key != "step"
                }
            save_checkpoint(
                path,
                raw_model,
                model_config,
                step,
                save_extra,
            )
            log(f"saved full checkpoint to {path}")


if __name__ == "__main__":
    main()
