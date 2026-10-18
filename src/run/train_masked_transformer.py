from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from src.evaluation.eval_suite import full_eval
from src.modeling.transformer import MaskedStreamTransformer, TransformerConfig
from src.training.checkpoint_hub import (
    build_metadata,
    collect_data_provenance,
    upload_checkpoint,
    upload_log,
)
from src.training.hf_data import download_split
from src.training.loop import (
    evaluate,
    get_git_commit,
    lr_lambda,
    make_logger,
    save_checkpoint,
    write_eval_artifacts,
)
from src.training.mlflow_tracker import MLflowTracker, resolve_tracking_uri
from src.training.window_dataset import (
    WindowDataset,
    WindowDatasetConfig,
    _load_game_segments,
    collate_windows,
    inverse_frequency_weights,
    unigram_counts,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train masked stream transformer")
    parser.add_argument("--train-dir", default="data/processed/train")
    parser.add_argument("--val-dir", default="data/processed/test")
    parser.add_argument(
        "--data-repo",
        default=None,
        help="HuggingFace dataset repo id; when set, train/val data is "
        "snapshot-downloaded from it and the dirs above are ignored",
    )
    parser.add_argument(
        "--data-cache",
        default=None,
        help="Cache directory for hub downloads (default: HF cache)",
    )
    parser.add_argument(
        "--data-limit-gb",
        type=float,
        default=None,
        help="Size budget for the train split download (test split is always "
        "downloaded in full); only used with --data-repo",
    )
    parser.add_argument(
        "--upload-bucket",
        default=None,
        help="HF bucket id (e.g. patrickfrank1/chesspos-checkpoints); when "
        "set, checkpoints + metadata sidecars are uploaded there",
    )
    parser.add_argument(
        "--upload-prefix",
        default="checkpoints",
        help="Path prefix inside the upload bucket (default: checkpoints)",
    )
    parser.add_argument(
        "--mlflow-uri",
        default=None,
        help="MLflow tracking server URI; falls back to MLFLOW_TRACKING_URI "
        "env var; tracking is disabled when neither is set",
    )
    parser.add_argument(
        "--mlflow-experiment",
        default="chesspos",
        help="MLflow experiment name (default: chesspos)",
    )
    parser.add_argument(
        "--run-name",
        default=None,
        help="MLflow run name (default: auto-generated)",
    )
    parser.add_argument("--checkpoint-dir", default="data/models")
    parser.add_argument("--resume", default=None)
    parser.add_argument("--steps", type=int, default=10_000)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup", type=int, default=500)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--d-model", type=int, default=512)
    parser.add_argument("--n-layers", type=int, default=8)
    parser.add_argument("--n-heads", type=int, default=16)
    parser.add_argument("--d-ff", type=int, default=2048)
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
    parser.add_argument(
        "--no-freq-weighting",
        action="store_true",
        help="Disable inverse-unigram-frequency loss weighting",
    )
    parser.add_argument(
        "--freq-cap",
        type=float,
        default=100.0,
        help="Cap on inverse-frequency loss weights (most frequent token = 1.0)",
    )
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--log-file", default=None)
    parser.add_argument("--seed", type=int, default=17)
    return parser.parse_args()


def resolve_data_dirs(args: argparse.Namespace) -> tuple[Path, Path]:
    """Resolve train/val data directories from args.

    With ``--data-repo`` both splits are snapshot-downloaded (train bounded
    by ``--data-limit-gb``, test always in full); otherwise the local
    ``--train-dir`` / ``--val-dir`` paths are used unchanged.
    """
    if args.data_repo is None:
        return Path(args.train_dir), Path(args.val_dir)
    max_bytes = (
        None if args.data_limit_gb is None else int(args.data_limit_gb * 1024**3)
    )
    train_dir = download_split(
        args.data_repo, "train", cache_dir=args.data_cache, max_bytes=max_bytes
    )
    val_dir = download_split(args.data_repo, "test", cache_dir=args.data_cache)
    return train_dir, val_dir


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

    train_dir, val_dir = resolve_data_dirs(args)
    train_dataset = WindowDataset(
        WindowDatasetConfig(
            data_dir=str(train_dir),
            max_seq_len=args.max_seq_len,
            max_segments=args.max_segments,
        )
    )
    if not args.no_freq_weighting:
        train_dataset.config.unigram_weights = inverse_frequency_weights(
            unigram_counts(train_dataset.games), cap=args.freq_cap
        )
        log("frequency-aware loss weighting enabled")
    val_games = _load_game_segments(Path(val_dir))
    val_dataset = WindowDataset(
        WindowDatasetConfig(
            games=val_games,
            max_seq_len=args.max_seq_len,
            max_segments=args.max_segments,
            seed=args.seed,
        )
    )
    log(f"train games={len(train_dataset)} val games={len(val_dataset)}")

    train_provenance = collect_data_provenance(Path(train_dir))
    val_provenance = collect_data_provenance(Path(val_dir))
    git_commit = get_git_commit()

    tracking_uri = resolve_tracking_uri(args.mlflow_uri)
    tracker = MLflowTracker(
        enabled=tracking_uri is not None,
        tracking_uri=tracking_uri,
        experiment=args.mlflow_experiment,
        run_name=args.run_name,
    )
    tracker.log_params(
        {**vars(args), "train_games": len(train_dataset), "val_games": len(val_dataset)}
    )

    def make_metadata(step: int, best_loss: float) -> dict:
        return build_metadata(
            training_config=dict(vars(args)),
            model_config=asdict(model_config),
            n_params=n_params,
            step=step,
            best_val_loss=best_loss,
            data_repo=args.data_repo,
            data_limit_gb=args.data_limit_gb,
            train_provenance=train_provenance,
            val_provenance=val_provenance,
            git_commit=git_commit,
        )

    def upload(path: Path, remote_name: str, step: int, best_loss: float) -> None:
        if not args.upload_bucket:
            return
        metadata = make_metadata(step, best_loss)
        upload_checkpoint(
            path,
            bucket_id=args.upload_bucket,
            remote_name=remote_name,
            metadata=metadata,
            prefix=args.upload_prefix,
        )
        if args.log_file and Path(args.log_file).exists():
            upload_log(
                Path(args.log_file),
                bucket_id=args.upload_bucket,
                prefix=args.upload_prefix,
            )
        tracker.log_artifact(path)
        log(f"uploaded {remote_name} to {args.upload_bucket}/{args.upload_prefix}")
        return

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
                weights=batch["weights"],
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
            avg_loss = running_loss / running_count
            tokens_per_second = tokens_per_step * args.log_every / elapsed
            log(
                f"step={step} loss={avg_loss:.4f} "
                f"lr={scheduler.get_last_lr()[0]:.2e} "
                f"tokens/s={tokens_per_second:.0f}"
            )
            tracker.log_metrics(
                {
                    "loss": avg_loss,
                    "lr": scheduler.get_last_lr()[0],
                    "tokens_per_second": tokens_per_second,
                },
                step=step,
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
            tracker.log_metrics({"val_loss": val_loss, "val_acc": val_acc}, step=step)
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
                    plot_dir=checkpoint_dir,
                    examples_dir=checkpoint_dir,
                )
                examples_path = checkpoint_dir / "neighbor_examples.json"
                if examples_path.exists():
                    for example in json.loads(examples_path.read_text()):
                        top = example["neighbors"][0]
                        log(
                            f"step={step} neighbor query material="
                            f"{example['query_material']:+.0f} {example['query_fen']}"
                        )
                        log(
                            f"  top sim={top['similarity']:.3f} material="
                            f"{top['material']:+.0f} {top['fen']}"
                        )
                    log(
                        "step={step} neighbor board report saved to {path}".format(
                            step=step,
                            path=checkpoint_dir / "neighbor_examples.html",
                        )
                    )
                log(
                    f"step={step} eval plot saved to {checkpoint_dir / 'pca_phase.png'}"
                )
                payload = {"step": step, **metrics}
                write_eval_artifacts(checkpoint_dir, payload)
                latest_eval = payload
                tracker.log_metrics(
                    {
                        key: float(value)
                        for key, value in metrics.items()
                        if isinstance(value, (int, float))
                    },
                    step=step,
                )
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
                upload(path, f"best_step{step}", step, best_val_loss)

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
            upload(path, "last", step, best_val_loss)

    tracker.end()
    return


if __name__ == "__main__":
    main()
