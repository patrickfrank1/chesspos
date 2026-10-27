from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from src.evaluation.eval_suite import full_eval
from src.evaluation.retrieval import evaluate_retrieval, load_benchmark
from src.modeling.masking import MaskingConfig
from src.modeling.transformer import MaskedStreamTransformer, TransformerConfig
from src.training.checkpoint_hub import (
    build_metadata,
    collect_data_provenance,
    upload_checkpoint,
    upload_file,
    upload_log,
)
from src.training.hf_data import download_split
from src.training.loop import (
    apply_compile,
    evaluate,
    format_system_snapshot,
    get_git_commit,
    lr_lambda,
    make_logger,
    periodic_checkpoint_plan,
    save_checkpoint,
    system_snapshot,
    token_effective_rank,
    use_fused_adamw,
    write_eval_artifacts,
)
from src.training.mlflow_tracker import MLflowTracker, resolve_tracking_uri
from src.training.hard_negatives import (
    collect_hard_negatives,
    write_hard_negatives,
)
from src.training.objectives import AuxiliaryLossConfig, compute_training_loss
from src.training.preflight import check_dependencies
from src.training.window_dataset import (
    StreamingWindowDataset,
    WindowDataset,
    WindowDatasetConfig,
    load_game_pool,
    collate_windows,
    inverse_frequency_weights,
    streaming_unigram_counts,
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
    parser.add_argument(
        "--eval-pool-games",
        type=int,
        default=8192,
        help="Maximum validation games retained in the bounded prefetched "
        "eval pool (the full test split is never materialized)",
    )
    parser.add_argument("--eval-counterfactual", type=int, default=256)
    parser.add_argument(
        "--prefetch-shards",
        type=int,
        default=2,
        help="Shards preloaded ahead for training and validation streaming "
        "(0 disables). Bounds the decoded-shard cache.",
    )
    parser.add_argument(
        "--prefetch-factor",
        type=int,
        default=4,
        help="Batches prefetched per DataLoader worker",
    )
    parser.add_argument(
        "--no-pin-memory",
        action="store_true",
        help="Disable pinned memory for host-to-device batch transfers",
    )
    parser.add_argument(
        "--no-freq-weighting",
        action="store_true",
        help="Disable inverse-unigram-frequency loss weighting",
    )
    parser.add_argument(
        "--streaming",
        action="store_true",
        help=(
            "Stream training windows shard by shard instead of preloading the "
            "whole train split into memory (validation uses a bounded, "
            "prefetched eval pool)"
        ),
    )
    parser.add_argument(
        "--shuffle-buffer",
        type=int,
        default=0,
        help="Number of games held in the streaming shuffle buffer (0 disables; "
        "only used with --streaming)",
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
    parser.add_argument(
        "--retrieval-benchmark",
        default=None,
        help="Frozen retrieval benchmark directory or HuggingFace dataset repo id",
    )
    parser.add_argument("--retrieval-benchmark-revision", default=None)
    parser.add_argument("--retrieval-batch-size", type=int, default=128)
    parser.add_argument("--token-rank-tokens", type=int, default=4096)
    _masking_defaults = MaskingConfig()
    parser.add_argument(
        "--p-random",
        type=float,
        default=0.0,
        help="Probability weight for random-token masking mode",
    )
    parser.add_argument(
        "--p-board",
        type=float,
        default=0.0,
        help="Probability weight for whole-board masking mode",
    )
    parser.add_argument(
        "--p-span",
        type=float,
        default=0.0,
        help="Probability weight for span masking mode",
    )
    parser.add_argument(
        "--p-rect",
        type=float,
        default=None,
        help="Probability weight for harder rectangle masking "
        "(default 0.8, mixed with 0.2 skip-gram)",
    )
    parser.add_argument(
        "--rect-min-rate",
        type=float,
        default=_masking_defaults.rect_min_rate,
        help="Minimum fraction of board squares masked by rectangles",
    )
    parser.add_argument(
        "--rect-max-rate",
        type=float,
        default=_masking_defaults.rect_max_rate,
        help="Maximum fraction of board squares masked by rectangles",
    )
    parser.add_argument(
        "--rect-min-side",
        type=int,
        default=_masking_defaults.rect_min_side,
    )
    parser.add_argument(
        "--rect-max-side",
        type=int,
        default=_masking_defaults.rect_max_side,
    )
    parser.add_argument(
        "--no-rect-sync",
        action="store_false",
        dest="rect_synchronize",
        help="Mask independent rectangles per segment instead of the same squares",
    )
    parser.add_argument(
        "--p-skip-gram",
        type=float,
        default=None,
        help="Probability weight for skip-gram style masking (default 0.2); "
        "hide all but a small fraction of whole boards",
    )
    parser.add_argument(
        "--skip-gram-unmask-rate",
        type=float,
        default=_masking_defaults.skip_gram_unmask_rate,
        help="Fraction of boards left visible in skip-gram mode (must be < 0.30)",
    )
    parser.add_argument("--mlm-weight", type=float, default=1.0)
    parser.add_argument("--inv-weight", type=float, default=0.5)
    parser.add_argument("--var-weight", type=float, default=1.0)
    parser.add_argument("--cov-weight", type=float, default=0.1)
    parser.add_argument("--nce-weight", type=float, default=0.1)
    parser.add_argument("--nce-temperature", type=float, default=0.1)
    parser.add_argument("--alone-var-weight", type=float, default=1.0)
    parser.add_argument("--alone-cov-weight", type=float, default=0.1)
    parser.add_argument(
        "--alone-targets-per-window",
        type=int,
        default=4,
        help="Fully visible target segments per window for invariance/InfoNCE",
    )
    parser.add_argument("--vicreg-gamma", type=float, default=1.0)
    parser.add_argument("--vicreg-eps", type=float, default=1e-4)
    parser.add_argument(
        "--aux-warmup",
        type=int,
        default=200,
        help="Steps over which auxiliary loss weights ramp from 0 to target",
    )
    parser.add_argument(
        "--content-pooling",
        action="store_true",
        help="Use square-token (content) pooling for retrieval and contrastive losses",
    )
    parser.add_argument(
        "--supcon",
        action="store_true",
        help="Use multi-positive SupCon instead of window/alone InfoNCE",
    )
    parser.add_argument(
        "--turn-flip",
        action="store_true",
        help="Add the turn-flipped board as an extra SupCon positive",
    )
    parser.add_argument(
        "--temporal-radius",
        type=int,
        default=3,
        help="Ply radius for same-game SupCon positives; farther pairs are ignored",
    )
    parser.add_argument(
        "--ignore-opening-negatives",
        action="store_true",
        help="Exclude opening positions from cross-game contrastive negatives",
    )
    parser.add_argument("--opening-ply-threshold", type=int, default=30)
    parser.add_argument("--hard-negative-top-k", type=int, default=100)
    parser.add_argument("--aux-material-weight", type=float, default=0.0)
    parser.add_argument("--aux-piece-count-weight", type=float, default=0.0)
    parser.add_argument("--aux-phase-weight", type=float, default=0.0)
    parser.add_argument("--aux-turn-weight", type=float, default=0.0)
    parser.add_argument("--aux-castle-weight", type=float, default=0.0)
    parser.add_argument("--aux-ep-weight", type=float, default=0.0)
    return parser.parse_args()


def masking_config(args: argparse.Namespace) -> MaskingConfig:
    """Build the MaskingConfig implied by the parsed masking args.

    Defaults are the rect/skip-gram mixture. Setting either p_rect or
    p_skip_gram explicitly zeroes the other unless both are provided; legacy
    modes (random/board/span) zero both when selected on their own.
    """
    defaults = MaskingConfig()
    explicit_rect = args.p_rect is not None
    explicit_skip = args.p_skip_gram is not None
    legacy = bool(args.p_random or args.p_board or args.p_span)

    if explicit_rect:
        p_rect = args.p_rect
    elif legacy or explicit_skip:
        p_rect = 0.0
    else:
        p_rect = defaults.p_rect

    if explicit_skip:
        p_skip_gram = args.p_skip_gram
    elif legacy or explicit_rect:
        p_skip_gram = 0.0
    else:
        p_skip_gram = defaults.p_skip_gram

    args.p_rect = p_rect
    args.p_skip_gram = p_skip_gram
    return MaskingConfig(
        p_random=args.p_random,
        p_board=args.p_board,
        p_span=args.p_span,
        p_rect=p_rect,
        p_skip_gram=p_skip_gram,
        rect_min_rate=args.rect_min_rate,
        rect_max_rate=args.rect_max_rate,
        rect_min_side=args.rect_min_side,
        rect_max_side=args.rect_max_side,
        rect_synchronize=args.rect_synchronize,
        skip_gram_unmask_rate=args.skip_gram_unmask_rate,
    )


def auxiliary_config(args: argparse.Namespace) -> AuxiliaryLossConfig:
    """Build the weighted loss configuration from parsed args."""
    return AuxiliaryLossConfig(
        mlm_weight=args.mlm_weight,
        inv_weight=args.inv_weight,
        var_weight=args.var_weight,
        cov_weight=args.cov_weight,
        nce_weight=args.nce_weight,
        nce_temperature=args.nce_temperature,
        alone_var_weight=args.alone_var_weight,
        alone_cov_weight=args.alone_cov_weight,
        alone_targets_per_window=args.alone_targets_per_window,
        vicreg_gamma=args.vicreg_gamma,
        vicreg_eps=args.vicreg_eps,
        warmup_steps=args.aux_warmup,
        content_pooling=args.content_pooling,
        use_supcon=args.supcon,
        turn_flip=args.turn_flip,
        temporal_radius=args.temporal_radius,
        ignore_opening_negatives=args.ignore_opening_negatives,
        opening_ply_threshold=args.opening_ply_threshold,
        material_weight=args.aux_material_weight,
        piece_count_weight=args.aux_piece_count_weight,
        phase_weight=args.aux_phase_weight,
        turn_weight=args.aux_turn_weight,
        castle_weight=args.aux_castle_weight,
        ep_weight=args.aux_ep_weight,
    )


def streaming_train_config(
    args: argparse.Namespace, data_dir: str, start_step: int
) -> WindowDatasetConfig:
    """Build the streaming train dataset config, resuming the data stream.

    With ``start_step > 0`` each DataLoader worker fast-forwards its
    deterministic stream to the position the checkpoint left off, so a
    resumed run continues on fresh data instead of replaying from the start.
    """
    return WindowDatasetConfig(
        data_dir=data_dir,
        max_seq_len=args.max_seq_len,
        max_segments=args.max_segments,
        masking=masking_config(args),
        seed=args.seed,
        shuffle_buffer=args.shuffle_buffer,
        prefetch_shards=args.prefetch_shards,
        resume_step=start_step,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )


def resolve_data_dirs(args: argparse.Namespace, log=None) -> tuple[Path, Path]:
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
        args.data_repo,
        "train",
        cache_dir=args.data_cache,
        max_bytes=max_bytes,
        log=log,
    )
    val_dir = download_split(args.data_repo, "test", cache_dir=args.data_cache, log=log)
    return train_dir, val_dir


def main() -> None:
    args = parse_args()
    check_dependencies()
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
            content_pooling=args.content_pooling,
            aux_heads=any(
                weight > 0
                for weight in (
                    args.aux_material_weight,
                    args.aux_piece_count_weight,
                    args.aux_phase_weight,
                    args.aux_turn_weight,
                    args.aux_castle_weight,
                    args.aux_ep_weight,
                )
            ),
        )

    model = MaskedStreamTransformer(model_config).to(device)
    raw_model = model
    model = apply_compile(model, args.compile)
    n_params = sum(p.numel() for p in model.parameters())
    log(
        f"device={device} params={n_params / 1e6:.2f}M bf16={bf16} compile={args.compile}"
    )

    train_dir, val_dir = resolve_data_dirs(args, log=log)
    if args.streaming:
        train_dataset = StreamingWindowDataset(
            streaming_train_config(args, data_dir=str(train_dir), start_step=start_step)
        )
        if not args.no_freq_weighting:
            train_dataset.config.unigram_weights = inverse_frequency_weights(
                streaming_unigram_counts(str(train_dir), log=log), cap=args.freq_cap
            )
            log("frequency-aware loss weighting enabled (streaming counts)")
    else:
        train_dataset = WindowDataset(
            WindowDatasetConfig(
                data_dir=str(train_dir),
                max_seq_len=args.max_seq_len,
                max_segments=args.max_segments,
                masking=masking_config(args),
            )
        )
        if not args.no_freq_weighting:
            train_dataset.config.unigram_weights = inverse_frequency_weights(
                unigram_counts(train_dataset.games), cap=args.freq_cap
            )
            log("frequency-aware loss weighting enabled")
    log(f"loading val split from {val_dir} ...")
    val_games = load_game_pool(
        Path(val_dir),
        args.eval_pool_games,
        prefetch_shards=args.prefetch_shards,
    )
    log(
        f"loaded {len(val_games)} val games "
        f"(pool cap={args.eval_pool_games}, "
        f"prefetch_shards={args.prefetch_shards})"
    )
    val_dataset = WindowDataset(
        WindowDatasetConfig(
            games=val_games,
            max_seq_len=args.max_seq_len,
            max_segments=args.max_segments,
            masking=masking_config(args),
            seed=args.seed,
        )
    )
    log(f"train games={len(train_dataset)} val games={len(val_dataset)}")

    retrieval_benchmark = None
    if args.retrieval_benchmark is not None:
        retrieval_benchmark = load_benchmark(
            args.retrieval_benchmark, revision=args.retrieval_benchmark_revision
        )
        log(
            "retrieval benchmark "
            f"positions={len(retrieval_benchmark.positions)} "
            f"near_queries={len(retrieval_benchmark.near_queries)} "
            f"strategic_queries="
            f"{sum(ids.size for ids in retrieval_benchmark.strategic_queries.values())}"
        )

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

    pin_memory = not args.no_pin_memory and device.type == "cuda"
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=not args.streaming,
        num_workers=args.num_workers,
        collate_fn=collate_windows,
        drop_last=True,
        persistent_workers=args.num_workers > 0,
        pin_memory=pin_memory,
        prefetch_factor=args.prefetch_factor if args.num_workers > 0 else None,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_windows,
        persistent_workers=args.num_workers > 0,
        pin_memory=pin_memory,
        prefetch_factor=args.prefetch_factor if args.num_workers > 0 else None,
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
        fused=use_fused_adamw(device),
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
    running_components: dict[str, float] = {}
    latest_eval: dict | None = None
    latest_metrics: dict = {}

    def save_last_checkpoint(step: int) -> None:
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
        save_extra.update(latest_metrics)
        save_checkpoint(
            path,
            raw_model,
            model_config,
            step,
            save_extra,
        )
        log(f"saved full checkpoint to {path}")
        upload(path, "last", step, best_val_loss)
        return

    start = time.time()
    loss_config = auxiliary_config(args)
    target_generator = torch.Generator(device=device).manual_seed(args.seed)
    log(
        "masking weights: "
        f"rect={args.p_rect:.2f} skip_gram={args.p_skip_gram:.2f} "
        f"random={args.p_random:.2f} board={args.p_board:.2f} "
        f"span={args.p_span:.2f}"
    )
    log(
        "loss weights: "
        f"mlm={loss_config.mlm_weight:.3f} inv={loss_config.inv_weight:.3f} "
        f"var={loss_config.var_weight:.3f} cov={loss_config.cov_weight:.3f} "
        f"nce={loss_config.nce_weight:.3f} warmup={loss_config.warmup_steps}"
    )
    if loss_config.nce_weight == 0:
        log("InfoNCE disabled (nce_weight=0)")

    for step in range(start_step + 1, args.steps + 1):
        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            batch = next(train_iter)

        batch = {k: v.to(device, non_blocking=pin_memory) for k, v in batch.items()}
        with autocast():
            loss, components = compute_training_loss(
                model,
                batch,
                loss_config,
                step=step,
                generator=target_generator,
            )
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()
        scheduler.step()

        running_loss += loss.item()
        running_count += 1
        for key, value in components.items():
            running_components[key] = running_components.get(key, 0.0) + value.item()

        if step % args.log_every == 0:
            elapsed = time.time() - start
            tokens_per_step = args.batch_size * args.max_seq_len
            avg_loss = running_loss / running_count
            tokens_per_second = tokens_per_step * args.log_every / elapsed
            component_means = {
                f"{key}_loss": value / running_count
                for key, value in running_components.items()
            }
            log(
                f"step={step} loss={avg_loss:.4f} "
                + " ".join(
                    f"{key}={value:.4f}"
                    for key, value in sorted(component_means.items())
                )
                + f" lr={scheduler.get_last_lr()[0]:.2e} "
                f"tokens/s={tokens_per_second:.0f}"
            )
            log(f"step={step} system {format_system_snapshot(system_snapshot())}")
            tracker.log_metrics(
                {
                    "loss": avg_loss,
                    **component_means,
                    "lr": scheduler.get_last_lr()[0],
                    "tokens_per_second": tokens_per_second,
                },
                step=step,
            )
            running_loss = 0.0
            running_count = 0
            running_components = {}
            start = time.time()

        actions = periodic_checkpoint_plan(
            step,
            steps=args.steps,
            save_every=args.save_every,
            eval_every=args.eval_every,
        )
        if actions.save_before_eval:
            save_last_checkpoint(step)
        if actions.evaluate:
            val_loss, val_acc = evaluate(
                model, val_loader, device, args.eval_batches, bf16
            )
            log(f"step={step} val_loss={val_loss:.4f} val_acc={val_acc:.4f}")
            extras = {"val_loss": val_loss, "val_acc": val_acc}
            tracker.log_metrics({"val_loss": val_loss, "val_acc": val_acc}, step=step)
            token_rank = token_effective_rank(
                raw_model,
                val_loader,
                device,
                args.eval_batches,
                bf16,
                max_tokens=args.token_rank_tokens,
                seed=args.seed,
            )
            log(f"step={step} token_rank_effective={token_rank:.2f}")
            extras["token_rank_effective"] = token_rank
            tracker.log_metrics({"token_rank_effective": token_rank}, step=step)
            if retrieval_benchmark is not None:
                retrieval_metrics = evaluate_retrieval(
                    raw_model,
                    retrieval_benchmark,
                    device,
                    batch_size=args.retrieval_batch_size,
                )
                log(
                    f"step={step} "
                    + " ".join(
                        f"retrieval_{key}={value:.4f}"
                        for key, value in sorted(retrieval_metrics.items())
                    )
                )
                extras.update(
                    {
                        f"retrieval_{key}": value
                        for key, value in retrieval_metrics.items()
                    }
                )
                tracker.log_metrics(
                    {
                        f"retrieval_{key}": float(value)
                        for key, value in retrieval_metrics.items()
                    },
                    step=step,
                )
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
                    bf16=bf16,
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
                    "mlm_board_turn_acc",
                    "mlm_board_castle_acc",
                    "mlm_board_piece_acc",
                    "mlm_board_prior_baseline",
                    "mlm_random_piece_acc",
                    "mlm_span_piece_acc",
                    "mean_top10_cosine",
                    "mean_random_pair_cosine",
                    "cos_to_move_child",
                    "cos_to_second_child",
                    "cos_to_corrupted",
                    "cos_to_random_position",
                    "test_r2_material",
                    "test_r2_material_ci_low",
                    "test_r2_material_ci_high",
                    "test_acc_side_to_move",
                    "pc1_piece_corr",
                    "pc1_piece_corr_ci_low",
                    "pc1_piece_corr_ci_high",
                    "pc2_piece_corr",
                    "pc2_piece_corr_ci_low",
                    "pc2_piece_corr_ci_high",
                ]
                log(
                    f"step={step} "
                    + " ".join(
                        f"{key}={metrics[key]:.4f}" for key in watch if key in metrics
                    )
                )
                extras["full_eval"] = metrics
            hard_records = collect_hard_negatives(
                raw_model,
                batch,
                loss_config,
                step=step,
                top_k=args.hard_negative_top_k,
                generator=torch.Generator(device=device).manual_seed(args.seed + step),
            )
            hard_path = write_hard_negatives(hard_records, checkpoint_dir, step)
            if hard_path is not None:
                log(f"step={step} hard negatives saved to {hard_path}")
                if args.upload_bucket:
                    upload_file(
                        hard_path,
                        bucket_id=args.upload_bucket,
                        remote_path=(
                            f"{args.upload_prefix}/hard_negatives/{hard_path.name}"
                        ),
                    )
            latest_metrics = {
                key: value for key, value in extras.items() if key != "full_eval"
            }
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

        if actions.save_after_eval:
            save_last_checkpoint(step)

    tracker.end()
    return


if __name__ == "__main__":
    main()
