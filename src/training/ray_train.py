"""Ray Train implementation of the masked transformer training job.

The training loop runs on a Ray cluster via
:class:`ray.train.torch.TorchTrainer` so data extraction (Ray Data ETL) and
model training can share one cluster. The training data is always
snapshot-downloaded from the HuggingFace hub inside each worker
(``resolve_data_dirs``); local ``train_dir``/``val_dir`` paths exist only as
an offline/test fallback and are not exposed on the CLI.

This supersedes the earlier decision recorded in
``docs/plan_gpu_training_vastai.md`` to keep Ray ETL-only.
"""

from __future__ import annotations

import tempfile
import time
from dataclasses import asdict, dataclass, fields
from pathlib import Path

import ray.train as train
import torch
import yaml
from ray.train import Checkpoint, CheckpointConfig, RunConfig, ScalingConfig
from ray.train.torch import TorchTrainer
from torch.utils.data import DataLoader

from src.evaluation.eval_suite import full_eval
from src.modeling.transformer import MaskedStreamTransformer, TransformerConfig
from src.training.checkpoint_hub import (
    build_metadata,
    collect_data_provenance,
    upload_checkpoint,
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
)


@dataclass
class RayTrainingConfig:
    """Configuration for the Ray training job.

    Field names double as YAML keys (under a ``training:`` section) and CLI
    destination names. CLI values take precedence over YAML values, which
    take precedence over the defaults below.
    """

    data_repo: str | None = None
    data_cache: str | None = None
    data_limit_gb: float | None = None
    train_dir: str | None = None
    val_dir: str | None = None
    upload_bucket: str | None = None
    upload_prefix: str = "checkpoints"
    mlflow_uri: str | None = None
    mlflow_experiment: str = "chesspos"
    run_name: str | None = None
    checkpoint_dir: str = "data/models"
    steps: int = 10_000
    batch_size: int = 8
    lr: float = 3e-4
    weight_decay: float = 0.01
    warmup: int = 500
    grad_clip: float = 1.0
    d_model: int = 256
    n_layers: int = 4
    n_heads: int = 8
    d_ff: int = 1024
    dropout: float = 0.1
    max_seq_len: int = 2048
    max_segments: int = 128
    dataloader_workers: int = 4
    log_every: int = 20
    eval_every: int = 1_000
    eval_batches: int = 20
    save_every: int = 2_000
    full_eval: bool = True
    eval_pool_size: int = 1024
    eval_counterfactual: int = 256
    bf16: bool = False
    compile: bool = False
    seed: int = 17
    num_training_workers: int = 1
    use_gpu: bool = False
    storage_path: str | None = None
    checkpoints_to_keep: int = 1

    @classmethod
    def from_dict(cls, values: dict) -> RayTrainingConfig:
        """Build a config from a mapping, ignoring unknown keys."""
        known = {field.name for field in fields(cls)}
        return cls(**{key: value for key, value in values.items() if key in known})

    def to_dict(self) -> dict:
        """Return the config as a plain dict (e.g. for ``train_loop_config``)."""
        return asdict(self)

    @property
    def model_config(self) -> TransformerConfig:
        """Build the model architecture config from the hyperparameters."""
        return TransformerConfig(
            d_model=self.d_model,
            n_layers=self.n_layers,
            n_heads=self.n_heads,
            d_ff=self.d_ff,
            dropout=self.dropout,
            max_seq_len=self.max_seq_len,
            max_segments=self.max_segments,
        )


def load_yaml_overrides(path: str | Path) -> dict:
    """Load the ``training`` section of a YAML config file."""
    document = yaml.safe_load(Path(path).read_text()) or {}
    if "training" not in document:
        raise ValueError(f"{path} must contain a 'training' section")
    overrides = document["training"] or {}
    if not isinstance(overrides, dict):
        raise ValueError(f"'training' section in {path} must be a mapping")
    return overrides


def resolve_config(args) -> RayTrainingConfig:
    """Resolve the training config with CLI > YAML > defaults precedence.

    Unset CLI options are ``None`` and fall through to the YAML config, then
    to the dataclass defaults. ``use_gpu`` auto-detects CUDA when unset;
    ``bf16`` defaults to ``use_gpu``.
    """
    values: dict = {}
    if getattr(args, "config", None):
        values.update(load_yaml_overrides(args.config))
    for field in fields(RayTrainingConfig):
        cli_value = getattr(args, field.name, None)
        if cli_value is not None:
            values[field.name] = cli_value
    if values.get("use_gpu") is None:
        values["use_gpu"] = bool(torch.cuda.is_available())
    if values.get("bf16") is None:
        values["bf16"] = values["use_gpu"]
    if values.get("full_eval") is None:
        values["full_eval"] = True
    if values.get("compile") is None:
        values["compile"] = False
    return RayTrainingConfig.from_dict(values)


def resolve_data_dirs(config: RayTrainingConfig) -> tuple[Path, Path]:
    """Resolve the train/val data directories for a training worker.

    With ``data_repo`` set, both splits are snapshot-downloaded from the
    HuggingFace hub (train bounded by ``data_limit_gb``, test in full).
    Otherwise the local ``train_dir``/``val_dir`` fallback is used.
    """
    if config.data_repo:
        max_bytes = (
            None
            if config.data_limit_gb is None
            else int(config.data_limit_gb * 1024**3)
        )
        train_dir = download_split(
            config.data_repo,
            "train",
            cache_dir=config.data_cache,
            max_bytes=max_bytes,
        )
        val_dir = download_split(config.data_repo, "test", cache_dir=config.data_cache)
        return train_dir, val_dir
    if config.train_dir and config.val_dir:
        train_dir = Path(config.train_dir)
        val_dir = Path(config.val_dir)
        if train_dir.is_dir() and val_dir.is_dir():
            return train_dir, val_dir
    raise ValueError(
        "training data source required: set data_repo (HuggingFace dataset) "
        "or existing train_dir/val_dir directories"
    )


def build_scaling_config(config: RayTrainingConfig) -> ScalingConfig:
    """Build the Ray Train scaling config from the training config."""
    return ScalingConfig(
        num_workers=config.num_training_workers, use_gpu=config.use_gpu
    )


def build_run_config(config: RayTrainingConfig) -> RunConfig:
    """Build the Ray Train run config (naming, storage, checkpoints)."""
    return RunConfig(
        name=config.run_name or "chesspos-ray",
        storage_path=config.storage_path or "ray_results",
        checkpoint_config=CheckpointConfig(num_to_keep=config.checkpoints_to_keep),
    )


def build_trainer(config: RayTrainingConfig) -> TorchTrainer:
    """Build a TorchTrainer wired to the training loop and config."""
    return TorchTrainer(
        train_loop_per_worker=train_loop_per_worker,
        train_loop_config=config.to_dict(),
        scaling_config=build_scaling_config(config),
        run_config=build_run_config(config),
    )


def train_loop_per_worker(config: dict) -> None:
    """Training loop executed on each Ray Train worker."""
    cfg = RayTrainingConfig.from_dict(config)
    log = make_logger(None)
    torch.manual_seed(cfg.seed)
    if torch.cuda.is_available():
        torch.set_float32_matmul_precision("high")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    bf16 = cfg.bf16 and device.type == "cuda"
    world_rank = train.get_context().get_world_rank()
    world_size = train.get_context().get_world_size()

    train_dir, val_dir = resolve_data_dirs(cfg)
    train_dataset = WindowDataset(
        WindowDatasetConfig(
            data_dir=str(train_dir),
            max_seq_len=cfg.max_seq_len,
            max_segments=cfg.max_segments,
        )
    )
    val_games = _load_game_segments(Path(val_dir))
    val_dataset = WindowDataset(
        WindowDatasetConfig(
            games=val_games,
            max_seq_len=cfg.max_seq_len,
            max_segments=cfg.max_segments,
            seed=cfg.seed,
        )
    )
    log(
        f"rank={world_rank} train games={len(train_dataset)} "
        f"val games={len(val_dataset)} device={device}"
    )

    model_config = cfg.model_config
    model = MaskedStreamTransformer(model_config).to(device)
    raw_model = model
    if cfg.compile:
        model = torch.compile(model)
    model = train.torch.prepare_model(model)
    n_params = sum(p.numel() for p in raw_model.parameters())

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.dataloader_workers,
        collate_fn=collate_windows,
        drop_last=True,
        persistent_workers=cfg.dataloader_workers > 0,
    )
    train_loader = train.torch.prepare_data_loader(train_loader)
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.dataloader_workers,
        collate_fn=collate_windows,
    )

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda step: lr_lambda(step, cfg.warmup, cfg.steps)
    )

    def reduce_metric(value: float) -> float:
        tensor = torch.tensor([value], dtype=torch.float64, device=device)
        if world_size > 1:
            torch.distributed.all_reduce(tensor)
            tensor /= world_size
        return tensor.item()

    start_step = 0
    best_val_loss = float("inf")
    session_checkpoint = train.get_checkpoint()
    if session_checkpoint is not None:
        with session_checkpoint.as_directory() as checkpoint_path:
            checkpoint = torch.load(
                Path(checkpoint_path) / "checkpoint.pt",
                map_location="cpu",
                weights_only=False,
            )
        raw_model.load_state_dict(checkpoint["model_state"])
        start_step = checkpoint["step"]
        best_val_loss = checkpoint.get("best_val_loss", float("inf"))
        if "optimizer_state" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer_state"])
        if "scheduler_state" in checkpoint:
            scheduler.load_state_dict(checkpoint["scheduler_state"])
        log(f"rank={world_rank} resumed from ray checkpoint step={start_step}")

    tracker = None
    if world_rank == 0:
        tracking_uri = cfg.mlflow_uri
        if tracking_uri is None:
            tracking_uri = resolve_tracking_uri(None)
        if tracking_uri:
            tracker = MLflowTracker(
                enabled=True,
                tracking_uri=tracking_uri,
                experiment=cfg.mlflow_experiment,
                run_name=cfg.run_name,
            )
            tracker.log_params(
                {
                    **cfg.to_dict(),
                    "train_games": len(train_dataset),
                    "val_games": len(val_dataset),
                }
            )

    train_provenance = collect_data_provenance(Path(train_dir))
    val_provenance = collect_data_provenance(Path(val_dir))
    git_commit = get_git_commit()

    def upload(path: Path, remote_name: str, step: int, best_loss: float) -> None:
        if not cfg.upload_bucket or world_rank != 0:
            return
        metadata = build_metadata(
            training_config=cfg.to_dict(),
            model_config=asdict(model_config),
            n_params=n_params,
            step=step,
            best_val_loss=best_loss,
            data_repo=cfg.data_repo,
            data_limit_gb=cfg.data_limit_gb,
            train_provenance=train_provenance,
            val_provenance=val_provenance,
            git_commit=git_commit,
        )
        upload_checkpoint(
            path,
            bucket_id=cfg.upload_bucket,
            remote_name=remote_name,
            metadata=metadata,
            prefix=cfg.upload_prefix,
        )
        if tracker is not None:
            tracker.log_artifact(path)
        log(
            f"rank={world_rank} uploaded {remote_name} to "
            f"{cfg.upload_bucket}/{cfg.upload_prefix}"
        )
        return

    checkpoint_dir = Path(cfg.checkpoint_dir)
    if world_rank == 0:
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

    for step in range(start_step + 1, cfg.steps + 1):
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
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        optimizer.step()
        scheduler.step()

        running_loss += output["loss"].item()
        running_count += 1

        if step % cfg.log_every == 0:
            elapsed = time.time() - start
            tokens_per_step = cfg.batch_size * cfg.max_seq_len
            avg_loss = running_loss / running_count
            tokens_per_second = tokens_per_step * cfg.log_every / elapsed
            log(
                f"step={step} loss={avg_loss:.4f} "
                f"lr={scheduler.get_last_lr()[0]:.2e} "
                f"tokens/s={tokens_per_second:.0f}"
            )
            if tracker is not None:
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

        extras: dict = {}
        if step % cfg.eval_every == 0 or step == cfg.steps:
            val_loss, val_acc = evaluate(
                model, val_loader, device, cfg.eval_batches, bf16
            )
            val_loss = reduce_metric(val_loss)
            val_acc = reduce_metric(val_acc)
            log(f"step={step} val_loss={val_loss:.4f} val_acc={val_acc:.4f}")
            extras = {"val_loss": val_loss, "val_acc": val_acc}
            if tracker is not None:
                tracker.log_metrics(
                    {"val_loss": val_loss, "val_acc": val_acc}, step=step
                )
            if cfg.full_eval and world_rank == 0:
                metrics = full_eval(
                    raw_model,
                    val_games,
                    device,
                    windows=cfg.eval_batches,
                    batch_size=cfg.batch_size,
                    max_seq_len=cfg.max_seq_len,
                    max_segments=cfg.max_segments,
                    pool_size=cfg.eval_pool_size,
                    counterfactual_n=cfg.eval_counterfactual,
                    seed=cfg.seed,
                    plot_dir=checkpoint_dir,
                    examples_dir=checkpoint_dir,
                )
                payload = {"step": step, **metrics}
                write_eval_artifacts(checkpoint_dir, payload)
                latest_eval = payload
                if tracker is not None:
                    tracker.log_metrics(
                        {
                            key: float(value)
                            for key, value in metrics.items()
                            if isinstance(value, (int, float))
                        },
                        step=step,
                    )
                extras["full_eval"] = metrics
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                if world_rank == 0:
                    path = checkpoint_dir / "best.pt"
                    save_checkpoint(path, raw_model, model_config, step, extras)
                    log(f"saved best checkpoint to {path} (val_loss={val_loss:.4f})")
                    upload(path, f"best_step{step}", step, best_val_loss)

        if step % cfg.save_every == 0 or step == cfg.steps:
            avg_loss = running_loss / max(1, running_count)
            metrics = {"step": step, "train_loss": avg_loss}
            metrics.update(
                {
                    key: value
                    for key, value in extras.items()
                    if isinstance(value, (int, float))
                }
            )
            with tempfile.TemporaryDirectory() as tmpdir:
                checkpoint = None
                if world_rank == 0:
                    save_extra: dict = {
                        "optimizer_state": optimizer.state_dict(),
                        "scheduler_state": scheduler.state_dict(),
                        "best_val_loss": best_val_loss,
                    }
                    if latest_eval is not None:
                        save_extra["full_eval"] = {
                            key: value
                            for key, value in latest_eval.items()
                            if key != "step"
                        }
                    save_checkpoint(
                        Path(tmpdir) / "checkpoint.pt",
                        raw_model,
                        model_config,
                        step,
                        save_extra,
                    )
                    local_last = checkpoint_dir / "last.pt"
                    save_checkpoint(
                        local_last, raw_model, model_config, step, save_extra
                    )
                    log(f"saved full checkpoint to {local_last}")
                    upload(local_last, "last", step, best_val_loss)
                    checkpoint = Checkpoint.from_directory(tmpdir)
                train.report(metrics, checkpoint=checkpoint)

    if tracker is not None:
        tracker.end()
    return
