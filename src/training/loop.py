"""Shared building blocks for the masked transformer training loop.

Both the standalone training script (``src/run/train_masked_transformer.py``)
and the Ray Train loop (``src/training/ray_train.py``) use these helpers so
the two entry points train identical models with identical evaluation.
"""

from __future__ import annotations

import json
import math
import os
import subprocess
from dataclasses import asdict
from pathlib import Path
from typing import NamedTuple

import torch
from torch.utils.data import DataLoader

from src.dataset.token_stream import CLS, MASK, SEP
from src.evaluation.retrieval import effective_rank
from src.modeling.masking import IGNORE_INDEX
from src.modeling.transformer import MaskedStreamTransformer, TransformerConfig


def use_fused_adamw(device: torch.device) -> bool:
    """Whether the fused CUDA AdamW kernel can be used on ``device``."""
    return device.type == "cuda" and torch.cuda.is_available()


def apply_compile(model: torch.nn.Module, enabled: bool) -> torch.nn.Module:
    """Wrap ``model`` with torch.compile when enabled.

    ``dynamic=True`` lets the varlen valid-token count vary across batches
    without triggering a recompilation for every new shape.
    """
    if not enabled:
        return model
    return torch.compile(model, dynamic=True)


def make_logger(path: str | None):
    file = open(path, "a", encoding="utf-8") if path else None

    def log(message: str) -> None:
        print(message, flush=True)
        if file is not None:
            file.write(f"{message}\n")
            file.flush()
        return

    return log


def rss_gigabytes() -> float:
    """Return this process's resident set size in GiB."""
    try:
        with open("/proc/self/status", encoding="utf-8") as status:
            for line in status:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) / 1024**2
    except OSError:
        pass
    import resource

    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**2


def cpu_load_average() -> float:
    """Return the one-minute load average, or 0.0 when unavailable."""
    try:
        return os.getloadavg()[0]
    except (AttributeError, OSError):
        return 0.0


def _nvidia_smi_utilization() -> float | None:
    """Read GPU utilization via the nvidia-smi CLI (fallback for missing pynvml)."""
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=utilization.gpu",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=5,
            check=True,
        )
        return float(result.stdout.strip().splitlines()[0])
    except Exception:
        return None


def gpu_snapshot() -> dict[str, float] | None:
    """Return VRAM usage and GPU utilization, or None without CUDA."""
    if not torch.cuda.is_available():
        return None
    free, total = torch.cuda.mem_get_info()
    snapshot = {
        "vram_alloc_gb": torch.cuda.memory_allocated() / 1024**3,
        "vram_used_gb": (total - free) / 1024**3,
        "vram_total_gb": total / 1024**3,
    }
    try:
        snapshot["gpu_util_pct"] = float(torch.cuda.utilization())
    except Exception:
        util = _nvidia_smi_utilization()
        if util is not None:
            snapshot["gpu_util_pct"] = util
    return snapshot


def system_snapshot() -> dict[str, float]:
    """Collect CPU-side and (when present) GPU-side resource metrics."""
    snapshot: dict[str, float] = {
        "rss_gb": rss_gigabytes(),
        "cpu_load": cpu_load_average(),
        "n_cpus": float(os.cpu_count() or 1),
    }
    gpu = gpu_snapshot()
    if gpu is not None:
        snapshot.update(gpu)
    return snapshot


def format_system_snapshot(snapshot: dict[str, float]) -> str:
    return " ".join(f"{key}={value:.2f}" for key, value in snapshot.items())


def get_git_commit() -> str | None:
    """Return the current git commit, or None outside a git repo."""
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
    except Exception:
        return None


def lr_lambda(step: int, warmup: int, total: int):
    if step < warmup:
        return step / max(1, warmup)
    progress = (step - warmup) / max(1, total - warmup)
    return 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))


class PeriodicCheckpointPlan(NamedTuple):
    """Which periodic checkpoint/evaluation hooks run at a training step."""

    save_before_eval: bool
    evaluate: bool
    save_after_eval: bool


def periodic_checkpoint_plan(
    step: int,
    *,
    steps: int,
    save_every: int,
    eval_every: int,
) -> PeriodicCheckpointPlan:
    """Decide which periodic hooks run at ``step``.

    A resumable checkpoint is written before evaluation so an evaluation
    failure cannot discard the step that just finished. At the final step the
    checkpoint is saved again after evaluation so it carries the final
    evaluation metrics.
    """
    evaluate = step % eval_every == 0 or step == steps
    save = step % save_every == 0 or step == steps
    return PeriodicCheckpointPlan(
        save_before_eval=save or evaluate,
        evaluate=evaluate,
        save_after_eval=evaluate and step == steps,
    )


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
    return


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
        valid = batch["targets"].ne(IGNORE_INDEX)
        predictions = model.head(output["hidden_states"][valid]).argmax(-1)
        correct += (predictions == batch["targets"][valid]).sum().item()
        total += valid.sum().item()
    model.train()
    loss = sum(losses) / max(1, len(losses))
    accuracy = correct / max(1, total)
    return loss, accuracy


def effective_rank_from_hidden(
    hidden: torch.Tensor,
    valid: torch.Tensor,
    max_tokens: int = 4096,
    seed: int = 0,
) -> float:
    """Effective rank of the valid token states, optionally subsampled."""
    vectors = hidden[valid].detach().float().cpu()
    if vectors.numel() == 0:
        return 1.0
    if max_tokens > 0 and vectors.shape[0] > max_tokens:
        generator = torch.Generator().manual_seed(seed)
        indices = torch.randperm(vectors.shape[0], generator=generator)[:max_tokens]
        vectors = vectors[indices]
    return effective_rank(vectors.numpy())


@torch.no_grad()
def token_effective_rank(
    model: MaskedStreamTransformer,
    loader: DataLoader,
    device: torch.device,
    max_batches: int,
    bf16: bool,
    max_tokens: int = 4096,
    seed: int = 0,
) -> float:
    """Measure token-level rank on a few validation batches."""
    model.eval()
    vectors: list[torch.Tensor] = []
    for i, batch in enumerate(loader):
        if i >= max_batches:
            break
        batch = {key: value.to(device) for key, value in batch.items()}
        with torch.autocast(
            device_type=device.type, dtype=torch.bfloat16, enabled=bf16
        ):
            output = model(
                batch["tokens"],
                batch["segment_ids"],
                padding_mask=batch["padding_mask"],
            )
        tokens = batch["tokens"]
        valid = batch["padding_mask"]
        for special in (CLS, SEP, MASK):
            valid = valid & tokens.ne(special)
        vectors.append(output["hidden_states"][valid])
    model.train()
    if not vectors:
        return 1.0
    return effective_rank_from_hidden(
        torch.cat(vectors, dim=0),
        torch.ones(sum(v.shape[0] for v in vectors), dtype=torch.bool),
        max_tokens=max_tokens,
        seed=seed,
    )


def write_eval_artifacts(checkpoint_dir: Path, payload: dict) -> None:
    (checkpoint_dir / "eval_metrics.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True)
    )
    with (checkpoint_dir / "eval_history.jsonl").open("a") as file:
        file.write(json.dumps(payload, sort_keys=True) + "\n")
    return
