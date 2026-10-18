"""Shared building blocks for the masked transformer training loop.

Both the standalone training script (``src/run/train_masked_transformer.py``)
and the Ray Train loop (``src/training/ray_train.py``) use these helpers so
the two entry points train identical models with identical evaluation.
"""

from __future__ import annotations

import json
import math
import subprocess
from dataclasses import asdict
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from src.modeling.transformer import MaskedStreamTransformer, TransformerConfig


def make_logger(path: str | None):
    file = open(path, "a", encoding="utf-8") if path else None

    def log(message: str) -> None:
        print(message, flush=True)
        if file is not None:
            file.write(f"{message}\n")
            file.flush()
        return

    return log


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
    return
