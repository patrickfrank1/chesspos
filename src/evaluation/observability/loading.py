from __future__ import annotations

import torch
from torch import nn

from src.modeling.factory import build_model, model_config_from_dict


def load_checkpoint(path: str, device: torch.device | str) -> tuple[nn.Module, dict]:
    """Load a checkpoint and its training metadata.

    Foundation checkpoints store ``model_config``/``model_state``; classifier
    checkpoints store the encoder under ``encoder_config``/``encoder_state``.
    The architecture is selected by the ``arch`` field when present and
    defaults to the legacy transformer for old checkpoints.
    """
    device = torch.device(device) if isinstance(device, str) else device
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if "model_config" in checkpoint and "model_state" in checkpoint:
        config = model_config_from_dict(checkpoint["model_config"])
        state = checkpoint["model_state"]
    elif "encoder_config" in checkpoint and "encoder_state" in checkpoint:
        config = model_config_from_dict(checkpoint["encoder_config"])
        state = checkpoint["encoder_state"]
    else:
        raise ValueError(
            "Checkpoint must contain model_config/model_state or "
            "encoder_config/encoder_state"
        )
    model = build_model(config)
    model.load_state_dict(state)
    model.to(device).eval()
    for parameter in model.parameters():
        parameter.requires_grad = False
    info = {
        "path": str(path),
        "step": checkpoint.get("step"),
        "epoch": checkpoint.get("epoch"),
        "val_loss": checkpoint.get("val_loss"),
        "val_acc": checkpoint.get("val_acc"),
        "arch": getattr(config, "arch", "transformer"),
        "content_pooling": bool(getattr(config, "content_pooling", False)),
        "aux_heads": bool(getattr(config, "aux_heads", False)),
    }
    return model, info
