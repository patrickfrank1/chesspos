from __future__ import annotations

import torch

from src.modeling.transformer import MaskedStreamTransformer, TransformerConfig


def load_checkpoint(
    path: str, device: torch.device | str
) -> tuple[MaskedStreamTransformer, dict]:
    """Load a checkpoint and its training metadata.

    Foundation checkpoints store ``model_config``/``model_state``; classifier
    checkpoints store the encoder under ``encoder_config``/``encoder_state``.
    """
    device = torch.device(device) if isinstance(device, str) else device
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if "model_config" in checkpoint and "model_state" in checkpoint:
        config = TransformerConfig(**checkpoint["model_config"])
        state = checkpoint["model_state"]
    elif "encoder_config" in checkpoint and "encoder_state" in checkpoint:
        config = TransformerConfig(**checkpoint["encoder_config"])
        state = checkpoint["encoder_state"]
    else:
        raise ValueError(
            "Checkpoint must contain model_config/model_state or "
            "encoder_config/encoder_state"
        )
    model = MaskedStreamTransformer(config)
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
        "content_pooling": bool(getattr(config, "content_pooling", False)),
        "aux_heads": bool(getattr(config, "aux_heads", False)),
    }
    return model, info


def load_encoder(path: str, device: torch.device | str) -> MaskedStreamTransformer:
    return load_checkpoint(path, device)[0]
