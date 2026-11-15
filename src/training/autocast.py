from __future__ import annotations

from contextlib import nullcontext

import torch

AMP_MODES = ("none", "bf16")


def autocast_context(device: torch.device, amp: str = "none"):
    """Return a nullcontext or a CUDA bf16 autocast context."""
    if amp not in AMP_MODES:
        raise ValueError(f"amp must be one of {AMP_MODES}, got {amp!r}")
    if amp == "bf16" and device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return nullcontext()
