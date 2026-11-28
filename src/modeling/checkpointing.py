"""Activation checkpointing that keeps the FlashAttention dtype path."""

from __future__ import annotations

import contextlib

import torch
from torch.utils.checkpoint import checkpoint

_AUTOCAST_DEVICES = {"cuda": torch.bfloat16}


def autocast_for(device_type: str):
    """Autocast context used for the block loop on ``device_type``.

    The block loop enters this context itself, so the recompute pass run by
    checkpointing re-enters it too. That keeps :func:`varlen_attention` on the
    flash kernel during recompute; without it the backward pass would run in
    fp32 and silently fall back to SDPA — a ~2x attention slowdown that looks
    like "checkpointing is expensive".

    ``torch.utils.checkpoint.checkpoint(..., context_fn=...)`` would express the
    same thing, but Dynamo cannot trace it (``checkpoint not implemented for
    ... context_fn``), so the context lives inside the checkpointed callable.
    """
    dtype = _AUTOCAST_DEVICES.get(device_type)
    if dtype is None:
        return contextlib.nullcontext()
    return torch.autocast(device_type, dtype=dtype)


def run_checkpointed(fn, *args, enabled: bool):
    """Call ``fn`` under activation checkpointing when ``enabled``."""
    if not enabled:
        return fn(*args)
    return checkpoint(
        fn,
        *args,
        use_reentrant=False,
        preserve_rng_state=True,
    )
