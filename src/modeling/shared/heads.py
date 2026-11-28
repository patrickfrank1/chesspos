"""Vocabulary-head helpers shared by the training loop and the eval suite."""

from __future__ import annotations

import torch
from torch import nn


def chunked_argmax(
    head: nn.Module,
    hidden: torch.Tensor,
    chunks: int = 8,
) -> torch.Tensor:
    """Return ``head(hidden).argmax(-1)`` without scoring every row at once.

    The ``[tokens, vocab]`` logits tensor is the largest single activation in
    the eval paths. Evaluation only needs the argmax, so scoring one slice at a
    time bounds the peak without changing the result.
    """
    if hidden.shape[0] == 0:
        return hidden.new_zeros((0,), dtype=torch.long)
    if chunks <= 1:
        return head(hidden).argmax(-1)
    predictions = [
        head(part).argmax(-1)
        for part in torch.tensor_split(hidden, chunks)
        if part.shape[0] > 0
    ]
    return torch.cat(predictions)
