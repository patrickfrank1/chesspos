from __future__ import annotations

from typing import Sequence

import numpy as np
import torch

from src.training.window_dataset import CLS, PAD, SEP


def collate_segments(segments: Sequence[np.ndarray]) -> dict[str, torch.Tensor]:
    if not segments:
        raise ValueError("segments cannot be empty")
    lengths = [len(segment) + 2 for segment in segments]
    max_len = max(lengths)
    tokens = np.full((len(segments), max_len), PAD, dtype=np.int64)
    for row, segment in enumerate(segments):
        tokens[row, 0] = CLS
        tokens[row, 1 : 1 + len(segment)] = segment
        tokens[row, 1 + len(segment)] = SEP
    segment_ids = np.cumsum(tokens == SEP, axis=1)
    return {
        "tokens": torch.from_numpy(tokens),
        "segment_ids": torch.from_numpy(segment_ids),
        "padding_mask": torch.from_numpy(tokens != PAD),
    }


@torch.no_grad()
def embed_tokens(
    model: torch.nn.Module,
    segments: Sequence[np.ndarray],
    device: torch.device | str,
    *,
    batch_size: int = 256,
    bf16: bool | None = None,
) -> torch.Tensor:
    """Embed segments into L2-normalized CPU float32 vectors."""
    device = torch.device(device) if isinstance(device, str) else device
    segments = list(segments)
    if not segments:
        return torch.zeros((0, 0), dtype=torch.float32)
    enabled = (device.type == "cuda") if bf16 is None else bf16
    was_training = model.training
    model.eval()
    vectors = []
    try:
        for start in range(0, len(segments), batch_size):
            batch = {
                key: value.to(device)
                for key, value in collate_segments(
                    segments[start : start + batch_size]
                ).items()
            }
            with torch.autocast(
                device_type=device.type, dtype=torch.bfloat16, enabled=enabled
            ):
                pooled, _ = model.embed(
                    batch["tokens"],
                    batch["segment_ids"],
                    padding_mask=batch["padding_mask"],
                )
            vectors.append(pooled[:, 0, :].float().cpu())
    finally:
        model.train(was_training)
    return torch.nn.functional.normalize(torch.cat(vectors), dim=-1)
