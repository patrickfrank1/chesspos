"""Shared instrumentation for the encoder-depth probes.

Every probe in ``docs/plan_depth_probing.md`` reads the same paired window
pool and the same per-block hidden states:

* residual drift compares consecutive block outputs,
* early exit pools the state after block ``k`` and decodes from it,
* latent truncation runs a PCA probe on the embedding at each depth,
* layer pruning skips a block and measures the causal accuracy drop.

Keeping the forward-pass plumbing in one place guarantees that all four
probes agree on what "depth ``k``" means.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from src.dataset.token_stream import EMPTY, TURN_WHITE
from src.evaluation.core.embeddings import collate_segments
from src.evaluation.eval_suite import MODE_CONFIGS, SCALAR_CLASS_BOUNDS
from src.modeling.autoencoder.model import (
    AttentionAutoencoderModel,
    restoration_targets,
)
from src.modeling.factory import build_model, model_config_from_dict
from src.modeling.masking import IGNORE_INDEX
from src.modeling.varlen import cu_seqlens, flatten_valid, scatter_valid
from src.training.window_dataset import (
    WindowDataset,
    WindowDatasetConfig,
    collate_windows,
)

SLOT_CONTENT_BOUNDS = (EMPTY, TURN_WHITE)


def load_checkpoint_model(
    path: str | Path, device: torch.device | str = "cpu"
) -> tuple[dict, AttentionAutoencoderModel]:
    """Load a training checkpoint and rebuild the model in eval mode."""
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    model = build_model(model_config_from_dict(checkpoint["model_config"]))
    model.load_state_dict(checkpoint["model_state"])
    return checkpoint, model.to(device).eval()


def count_parameters(model: torch.nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


def select_window_batches(
    games: list[list[np.ndarray]],
    *,
    windows: int,
    batch_size: int,
    seed: int,
    mode: str = "board",
    max_seq_len: int = 2048,
    max_segments: int = 30,
    device: torch.device | str = "cpu",
) -> list[dict[str, torch.Tensor]]:
    """Materialize the same paired window indices the eval suite uses."""
    if mode not in MODE_CONFIGS:
        raise ValueError(f"unknown masking mode: {mode!r}")
    dataset = WindowDataset(
        WindowDatasetConfig(
            games=games,
            max_seq_len=max_seq_len,
            max_segments=max_segments,
            masking=MODE_CONFIGS[mode],
            seed=seed,
        )
    )
    rng = np.random.default_rng(seed)
    indices = rng.permutation(len(dataset))[:windows]
    batches = []
    for start in range(0, len(indices), batch_size):
        samples = [dataset[int(i)] for i in indices[start : start + batch_size]]
        batches.append(
            {key: value.to(device) for key, value in collate_windows(samples).items()}
        )
    return batches


@torch.no_grad()
def encoder_states(
    model: AttentionAutoencoderModel,
    tokens: torch.Tensor,
    segment_ids: torch.Tensor,
    padding_mask: torch.Tensor,
    *,
    skip: int | None = None,
) -> list[torch.Tensor]:
    """Return dense hidden states ``h_0 … h_L`` in ``[batch, seq, d]`` layout.

    ``h_0`` is the embedding output and ``h_l`` is the output of encoder block
    ``l`` *before* ``ln_final``. ``skip`` omits one block, which is the
    zero-ablation used by the pruning probe.
    """
    valid = padding_mask.bool()
    x = model.encoder.embedding(tokens, segment_ids)
    cu, max_seqlen = cu_seqlens(valid)
    flat = flatten_valid(x, valid)
    states = [scatter_valid(flat, valid)]
    for index, block in enumerate(model.encoder.blocks):
        if skip is not None and index == skip:
            continue
        flat = block(flat, cu, max_seqlen)
        states.append(scatter_valid(flat, valid))
    return states


def finalize_state(
    model: AttentionAutoencoderModel, state: torch.Tensor, padding_mask: torch.Tensor
) -> torch.Tensor:
    """Apply ``ln_final`` and scatter the flat stream back to dense layout."""
    valid = padding_mask.bool()
    return scatter_valid(model.encoder.ln_final(flatten_valid(state, valid)), valid)


def latent_from_state(
    model: AttentionAutoencoderModel,
    state: torch.Tensor,
    padding_mask: torch.Tensor,
) -> torch.Tensor:
    """Pool one encoder state through the trained bottleneck."""
    hidden = finalize_state(model, state, padding_mask)
    return model.bottleneck(hidden, padding_mask)


@torch.no_grad()
def decode_logits(
    model: AttentionAutoencoderModel,
    batch: dict[str, torch.Tensor],
    latent: torch.Tensor,
) -> torch.Tensor:
    """Decode a latent and return logits at every masked token."""
    hidden = model.decode(
        latent,
        batch["tokens"],
        batch["segment_ids"],
        batch["padding_mask"],
    )
    valid = batch["targets"].ne(IGNORE_INDEX)
    return model.head(hidden[valid])


def accuracy_counts(predictions: torch.Tensor, targets: torch.Tensor) -> dict[str, int]:
    """Accumulate per-class accuracy counts for the masked-token split."""
    correct = predictions == targets
    low, high = SLOT_CONTENT_BOUNDS
    piece = targets.ge(low) & targets.lt(high)
    counts = {
        "total": int(targets.numel()),
        "correct": int(correct.sum()),
        "piece": int(piece.sum()),
        "piece_correct": int(correct[piece].sum()),
        "scalar": int((~piece).sum()),
        "scalar_correct": int(correct[~piece].sum()),
    }
    for name, (low_bound, high_bound) in SCALAR_CLASS_BOUNDS.items():
        member = targets.ge(low_bound) & targets.lt(high_bound)
        counts[name] = int(member.sum())
        counts[f"{name}_correct"] = int(correct[member].sum())
    return counts


def merge_counts(total: dict[str, int], extra: dict[str, int]) -> dict[str, int]:
    for key, value in extra.items():
        total[key] = total.get(key, 0) + value
    return total


def counts_to_metrics(counts: dict[str, int]) -> dict[str, float]:
    def ratio(hit: str, denominator: str) -> float:
        return counts.get(hit, 0) / max(1, counts.get(denominator, 0))

    metrics = {
        "masked_acc": ratio("correct", "total"),
        "piece_acc": ratio("piece_correct", "piece"),
        "scalar_acc": ratio("scalar_correct", "scalar"),
        "n_tokens": float(counts.get("total", 0)),
    }
    for name in SCALAR_CLASS_BOUNDS:
        metrics[f"{name}_acc"] = ratio(f"{name}_correct", name)
    return metrics


@torch.no_grad()
def depth_accuracy(
    model: AttentionAutoencoderModel,
    batches: list[dict[str, torch.Tensor]],
    *,
    depth: int | None = None,
    skip: int | None = None,
) -> dict[str, float]:
    """Masked-token accuracy of the encoder truncated or ablated at a block."""
    counts: dict[str, int] = {}
    for batch in batches:
        states = encoder_states(
            model,
            batch["tokens"],
            batch["segment_ids"],
            batch["padding_mask"],
            skip=skip,
        )
        state = states[-1] if depth is None else states[depth]
        latent = latent_from_state(model, state, batch["padding_mask"])
        logits = decode_logits(model, batch, latent)
        targets = batch["targets"][batch["targets"].ne(IGNORE_INDEX)]
        merge_counts(counts, accuracy_counts(logits.argmax(-1), targets))
    return counts_to_metrics(counts)


@torch.no_grad()
def residual_drift(
    model: AttentionAutoencoderModel, batches: list[dict[str, torch.Tensor]]
) -> list[dict[str, float]]:
    """Per-block relative change and cosine between consecutive states.

    Statistics are per window (mean over that row's valid tokens) and then
    averaged across the pool, so long windows do not dominate.
    """
    deltas: list[list[float]] | None = None
    cosines: list[list[float]] | None = None
    for batch in batches:
        states = encoder_states(
            model,
            batch["tokens"],
            batch["segment_ids"],
            batch["padding_mask"],
        )
        if deltas is None:
            deltas = [[] for _ in range(len(states) - 1)]
            cosines = [[] for _ in range(len(states) - 1)]
        valid = batch["padding_mask"].bool()
        for index, (previous, current) in enumerate(zip(states[:-1], states[1:])):
            prev_flat = flatten_valid(previous, valid)
            curr_flat = flatten_valid(current, valid)
            prev_rows = prev_flat.split(valid.sum(dim=1).tolist(), dim=0)
            curr_rows = curr_flat.split(valid.sum(dim=1).tolist(), dim=0)
            for before, after in zip(prev_rows, curr_rows):
                norm = float(before.norm())
                deltas[index].append(float((after - before).norm()) / max(norm, 1e-12))
                cosines[index].append(
                    float(
                        torch.nn.functional.cosine_similarity(
                            after.reshape(-1), before.reshape(-1), dim=0
                        )
                    )
                )
    deltas = deltas or []
    cosines = cosines or []
    rows = []
    for index, (delta, cosine) in enumerate(zip(deltas, cosines), start=1):
        delta_array = np.array(delta)
        cosine_array = np.array(cosine)
        rows.append(
            {
                "block": index,
                "rel_delta_mean": float(delta_array.mean()),
                "rel_delta_std": float(delta_array.std(ddof=1))
                if delta_array.size > 1
                else 0.0,
                "cos_mean": float(cosine_array.mean()),
                "cos_std": float(cosine_array.std(ddof=1))
                if cosine_array.size > 1
                else 0.0,
                "n_windows": int(delta_array.size),
            }
        )
    return rows


def pca_basis(matrix: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return ``(mean, components [D, D], explained-variance fraction)``."""
    mean = matrix.mean(dim=0, keepdim=True)
    centered = matrix - mean
    _, singular, components = torch.pca_lowrank(centered, q=centered.shape[1])
    variance = singular**2
    explained = variance / variance.sum()
    return mean, components, explained


def truncate(
    matrix: torch.Tensor, mean: torch.Tensor, components: torch.Tensor, k: int
) -> torch.Tensor:
    """Project onto the top-``k`` principal components and back."""
    basis = components[:, :k]
    return (matrix - mean) @ basis @ basis.T + mean


def effective_rank(explained: torch.Tensor, threshold: float) -> int:
    """Smallest ``k`` whose leading components reach the variance threshold."""
    cumulative = torch.cumsum(explained, dim=0)
    return int(torch.searchsorted(cumulative, torch.tensor(threshold)).item()) + 1


@torch.no_grad()
def window_latents(
    model: AttentionAutoencoderModel,
    batches: list[dict[str, torch.Tensor]],
    *,
    depth: int | None = None,
) -> torch.Tensor:
    """Encode every window once and stack its latent."""
    latents = []
    for batch in batches:
        states = encoder_states(
            model,
            batch["tokens"],
            batch["segment_ids"],
            batch["padding_mask"],
        )
        state = states[-1] if depth is None else states[depth]
        latents.append(latent_from_state(model, state, batch["padding_mask"]).float())
    return torch.cat(latents, dim=0)


@torch.no_grad()
def embed_segments_at_depth(
    model: AttentionAutoencoderModel,
    segments: list[np.ndarray],
    device: torch.device | str,
    *,
    depth: int | None = None,
    skip: int | None = None,
    batch_size: int = 64,
    bf16: bool | None = None,
) -> torch.Tensor:
    """Board embeddings from the encoder truncated at ``depth`` (L2-normalized)."""
    device = torch.device(device) if isinstance(device, str) else device
    enabled = (device.type == "cuda") if bf16 is None else bf16
    vectors = []
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
            states = encoder_states(
                model,
                batch["tokens"],
                batch["segment_ids"],
                batch["padding_mask"],
                skip=skip,
            )
            state = states[-1] if depth is None else states[depth]
            latent = latent_from_state(model, state, batch["padding_mask"])
        vectors.append(latent.float().cpu())
    return torch.nn.functional.normalize(torch.cat(vectors), dim=-1)


@torch.no_grad()
def reconstruction_accuracy(
    model: AttentionAutoencoderModel,
    batch: dict[str, torch.Tensor],
    latent: torch.Tensor,
) -> dict[str, int]:
    """Accuracy counts for one batch decoded from a (possibly truncated) latent."""
    logits = decode_logits(model, batch, latent)
    targets = batch["targets"][batch["targets"].ne(IGNORE_INDEX)]
    return accuracy_counts(logits.argmax(-1), targets)


def restoration_ground_truth(
    batch: dict[str, torch.Tensor],
) -> torch.Tensor:
    """Unmasked gold tokens for a probe batch."""
    return restoration_targets(batch["tokens"], batch["targets"])
