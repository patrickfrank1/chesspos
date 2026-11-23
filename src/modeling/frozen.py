from __future__ import annotations

import torch
from torch import nn

FEATURE_RECIPES = ("native", "full", "native_positions")


def _native_embed(
    encoder: nn.Module,
    tokens: torch.Tensor,
    segment_ids: torch.Tensor,
    padding_mask: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Call the native embedding, bypassing embed()'s inference decorator.

    The autoencoder's ``embed`` is decorated with ``torch.no_grad`` for
    retrieval. When the probe is fine-tuning, use the undecorated
    ``encode_content`` so gradients reach the bottleneck.
    """
    if (
        getattr(encoder, "supports_native_finetuning", False)
        and torch.is_grad_enabled()
    ):
        return encoder.encode_content(tokens, segment_ids, padding_mask=padding_mask)
    return encoder.embed(tokens, segment_ids, padding_mask=padding_mask)


def encoder_embeddings(
    encoder: nn.Module,
    tokens: torch.Tensor,
    segment_ids: torch.Tensor,
    padding_mask: torch.Tensor | None = None,
    feature: str = "native",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pool a frozen encoder into per-segment embeddings.

    ``native`` uses the checkpoint's own retrieval embedding (``embed``),
    which is content-only when the checkpoint was trained with
    ``content_pooling``. ``full`` uses the non-isolated trunk (``encode``), so
    every pooled segment carries the turn/castling/en-passant tokens.
    """
    if feature == "native":
        return _native_embed(encoder, tokens, segment_ids, padding_mask)
    if feature == "full":
        return encoder.encode(tokens, segment_ids, padding_mask=padding_mask)
    if feature == "native_positions":
        return embed_per_position(encoder, tokens, segment_ids, padding_mask)
    raise ValueError(f"feature must be one of {FEATURE_RECIPES}, got {feature!r}")


def _embed_per_position_fixed(
    encoder: nn.Module,
    tokens: torch.Tensor,
    lengths: torch.Tensor,
    stride: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Vectorized per-position embedding for fixed-width packed segments."""
    from src.dataset.token_stream import CLS, SEP

    device = tokens.device
    batch = tokens.shape[0]
    n_segments = torch.div(lengths - 1, stride, rounding_mode="floor")
    positions = int(n_segments.max().item())
    rows = torch.arange(positions, device=device)[:, None] * stride
    columns = torch.arange(stride, device=device)[None, :]
    offsets = (rows + columns).reshape(1, positions, stride) + 1
    index = offsets.expand(batch, -1, -1).clamp(max=tokens.shape[1] - 1)
    gathered = torch.gather(tokens[:, None, :].expand(-1, positions, -1), 2, index)
    windows = torch.cat([tokens.new_full((batch, positions, 1), CLS), gathered], dim=2)
    valid = torch.arange(positions, device=device)[None, :] < n_segments[:, None]
    flat = windows.reshape(batch * positions, stride + 1)
    flat_mask = valid.unsqueeze(-1).expand(batch, positions, stride + 1)
    embed_mask = flat_mask.reshape(batch * positions, stride + 1).clone()
    embed_mask[~valid.reshape(-1)] = True
    window_ids = torch.cumsum(flat == SEP, dim=1)
    vectors, _ = _native_embed(encoder, flat, window_ids, embed_mask)
    vectors = torch.nan_to_num(vectors[:, 0].reshape(batch, positions, -1))
    return vectors * valid.unsqueeze(-1).to(vectors.dtype), valid


def embed_per_position(
    encoder: nn.Module,
    tokens: torch.Tensor,
    segment_ids: torch.Tensor,
    padding_mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Embed every position of a packed row as its own single-position window.

    The attention autoencoder returns one vector per *row*, so a puzzle packed
    as one row with several positions would collapse to a single vector. This
    recipe re-packs each segment as ``CLS + segment + SEP``, embeds the batch of
    windows, and scatters the vectors back into a ``[batch, positions, d]``
    tensor with a validity mask.
    """
    from src.dataset.token_stream import CLS, PAD, SEGMENT_LEN, SEP

    if padding_mask is None:
        padding_mask = tokens.ne(PAD)
    batch = tokens.shape[0]
    device = tokens.device
    lengths = padding_mask.sum(dim=1)
    stride = SEGMENT_LEN + 1
    fixed_width = bool((lengths >= 1).all()) and bool(
        ((lengths - 1) % stride == 0).all()
    )
    if fixed_width:
        return _embed_per_position_fixed(encoder, tokens, lengths, stride)
    lengths = lengths.tolist()
    windows = []
    owners: list[tuple[int, int]] = []
    max_positions = 0
    for row in range(batch):
        length = int(lengths[row])
        if length == 0:
            continue
        row_tokens = tokens[row, :length]
        start = 0
        position = 0
        for index in torch.nonzero(row_tokens == SEP).flatten().tolist():
            body = row_tokens[start:index]
            if body.numel() and int(body[0]) == CLS:
                body = body[1:]
            window = torch.cat(
                [
                    row_tokens.new_tensor([CLS]),
                    body,
                    row_tokens.new_tensor([SEP]),
                ]
            )
            windows.append(window)
            owners.append((row, position))
            position += 1
            start = index + 1
        max_positions = max(max_positions, position)
    if not windows:
        return (
            torch.zeros(
                (batch, 0, embedding_dim(encoder, "native")),
                device=device,
                dtype=next(encoder.parameters()).dtype,
            ),
            torch.zeros((batch, 0), dtype=torch.bool, device=device),
        )

    max_len = max(window.numel() for window in windows)
    packed = windows[0].new_full((len(windows), max_len), PAD)
    for index, window in enumerate(windows):
        packed[index, : window.numel()] = window
    window_mask = packed.ne(PAD)
    window_ids = torch.cumsum(packed == SEP, dim=1)
    vectors, _ = _native_embed(encoder, packed, window_ids, window_mask)
    embeddings = vectors.new_zeros((batch, max_positions, vectors.shape[-1]))
    mask = torch.zeros((batch, max_positions), dtype=torch.bool, device=device)
    for index, (row, position) in enumerate(owners):
        embeddings[row, position] = vectors[index, 0]
        mask[row, position] = True
    return embeddings, mask


def freeze_encoder(encoder: nn.Module) -> None:
    """Freeze every encoder parameter."""
    for parameter in encoder.parameters():
        parameter.requires_grad = False
    return


def embedding_dim(encoder: nn.Module, feature: str = "native") -> int:
    """Return the width of the vectors produced by the feature recipe."""
    if feature in ("native", "native_positions"):
        return int(getattr(encoder.config, "latent_dim", encoder.config.d_model))
    if feature == "full":
        return int(encoder.config.d_model)
    raise ValueError(f"feature must be one of {FEATURE_RECIPES}, got {feature!r}")
