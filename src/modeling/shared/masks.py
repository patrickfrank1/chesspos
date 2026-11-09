"""Segment-table and isolation-mask helpers."""

from __future__ import annotations

import torch

from src.dataset.token_stream import EMPTY, N_SLOTS, SEGMENT_LEN, TURN_WHITE
from src.modeling.shared.embeddings import slot_indices


def build_segment_tables(
    tokens: torch.Tensor,
    segment_ids: torch.Tensor,
    padding_mask: torch.Tensor,
    max_segments: int,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    """Gather packed board tokens into ``[batch, segments, 67]`` tables.

    Returns ``token_table``, ``token_valid``, ``segment_valid``,
    ``flat_index`` (the table location of each packed token), and
    ``board_valid`` (which packed positions are board tokens).
    """
    batch, seq_len = tokens.shape
    device = tokens.device
    slot = slot_indices(tokens, segment_ids)
    segment = segment_ids.clamp(0, max_segments - 1)
    board_valid = (
        padding_mask.bool()
        & slot.lt(SEGMENT_LEN)
        & segment_ids.ge(0)
        & segment_ids.lt(max_segments)
    )
    batch_index = torch.arange(batch, device=device).unsqueeze(1).expand(batch, seq_len)
    flat_index = (batch_index * max_segments + segment) * SEGMENT_LEN + slot.clamp(
        0, SEGMENT_LEN - 1
    )
    safe_index = torch.where(board_valid, flat_index, torch.zeros_like(flat_index))
    source = torch.where(board_valid, tokens, torch.zeros_like(tokens))

    table_size = batch * max_segments * SEGMENT_LEN
    token_table = tokens.new_zeros(table_size)
    token_table.scatter_(0, safe_index.reshape(-1), source.reshape(-1))
    valid_counts = torch.zeros(table_size, dtype=torch.int32, device=device)
    valid_counts.scatter_add_(
        0, safe_index.reshape(-1), board_valid.reshape(-1).to(torch.int32)
    )
    token_table = token_table.view(batch, max_segments, SEGMENT_LEN)
    token_valid = valid_counts.view(batch, max_segments, SEGMENT_LEN) > 0
    segment_valid = token_valid.any(dim=-1)
    return token_table, token_valid, segment_valid, flat_index, board_valid


def encoder_isolation_mask(
    segment_len: int = SEGMENT_LEN, n_slots: int = N_SLOTS
) -> torch.Tensor:
    """Content queries see content keys; state queries see every slot."""
    slots = torch.arange(segment_len)
    content = slots < n_slots
    return (content[:, None] & content[None, :]) | ~content[:, None]


def decoder_isolation_mask(
    segment_len: int = SEGMENT_LEN, n_slots: int = N_SLOTS
) -> torch.Tensor:
    """Content decoder queries see content queries; state queries see all."""
    return encoder_isolation_mask(segment_len, n_slots)


def segment_selection_mask(
    tokens: torch.Tensor,
    segment_ids: torch.Tensor,
    padding_mask: torch.Tensor,
    max_segments: int,
) -> torch.Tensor:
    """Return which segments contain at least one content token."""
    content = (
        padding_mask.bool()
        & (tokens >= EMPTY)
        & (tokens < TURN_WHITE)
        & segment_ids.ge(0)
        & segment_ids.lt(max_segments)
    )
    counts = torch.zeros(
        (tokens.shape[0], max_segments), dtype=torch.int32, device=tokens.device
    )
    index = segment_ids.clamp(0, max_segments - 1)
    counts.scatter_add_(1, index, content.to(torch.int32))
    return counts > 0


def scatter_to_segment_table(
    values: torch.Tensor,
    flat_index: torch.Tensor,
    board_valid: torch.Tensor,
    max_segments: int,
) -> torch.Tensor:
    """Scatter packed values into a ``[batch, segments, 67]`` table."""
    batch = values.shape[0]
    safe_index = torch.where(board_valid, flat_index, torch.zeros_like(flat_index))
    source = torch.where(board_valid, values, torch.zeros_like(values))
    table = values.new_zeros(batch * max_segments * SEGMENT_LEN)
    table.scatter_(0, safe_index.reshape(-1), source.reshape(-1))
    return table.view(batch, max_segments, SEGMENT_LEN)
