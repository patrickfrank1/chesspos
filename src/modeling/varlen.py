"""Variable-length (varlen) attention helpers.

Windows in a batch hold one game each plus a PAD tail, and attention must
never cross rows. Instead of a boolean S x S mask (which disables PyTorch's
flash kernel), the valid tokens are flattened into one contiguous stream and
each row becomes a "sequence" delimited by cumulative offsets (cu_seqlens).
On CUDA the flash-attn kernel handles the whole batch in one call; elsewhere
the same computation runs as a per-segment SDPA loop, so CPU training and
tests exercise the identical code path.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

try:  # pragma: no cover - exercised only on CUDA hosts with flash-attn
    from flash_attn import flash_attn_varlen_func

    FLASH_AVAILABLE = True
except ImportError:
    FLASH_AVAILABLE = False


def cu_seqlens(padding_mask: torch.Tensor) -> tuple[torch.Tensor, int]:
    """Return cumulative token offsets plus the longest row's valid length.

    Rows may contain holes; valid tokens are gathered in row-major order, so
    a row's valid run is a single segment regardless of holes. This matches
    the masked attention semantics where every valid query may attend every
    valid key of its own row.
    """
    lengths = padding_mask.sum(dim=1, dtype=torch.int64)
    cumulative = torch.zeros(
        padding_mask.shape[0] + 1, dtype=torch.int64, device=padding_mask.device
    )
    torch.cumsum(lengths, dim=0, out=cumulative[1:])
    max_seqlen = padding_mask.shape[1]
    return cumulative.to(torch.int32), max_seqlen


def group_cu_seqlens(
    mask: torch.Tensor,
    segment_ids: torch.Tensor,
    num_groups: int,
    max_len_hint: int | None = None,
) -> tuple[torch.Tensor, int]:
    """Build flash-compatible offsets for row/segment groups.

    Groups are ordered by row and then by ``segment_ids``. Empty groups are
    omitted so every offset pair describes a non-empty sequence. The returned
    maximum length is the longest non-empty group.
    """
    if mask.shape != segment_ids.shape:
        raise ValueError("mask and segment_ids must have the same shape")
    if num_groups <= 0:
        raise ValueError("num_groups must be positive")
    batch = mask.shape[0]
    rows = torch.arange(batch, device=mask.device).unsqueeze(1)
    group = rows * num_groups + segment_ids.long()
    valid_groups = group[mask.bool()]
    counts = torch.zeros(
        batch * num_groups,
        dtype=torch.int32,
        device=mask.device,
    )
    counts.scatter_add_(
        0,
        valid_groups,
        torch.ones_like(valid_groups, dtype=torch.int32),
    )
    counts = torch.masked_select(counts, counts > 0)
    cumulative = torch.zeros(
        counts.shape[0] + 1,
        dtype=torch.int32,
        device=mask.device,
    )
    torch.cumsum(counts, dim=0, out=cumulative[1:])
    if max_len_hint is not None:
        max_len = max_len_hint
    else:
        max_len = int(counts.max().item()) if counts.numel() else 0
    return cumulative, max_len


def flatten_valid(x: torch.Tensor, padding_mask: torch.Tensor) -> torch.Tensor:
    """Gather the valid positions of a padded ``[B, S, ...]`` tensor to ``[T, ...]``.

    Uses ``masked_select`` rather than boolean indexing so ``torch.compile``
    does not have to lower an explicit ``aten.nonzero`` for the gather.
    """
    mask = padding_mask.unsqueeze(-1)
    return torch.masked_select(x, mask).view(-1, x.shape[-1])


def scatter_valid(flat: torch.Tensor, padding_mask: torch.Tensor) -> torch.Tensor:
    """Inverse of :func:`flatten_valid`: write ``[T, ...]`` values back into ``[B, S, ...]``.

    Uses ``masked_scatter`` instead of boolean assignment; both operators
    preserve row-major order, so the result matches ``out[mask] = flat`` while
    avoiding an explicit ``aten.nonzero`` in the compiled graph.
    """
    out = torch.zeros(
        (*padding_mask.shape, flat.shape[-1]), dtype=flat.dtype, device=flat.device
    )
    return out.masked_scatter(padding_mask.unsqueeze(-1), flat)


def varlen_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    max_seqlen_q: int,
    dropout_p: float = 0.0,
    *,
    cu_seqlens_k: torch.Tensor | None = None,
    max_seqlen_k: int | None = None,
) -> torch.Tensor:
    """Bidirectional attention over concatenated variable-length segments.

    ``q``, ``k``, ``v`` are ``[T, n_heads, head_dim]``. Query segments are
    delimited by ``cu_seqlens_q`` and key/value segments by ``cu_seqlens_k``
    (defaults to the query offsets). Uses the flash-attn varlen kernel on CUDA
    when the ``flash_attn`` package is installed, otherwise a per-segment SDPA
    loop.
    """
    if cu_seqlens_k is None:
        cu_seqlens_k = cu_seqlens_q
    if max_seqlen_k is None:
        max_seqlen_k = max_seqlen_q
    flash_dtypes = (torch.float16, torch.bfloat16)
    if (  # pragma: no cover - requires CUDA
        FLASH_AVAILABLE
        and q.is_cuda
        and q.dtype in flash_dtypes
        and k.dtype in flash_dtypes
        and v.dtype in flash_dtypes
    ):
        return flash_attn_varlen_func(
            q,
            k,
            v,
            cu_seqlens_q,
            cu_seqlens_k,
            max_seqlen_q,
            max_seqlen_k,
            dropout_p=dropout_p,
            causal=False,
        )
    outputs = []
    starts_q = cu_seqlens_q[:-1].tolist()
    ends_q = cu_seqlens_q[1:].tolist()
    starts_k = cu_seqlens_k[:-1].tolist()
    ends_k = cu_seqlens_k[1:].tolist()
    for start_q, end_q, start_k, end_k in zip(starts_q, ends_q, starts_k, ends_k):
        if end_q == start_q or end_k == start_k:
            continue
        segment_q = q[start_q:end_q].transpose(0, 1).unsqueeze(0)
        segment_k = k[start_k:end_k].transpose(0, 1).unsqueeze(0)
        segment_v = v[start_k:end_k].transpose(0, 1).unsqueeze(0)
        attended = F.scaled_dot_product_attention(
            segment_q, segment_k, segment_v, dropout_p=dropout_p
        )
        outputs.append(attended.squeeze(0).transpose(0, 1))
    if not outputs:
        return q.new_zeros(q.shape)
    return torch.cat(outputs, dim=0)
