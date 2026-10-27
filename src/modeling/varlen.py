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
    max_seqlen = int(lengths.max().item()) if lengths.numel() else 0
    return cumulative.to(torch.int32), max_seqlen


def flatten_valid(x: torch.Tensor, padding_mask: torch.Tensor) -> torch.Tensor:
    """Gather the valid positions of a padded ``[B, S, ...]`` tensor to ``[T, ...]``."""
    return x[padding_mask]


def scatter_valid(flat: torch.Tensor, padding_mask: torch.Tensor) -> torch.Tensor:
    """Inverse of :func:`flatten_valid`: write ``[T, ...]`` values back into ``[B, S, ...]``."""
    out = torch.zeros(
        (*padding_mask.shape, flat.shape[-1]), dtype=flat.dtype, device=flat.device
    )
    out[padding_mask] = flat
    return out


def varlen_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    max_seqlen_q: int,
    dropout_p: float = 0.0,
) -> torch.Tensor:
    """Bidirectional attention over concatenated variable-length segments.

    ``q``, ``k``, ``v`` are ``[T, n_heads, head_dim]``; segments are delimited
    by ``cu_seqlens_q``. Uses the flash-attn varlen kernel on CUDA when the
    ``flash_attn`` package is installed, otherwise a per-segment SDPA loop.
    """
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
            cu_seqlens_q,
            max_seqlen_q,
            max_seqlen_q,
            dropout_p=dropout_p,
            causal=False,
        )
    outputs = []
    starts = cu_seqlens_q[:-1].tolist()
    ends = cu_seqlens_q[1:].tolist()
    for start, end in zip(starts, ends):
        if end == start:
            continue
        segment_q = q[start:end].transpose(0, 1).unsqueeze(0)
        segment_k = k[start:end].transpose(0, 1).unsqueeze(0)
        segment_v = v[start:end].transpose(0, 1).unsqueeze(0)
        attended = F.scaled_dot_product_attention(
            segment_q, segment_k, segment_v, dropout_p=dropout_p
        )
        outputs.append(attended.squeeze(0).transpose(0, 1))
    if not outputs:
        return q.new_zeros(q.shape)
    return torch.cat(outputs, dim=0)
