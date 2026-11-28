"""DeepCrossAttention (DCA) routing over a per-module residual stack.

Implements the GRN-v3 composition and the increment-based block from
``docs/plan_autoencoder_optimizations.md`` §6.2 (Heddes et al.,
arXiv:2502.06785). The stack is always passed as a **list** of tensors and never
materialised: a ``[S, T, d]`` copy would dominate memory at depth.

Scope is per module. The encoder and the decoder each own their own stack and
their own GRNs; nothing crosses the bottleneck, so the latent stays the only
path from input to reconstruction.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from src.modeling.shared.layers import FeedForward
from src.modeling.varlen import varlen_attention


class GRNv3(nn.Module):
    """Compose ``outputs`` vectors from a stack of residual states.

    ``scores = ReLU(G W_sᵀ)`` gives an input-dependent weight per slot and
    output; the learned bias ``b[o, s]`` adds a static weight. ``b`` is
    initialised to ones so the composition is the plain sum at step 0, and
    ``W_s`` to ``N(0, 1e-2)``: zero-initialising it would kill the branch,
    because ``ReLU'(0) = 0`` and no gradient would ever reach it.
    """

    def __init__(
        self,
        d_model: int,
        slots: int,
        outputs: int = 3,
        init_std: float = 1e-2,
    ) -> None:
        super().__init__()
        if slots < 1:
            raise ValueError("slots must be positive")
        if outputs < 1:
            raise ValueError("outputs must be positive")
        self.outputs = outputs
        self.proj = nn.Linear(d_model, outputs, bias=False)
        nn.init.normal_(self.proj.weight, std=init_std)
        self.bias = nn.Parameter(torch.ones(outputs, slots, d_model))
        return

    def forward(self, slots: list[torch.Tensor]) -> list[torch.Tensor]:
        """Return one composed tensor per output, all shaped like a slot."""
        if not slots:
            raise ValueError("slots cannot be empty")
        if len(slots) > self.bias.shape[1]:
            raise ValueError(
                f"expected at most {self.bias.shape[1]} slots, got {len(slots)}"
            )
        composed = [torch.zeros_like(slots[0]) for _ in range(self.outputs)]
        for index, slot in enumerate(slots):
            scores = F.relu(self.proj(slot))
            for output in range(self.outputs):
                weight = self.bias[output, index].unsqueeze(0) + scores[
                    :, output
                ].unsqueeze(-1)
                composed[output] = composed[output] + slot * weight
        return composed


class DCAStackState:
    """Track the increments that form a block's input stack.

    ``past_k == 0`` keeps the full stack ``[x0, inc_0, …, inc_{t-1}]``.
    Otherwise the stack is the paper's first+last-k form
    ``[x0, collapse_t, inc_{t-1}, …, inc_{t-k}]`` with ``collapse_t`` the sum of
    every older increment, zero-padded to a fixed slot count so every block
    sees the same static shape (no ``torch.compile`` recompiles, no mask).
    """

    def __init__(self, x0: torch.Tensor, past_k: int, slots: int) -> None:
        if past_k < 0:
            raise ValueError("past_k must be non-negative")
        if past_k > 0 and slots != past_k + 2:
            raise ValueError("first+last-k stacks need past_k + 2 slots")
        self.x0 = x0
        self.past_k = past_k
        self.slots = slots
        self.zeros = torch.zeros_like(x0)
        self.collapse = torch.zeros_like(x0)
        self.recent: list[torch.Tensor] = []
        self._increments: list[torch.Tensor] = []
        return

    def stack(self) -> list[torch.Tensor]:
        """Slots for the next block, newest first after ``collapse``."""
        if self.past_k == 0:
            return [self.x0, *self._increments]
        recent = list(reversed(self.recent))
        while len(recent) < self.past_k:
            recent.append(self.zeros)
        return [self.x0, self.collapse, *recent]

    def update(self, increment: torch.Tensor) -> None:
        """Append a block increment and advance the collapse term."""
        self._increments.append(increment)
        if self.past_k == 0:
            return
        self.recent.append(increment)
        if len(self.recent) > self.past_k:
            self.recent.pop(0)
        drop_index = len(self._increments) - 1 - self.past_k
        if drop_index >= 0:
            self.collapse = self.collapse + self._increments[drop_index]
        return


class DCALayer(nn.Module):
    """Transformer block whose q/k/v inputs are composed from the stack.

    Two variants share one implementation:

    * ``"dca"`` — three composed inputs feed three slices of the fused ``qkv``
      projection, each behind its own LayerNorm.
    * ``"grn"`` — one composed input feeds the fused ``qkv`` behind the
      original LayerNorm (the single-GRN ablation).

    ``from_standard`` maps a trained :class:`VarlenTransformerLayer` onto a DCA
    layer without reinitialising attention: the fused ``qkv`` weight is sliced
    into ``W_q``/``W_k``/``W_v`` and each composed-input LayerNorm starts as a
    copy of ``ln_attn``.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ff: int,
        dropout: float,
        *,
        variant: str = "dca",
    ) -> None:
        super().__init__()
        if variant not in ("dca", "grn"):
            raise ValueError(f"unknown DCA variant: {variant!r}")
        if d_model % n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.dropout = dropout
        self.variant = variant
        if variant == "dca":
            self.ln_q = nn.LayerNorm(d_model)
            self.ln_k = nn.LayerNorm(d_model)
            self.ln_v = nn.LayerNorm(d_model)
        else:
            self.ln_attn = nn.LayerNorm(d_model)
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.attn_out = nn.Linear(d_model, d_model)
        self.ln_ff = nn.LayerNorm(d_model)
        self.ff = FeedForward(d_model, d_ff, dropout)
        return

    @classmethod
    def from_standard(cls, layer, *, variant: str = "dca") -> "DCALayer":
        """Build a DCA layer that reproduces ``layer`` at step 0."""
        new = cls(
            layer.qkv.in_features,
            layer.n_heads,
            layer.ff.fc_in.out_features,
            layer.dropout,
            variant=variant,
        )
        new.load_from_standard(layer)
        return new

    def load_from_standard(self, layer) -> None:
        """Copy a trained standard block into this DCA block in place."""
        with torch.no_grad():
            self.qkv.weight.copy_(layer.qkv.weight)
            if layer.qkv.bias is not None and self.qkv.bias is not None:
                self.qkv.bias.copy_(layer.qkv.bias)
            self.attn_out.load_state_dict(layer.attn_out.state_dict())
            self.ln_ff.load_state_dict(layer.ln_ff.state_dict())
            self.ff.load_state_dict(layer.ff.state_dict())
            if self.variant == "dca":
                for norm in (self.ln_q, self.ln_k, self.ln_v):
                    norm.load_state_dict(layer.ln_attn.state_dict())
            else:
                self.ln_attn.load_state_dict(layer.ln_attn.state_dict())
        return

    def forward(
        self,
        stream: torch.Tensor,
        q_in: torch.Tensor,
        k_in: torch.Tensor,
        v_in: torch.Tensor,
        cu_seqlens: torch.Tensor,
        max_seqlen: int,
    ) -> torch.Tensor:
        """Return the block increment (not the updated residual stream)."""
        d_model = self.d_model
        w, b = self.qkv.weight, self.qkv.bias
        if self.variant == "dca":
            q = F.linear(
                self.ln_q(q_in), w[:d_model], None if b is None else b[:d_model]
            )
            k = F.linear(
                self.ln_k(k_in),
                w[d_model : 2 * d_model],
                None if b is None else b[d_model : 2 * d_model],
            )
            v = F.linear(
                self.ln_v(v_in),
                w[2 * d_model :],
                None if b is None else b[2 * d_model :],
            )
        else:
            hidden = self.ln_attn(q_in)
            qkv = self.qkv(hidden).view(-1, 3, self.n_heads, self.head_dim)
            q, k, v = qkv.permute(1, 0, 2, 3).unbind(0)
        q = q.view(-1, self.n_heads, self.head_dim)
        k = k.view(-1, self.n_heads, self.head_dim)
        v = v.view(-1, self.n_heads, self.head_dim)
        attended = varlen_attention(
            q,
            k,
            v,
            cu_seqlens,
            max_seqlen,
            dropout_p=self.dropout if self.training else 0.0,
        )
        attn = self.attn_out(attended.reshape(stream.shape[0], d_model))
        return attn + self.ff(self.ln_ff(q_in + attn))
