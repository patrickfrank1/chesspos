"""Perceiver decoder producing per-slot hidden states."""

from __future__ import annotations

import torch
from torch import nn

from src.dataset.token_stream import N_SLOTS, SEGMENT_LEN
from src.modeling.perceiver.config import PerceiverIOConfig
from src.modeling.perceiver.cross_attention import SegmentCrossAttention
from src.modeling.shared.layers import TransformerLayer
from src.modeling.shared.masks import decoder_isolation_mask


class PerceiverDecoder(nn.Module):
    """Decode slot queries from contextual content and state latents.

    Content slot queries read only content latents; state slot queries read
    only state latents. The decoder's self-attention preserves that separation:
    content queries attend content queries, state queries attend every slot.
    """

    def __init__(self, config: PerceiverIOConfig, slot_embedding: nn.Embedding) -> None:
        super().__init__()
        self.config = config
        object.__setattr__(self, "slot_embedding", slot_embedding)
        self.content_cross = SegmentCrossAttention(
            config.d_model, config.n_heads, config.dropout
        )
        self.state_cross = SegmentCrossAttention(
            config.d_model, config.n_heads, config.dropout
        )
        self.blocks = nn.ModuleList(
            TransformerLayer(
                config.d_model, config.n_heads, config.d_ff, config.dropout
            )
            for _ in range(config.n_decoder_layers)
        )
        self.register_buffer(
            "isolation_mask",
            decoder_isolation_mask(config.segment_len),
            persistent=False,
        )
        return

    def forward(
        self,
        content_latents: torch.Tensor,
        state_latents: torch.Tensor,
        token_valid: torch.Tensor,
        segment_valid: torch.Tensor,
    ) -> torch.Tensor:
        batch, n_segments, _, d_model = content_latents.shape
        slots = torch.arange(SEGMENT_LEN, device=content_latents.device)
        queries = self.slot_embedding(slots)[None, None, :, :].expand(
            batch, n_segments, -1, -1
        )
        content_queries = queries[:, :, :N_SLOTS, :]
        state_queries = queries[:, :, N_SLOTS:, :]
        content = self.content_cross(content_queries, content_latents, segment_valid)
        state = self.state_cross(state_queries, state_latents, segment_valid)
        decoded = torch.cat([content, state], dim=2)
        flat = decoded.reshape(batch * n_segments, SEGMENT_LEN, d_model)
        key_mask = token_valid.reshape(batch * n_segments, SEGMENT_LEN)
        for block in self.blocks:
            flat = block(
                flat,
                key_padding_mask=key_mask,
                attn_mask=self.isolation_mask,
            )
        return flat.view(batch, n_segments, SEGMENT_LEN, d_model)
