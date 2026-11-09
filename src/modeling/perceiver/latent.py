"""Content and state latent banks for the Perceiver IO model."""

from __future__ import annotations

import torch
from torch import nn

from src.dataset.token_stream import N_SLOTS
from src.modeling.perceiver.config import PerceiverIOConfig
from src.modeling.perceiver.cross_attention import SegmentCrossAttention
from src.modeling.shared.layers import TransformerLayer


def _init_latents(count: int, d_model: int) -> nn.Parameter:
    parameter = nn.Parameter(torch.empty(count, d_model))
    nn.init.normal_(parameter, mean=0.0, std=0.02)
    return parameter


class LatentBottleneck(nn.Module):
    """Build contextual content and state latents independently.

    Content latents cross-attend only content slots; state latents cross-attend
    all slots. Each bank then runs local latent self-attention and
    cross-segment self-attention. The two banks never attend to each other.
    """

    def __init__(self, config: PerceiverIOConfig) -> None:
        super().__init__()
        self.config = config
        self.content_queries = _init_latents(config.n_content_latents, config.d_model)
        self.state_queries = _init_latents(config.n_state_latents, config.d_model)
        self.content_cross = SegmentCrossAttention(
            config.d_model, config.n_heads, config.dropout
        )
        self.state_cross = SegmentCrossAttention(
            config.d_model, config.n_heads, config.dropout
        )
        self.content_local = nn.ModuleList(
            TransformerLayer(
                config.d_model, config.n_heads, config.d_ff, config.dropout
            )
            for _ in range(config.n_local_latent_layers)
        )
        self.state_local = nn.ModuleList(
            TransformerLayer(
                config.d_model, config.n_heads, config.d_ff, config.dropout
            )
            for _ in range(config.n_local_latent_layers)
        )
        self.content_cross_segment = nn.ModuleList(
            TransformerLayer(
                config.d_model, config.n_heads, config.d_ff, config.dropout
            )
            for _ in range(config.n_cross_segment_layers)
        )
        self.state_cross_segment = nn.ModuleList(
            TransformerLayer(
                config.d_model, config.n_heads, config.d_ff, config.dropout
            )
            for _ in range(config.n_cross_segment_layers)
        )
        self.segment_embedding = nn.Embedding(config.max_segments, config.d_model)
        return

    def _local(
        self,
        latents: torch.Tensor,
        segment_valid: torch.Tensor,
        layers: nn.ModuleList,
    ) -> torch.Tensor:
        batch, n_segments, n_latents, d_model = latents.shape
        flat = latents.reshape(batch * n_segments, n_latents, d_model)
        key_mask = segment_valid.reshape(batch * n_segments, 1).expand(-1, n_latents)
        for layer in layers:
            flat = layer(flat, key_padding_mask=key_mask)
        return flat.view(batch, n_segments, n_latents, d_model)

    def _cross_segment(
        self,
        latents: torch.Tensor,
        segment_valid: torch.Tensor,
        layers: nn.ModuleList,
    ) -> torch.Tensor:
        batch, n_segments, n_latents, d_model = latents.shape
        positions = torch.arange(n_segments, device=latents.device)
        latents = latents + self.segment_embedding(positions)[None, :, None, :]
        flat = latents.reshape(batch, n_segments * n_latents, d_model)
        key_mask = (
            segment_valid[:, :, None]
            .expand(-1, -1, n_latents)
            .reshape(batch, n_segments * n_latents)
        )
        for layer in layers:
            flat = layer(flat, key_padding_mask=key_mask)
        return flat.view(batch, n_segments, n_latents, d_model)

    def forward(
        self,
        encoded: torch.Tensor,
        token_valid: torch.Tensor,
        segment_valid: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return contextual content and state latents."""
        batch, n_segments, _, d_model = encoded.shape
        content_tokens = encoded[:, :, :N_SLOTS, :]
        content_valid = token_valid[:, :, :N_SLOTS]
        content_queries = self.content_queries[None, None, :, :].expand(
            batch, n_segments, -1, -1
        )
        state_queries = self.state_queries[None, None, :, :].expand(
            batch, n_segments, -1, -1
        )

        content = self.content_cross(
            content_queries, content_tokens, segment_valid, content_valid
        )
        state = self.state_cross(state_queries, encoded, segment_valid, token_valid)
        content = self._local(content, segment_valid, self.content_local)
        state = self._local(state, segment_valid, self.state_local)
        content = self._cross_segment(
            content, segment_valid, self.content_cross_segment
        )
        state = self._cross_segment(state, segment_valid, self.state_cross_segment)
        return content, state


class LatentReadout(nn.Module):
    """Learned attention-pool readout over a segment's latents."""

    def __init__(self, d_model: int, n_heads: int, dropout: float) -> None:
        super().__init__()
        self.query = _init_latents(1, d_model)
        self.cross = SegmentCrossAttention(d_model, n_heads, dropout)
        return

    def forward(
        self, latents: torch.Tensor, segment_valid: torch.Tensor
    ) -> torch.Tensor:
        batch, n_segments, _, d_model = latents.shape
        query = self.query[None, None, :, :].expand(batch, n_segments, 1, d_model)
        pooled = self.cross(query, latents, segment_valid)
        return pooled[:, :, 0, :] * segment_valid.unsqueeze(-1)
