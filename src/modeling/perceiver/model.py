"""Segment-aware Perceiver IO foundation model."""

from __future__ import annotations


import torch
import torch.nn.functional as F
from torch import nn

from src.dataset.token_stream import EMPTY, N_SLOTS, PAD, TURN_WHITE, VOCAB_SIZE
from src.modeling.encoder.token_encoder import SegmentTokenEncoder
from src.modeling.masking import IGNORE_INDEX
from src.modeling.perceiver.config import PerceiverIOConfig
from src.modeling.perceiver.decoder import PerceiverDecoder
from src.modeling.perceiver.latent import LatentBottleneck, SlotAttentionPool
from src.modeling.shared.masks import (
    build_segment_tables,
    scatter_to_segment_table,
    segment_selection_mask,
)
from src.modeling.shared.pooling import (
    masked_loss,
    pool_content,
    pool_hidden,
    pool_state,
)
from src.training.aux_heads import AuxiliaryHeads


class PerceiverIOModel(nn.Module):
    """Perceiver IO with separate content/state latent banks.

    The model consumes the same packed token streams as the legacy transformer
    and exposes the same duck-typed interface, so training, checkpointing, and
    evaluation code can dispatch on ``config.arch``.
    """

    def __init__(self, config: PerceiverIOConfig) -> None:
        super().__init__()
        self.config = config
        self.encoder = SegmentTokenEncoder(config)
        self.bottleneck = LatentBottleneck(config)
        self.decoder = PerceiverDecoder(config, self.encoder.slot_embedding)
        self.content_readout = SlotAttentionPool(config.d_model)
        self.state_readout = SlotAttentionPool(config.d_model)
        self.head = nn.Linear(config.d_model, config.vocab_size)
        self.aux_heads = (
            AuxiliaryHeads(config.d_model, config.phase_classes)
            if config.aux_heads
            else None
        )
        return

    @property
    def model_config(self) -> PerceiverIOConfig:
        return self.config

    def _segment_capacity(self, segment_capacity: int | None) -> int:
        if segment_capacity is None:
            return self.config.max_segments
        if segment_capacity <= 0:
            raise ValueError("segment_capacity must be positive")
        return min(segment_capacity, self.config.max_segments)

    def _encode(
        self,
        tokens: torch.Tensor,
        segment_ids: torch.Tensor,
        padding_mask: torch.Tensor | None,
        segment_capacity: int | None = None,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        if padding_mask is None:
            padding_mask = tokens.ne(PAD)
        capacity = self._segment_capacity(segment_capacity)
        (
            token_table,
            token_valid,
            segment_valid,
            flat_index,
            board_valid,
        ) = build_segment_tables(tokens, segment_ids, padding_mask, capacity)
        segment_positions = torch.arange(capacity, device=tokens.device)
        segment_grid = segment_positions[None, :].expand(tokens.shape[0], -1)
        encoded = self.encoder(token_table, token_valid, segment_grid)
        content_latents, state_latents = self.bottleneck(
            encoded, token_valid, segment_valid
        )
        return (
            content_latents,
            state_latents,
            token_table,
            token_valid,
            segment_valid,
            flat_index,
            board_valid,
        )

    def _views_from_decoded(
        self,
        decoded_table: torch.Tensor,
        token_table: torch.Tensor,
        token_valid: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        content_tokens = token_table[:, :, :N_SLOTS]
        state_tokens = token_table[:, :, N_SLOTS:]
        content_mask = (
            token_valid[:, :, :N_SLOTS]
            & (content_tokens >= EMPTY)
            & (content_tokens < TURN_WHITE)
        )
        state_mask = (
            token_valid[:, :, N_SLOTS:]
            & (state_tokens >= TURN_WHITE)
            & (state_tokens < VOCAB_SIZE)
        )
        content = self.content_readout(decoded_table[:, :, :N_SLOTS], content_mask)
        state = self.state_readout(decoded_table[:, :, N_SLOTS:], state_mask)
        return content, state, content_mask.any(dim=-1)

    def _decode_table(
        self,
        content_latents: torch.Tensor,
        state_latents: torch.Tensor,
        token_valid: torch.Tensor,
        segment_valid: torch.Tensor,
    ) -> torch.Tensor:
        return self.decoder(content_latents, state_latents, token_valid, segment_valid)

    def _gather_hidden(
        self,
        decoded_table: torch.Tensor,
        flat_index: torch.Tensor,
        board_valid: torch.Tensor,
    ) -> torch.Tensor:
        flat = decoded_table.reshape(-1, decoded_table.shape[-1])
        gathered = flat[flat_index]
        return gathered * board_valid.unsqueeze(-1).to(gathered.dtype)

    def forward(
        self,
        tokens: torch.Tensor,
        segment_ids: torch.Tensor,
        padding_mask: torch.Tensor | None = None,
        targets: torch.Tensor | None = None,
        weights: torch.Tensor | None = None,
        segment_capacity: int | None = None,
    ) -> dict[str, object]:
        capacity = self._segment_capacity(segment_capacity)
        (
            content_latents,
            state_latents,
            token_table,
            token_valid,
            segment_valid,
            flat_index,
            board_valid,
        ) = self._encode(tokens, segment_ids, padding_mask, segment_capacity=capacity)
        decoded_table = self._decode_table(
            content_latents, state_latents, token_valid, segment_valid
        )
        hidden = self._gather_hidden(decoded_table, flat_index, board_valid)
        views = self._views_from_decoded(decoded_table, token_table, token_valid)
        if targets is None:
            return {
                "logits": self.head(hidden),
                "hidden_states": hidden,
                "views": views,
            }

        loss = masked_loss(hidden, targets, self.head, weights)
        output = {"loss": loss, "hidden_states": hidden, "views": views}
        if self.config.decoder_reconstruction_weight > 0:
            restored = tokens.clone()
            supervised = targets.ne(IGNORE_INDEX)
            restored[supervised] = targets[supervised]
            restored_table = scatter_to_segment_table(
                restored, flat_index, board_valid, capacity
            )
            recon_logits = self.head(decoded_table[token_valid])
            recon_targets = restored_table[token_valid]
            recon_loss = F.cross_entropy(recon_logits, recon_targets)
            loss = loss + self.config.decoder_reconstruction_weight * recon_loss
            output["loss"] = loss
            output["reconstruction_loss"] = recon_loss.detach()
        return output

    def _read_views(
        self,
        tokens: torch.Tensor,
        segment_ids: torch.Tensor,
        padding_mask: torch.Tensor | None = None,
        segment_capacity: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        (
            content_latents,
            state_latents,
            token_table,
            token_valid,
            segment_valid,
            _,
            _,
        ) = self._encode(
            tokens, segment_ids, padding_mask, segment_capacity=segment_capacity
        )
        decoded_table = self._decode_table(
            content_latents, state_latents, token_valid, segment_valid
        )
        return self._views_from_decoded(decoded_table, token_table, token_valid)

    def encode_views(
        self,
        tokens: torch.Tensor,
        segment_ids: torch.Tensor,
        padding_mask: torch.Tensor | None = None,
        *,
        segment_capacity: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self._read_views(
            tokens,
            segment_ids,
            padding_mask,
            segment_capacity=segment_capacity,
        )

    def encode_views_alone(
        self,
        tokens: torch.Tensor,
        segment_ids: torch.Tensor,
        padding_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Encode single-segment views without allocating empty segment slots."""
        return self._read_views(tokens, segment_ids, padding_mask, segment_capacity=1)

    def encode_content(
        self,
        tokens: torch.Tensor,
        segment_ids: torch.Tensor,
        padding_mask: torch.Tensor | None = None,
        *,
        segment_capacity: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        content, _, segment_mask = self._read_views(
            tokens,
            segment_ids,
            padding_mask,
            segment_capacity=segment_capacity,
        )
        return content, segment_mask

    def encode_state(
        self,
        tokens: torch.Tensor,
        segment_ids: torch.Tensor,
        padding_mask: torch.Tensor | None = None,
        *,
        segment_capacity: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        _, state, segment_mask = self._read_views(
            tokens,
            segment_ids,
            padding_mask,
            segment_capacity=segment_capacity,
        )
        return state, segment_mask

    def encode(
        self,
        tokens: torch.Tensor,
        segment_ids: torch.Tensor,
        padding_mask: torch.Tensor | None = None,
        *,
        segment_capacity: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self.encode_content(
            tokens,
            segment_ids,
            padding_mask,
            segment_capacity=segment_capacity,
        )

    @torch.no_grad()
    def embed(
        self,
        tokens: torch.Tensor,
        segment_ids: torch.Tensor,
        padding_mask: torch.Tensor | None = None,
        *,
        segment_capacity: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self.encode_content(
            tokens,
            segment_ids,
            padding_mask,
            segment_capacity=segment_capacity,
        )

    def content_segment_mask(
        self,
        tokens: torch.Tensor,
        segment_ids: torch.Tensor,
        padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if padding_mask is None:
            padding_mask = tokens.ne(PAD)
        return segment_selection_mask(
            tokens, segment_ids, padding_mask, self.config.max_segments
        )

    def pool_hidden(
        self,
        hidden: torch.Tensor,
        tokens: torch.Tensor,
        segment_ids: torch.Tensor,
        padding_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if padding_mask is None:
            padding_mask = tokens.ne(PAD)
        return pool_hidden(
            hidden, tokens, segment_ids, padding_mask, self.config.max_segments
        )

    def pool_content(
        self,
        hidden: torch.Tensor,
        tokens: torch.Tensor,
        segment_ids: torch.Tensor,
        padding_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if padding_mask is None:
            padding_mask = tokens.ne(PAD)
        return pool_content(
            hidden, tokens, segment_ids, padding_mask, self.config.max_segments
        )

    def pool_state(
        self,
        hidden: torch.Tensor,
        tokens: torch.Tensor,
        segment_ids: torch.Tensor,
        padding_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if padding_mask is None:
            padding_mask = tokens.ne(PAD)
        return pool_state(
            hidden, tokens, segment_ids, padding_mask, self.config.max_segments
        )

    def extra_repr(self) -> str:
        return (
            f"d_model={self.config.d_model}, "
            f"n_content_latents={self.config.n_content_latents}, "
            f"n_state_latents={self.config.n_state_latents}"
        )


def build_perceiver(config: PerceiverIOConfig) -> PerceiverIOModel:
    """Convenience constructor mirrored by the model factory."""
    return PerceiverIOModel(config)
