"""Isolated per-segment token encoder."""

from __future__ import annotations

import torch
from torch import nn

from src.modeling.encoder.config import EncoderConfig
from src.modeling.shared.embeddings import SegmentEmbedding
from src.modeling.shared.layers import TransformerLayer
from src.modeling.shared.masks import encoder_isolation_mask


class SegmentTokenEncoder(nn.Module):
    """Encode each segment independently with content/state isolation.

    Content queries attend only content slots in their own segment, while
    state queries may attend every slot in the segment. Attention never crosses
    segment boundaries because segments are processed as independent rows.
    """

    def __init__(self, config: EncoderConfig) -> None:
        super().__init__()
        self.config = config
        self.embedding = SegmentEmbedding(
            config.vocab_size,
            config.d_model,
            config.max_segments,
            config.segment_len,
        )
        self.blocks = nn.ModuleList(
            TransformerLayer(
                config.d_model, config.n_heads, config.d_ff, config.dropout
            )
            for _ in range(config.n_layers)
        )
        self.ln_final = nn.LayerNorm(config.d_model)
        self.register_buffer(
            "isolation_mask",
            encoder_isolation_mask(config.segment_len),
            persistent=False,
        )
        return

    @property
    def slot_embedding(self) -> nn.Embedding:
        return self.embedding.slot_embedding

    def forward(
        self,
        token_table: torch.Tensor,
        token_valid: torch.Tensor,
        segment_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Encode ``[batch, segments, segment_len]`` board tokens."""
        batch, n_segments, segment_len = token_table.shape
        x = self.embedding(token_table, segment_ids)
        x = x * token_valid.unsqueeze(-1)
        flat = x.reshape(batch * n_segments, segment_len, -1)
        key_mask = token_valid.reshape(batch * n_segments, segment_len)
        for block in self.blocks:
            flat = block(
                flat,
                key_padding_mask=key_mask,
                attn_mask=self.isolation_mask,
            )
        flat = self.ln_final(flat)
        return flat.view(batch, n_segments, segment_len, -1)
