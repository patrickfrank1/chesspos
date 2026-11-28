"""Masked attention autoencoder with a single window-level bottleneck."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from src.dataset.token_stream import PAD
from src.modeling.autoencoder.bottleneck import AttentionBottleneck
from src.modeling.autoencoder.config import AttentionAutoencoderConfig
from src.modeling.autoencoder.decoder import LatentDecoder
from src.modeling.autoencoder.encoder import SequenceEncoder
from src.modeling.masking import IGNORE_INDEX
from src.modeling.varlen import flatten_valid


def restoration_targets(tokens: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Return the unmasked gold sequence for a masked input window.

    ``mask_stream`` leaves every unmasked position in ``tokens`` untouched and
    stores the original value only at supervised positions, so combining them
    reconstructs the full gold sequence.
    """
    return torch.where(targets.ne(IGNORE_INDEX), targets, tokens)


class AttentionAutoencoderModel(nn.Module):
    """Transformer encoder/decoder with a 512-d attention-pooled bottleneck.

    One vector summarizes the whole packed window. The decoder reconstructs
    every non-padding position from that vector alone, supervised against the
    unmasked gold sequence.
    """

    def __init__(self, config: AttentionAutoencoderConfig) -> None:
        super().__init__()
        self.config = config
        self.encoder = SequenceEncoder(config)
        self.bottleneck = AttentionBottleneck(
            config.d_model, config.latent_dim, config.dropout
        )
        self.decoder = LatentDecoder(config)
        self.head = nn.Linear(config.d_model, config.vocab_size)
        return

    @property
    def model_config(self) -> AttentionAutoencoderConfig:
        return self.config

    def encode_latent(
        self,
        tokens: torch.Tensor,
        segment_ids: torch.Tensor,
        padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Encode a packed window into ``[batch, latent_dim]``."""
        if padding_mask is None:
            padding_mask = tokens.ne(PAD)
        hidden = self.encoder(tokens, segment_ids, padding_mask)
        return self.bottleneck(hidden, padding_mask)

    def decode(
        self,
        latent: torch.Tensor,
        tokens: torch.Tensor,
        segment_ids: torch.Tensor,
        padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Decode a latent back into per-token hidden states."""
        if padding_mask is None:
            padding_mask = tokens.ne(PAD)
        return self.decoder(latent, tokens, segment_ids, padding_mask)

    def _weighted_cross_entropy(
        self,
        logits: torch.Tensor,
        gold: torch.Tensor,
        weights: torch.Tensor | None,
    ) -> torch.Tensor:
        """Mean per-token cross entropy over an already-gathered token set."""
        if logits.shape[0] == 0:
            return logits.new_zeros(())
        per_token = F.cross_entropy(logits, gold, reduction="none")
        if weights is not None:
            w = weights.to(per_token.dtype)
            total = w.sum()
            if total > 0:
                return (per_token * w).sum() / total
        return per_token.mean()

    def _chunked_cross_entropy(
        self,
        hidden: torch.Tensor,
        gold: torch.Tensor,
        weights: torch.Tensor | None,
    ) -> torch.Tensor:
        """Run the vocab head one chunk of tokens at a time.

        Scoring every supervised token at once materialises ``[tokens, vocab]``
        logits on top of the gathered hidden states. With the shipped 52-token
        vocabulary that tensor is small (~55 MiB bf16 at a 524k-token batch), so
        this is a guard rather than a big win; it matters for wide vocabularies
        and keeps eval memory flat. The chunk count is fixed so the loop is a
        static unrolled graph instead of a data-dependent trip count.
        """
        if hidden.shape[0] == 0:
            return hidden.new_zeros(())
        chunks = self.config.loss_chunks
        if chunks == 1:
            return self._weighted_cross_entropy(self.head(hidden), gold, weights)
        hidden_parts = torch.tensor_split(hidden, chunks)
        gold_parts = torch.tensor_split(gold, chunks)
        weight_parts = (
            (None,) * chunks if weights is None else torch.tensor_split(weights, chunks)
        )
        total = hidden.new_zeros(())
        unweighted = hidden.new_zeros(())
        weight_total = hidden.new_zeros(())
        count = 0
        for hidden_part, gold_part, weight_part in zip(
            hidden_parts, gold_parts, weight_parts
        ):
            if hidden_part.shape[0] == 0:
                continue
            per_token = F.cross_entropy(
                self.head(hidden_part), gold_part, reduction="none"
            )
            unweighted = unweighted + per_token.sum()
            if weight_part is not None:
                w = weight_part.to(per_token.dtype)
                total = total + (per_token * w).sum()
                weight_total = weight_total + w.sum()
            count += per_token.shape[0]
        if weights is not None and weight_total > 0:
            return total / weight_total
        return unweighted / count

    def loss_from_logits(
        self,
        logits: torch.Tensor,
        tokens: torch.Tensor,
        targets: torch.Tensor,
        padding_mask: torch.Tensor | None = None,
        weights: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Weighted cross entropy over every non-padding position."""
        if padding_mask is None:
            padding_mask = tokens.ne(PAD)
        gold = restoration_targets(tokens, targets)
        valid = padding_mask.bool() & gold.ne(PAD)
        if not valid.any():
            return logits.new_zeros(())
        return self._weighted_cross_entropy(
            logits[valid],
            gold[valid],
            None if weights is None else weights[valid],
        )

    def forward(
        self,
        tokens: torch.Tensor,
        segment_ids: torch.Tensor,
        padding_mask: torch.Tensor | None = None,
        targets: torch.Tensor | None = None,
        weights: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor | int]:
        if padding_mask is None:
            padding_mask = tokens.ne(PAD)
        latent = self.encode_latent(tokens, segment_ids, padding_mask)
        hidden = self.decode(latent, tokens, segment_ids, padding_mask)
        output: dict[str, torch.Tensor | int] = {
            "latent": latent,
            "hidden_states": hidden,
        }
        if targets is None:
            output["logits"] = self.head(hidden)
            return output
        gold = restoration_targets(tokens, targets)
        supervised = padding_mask.bool() & gold.ne(PAD)
        # The vocabulary head runs only on real tokens, which keeps the
        # [tokens, vocab] tensor small; chunking keeps it bounded in memory.
        flat_hidden = flatten_valid(hidden, supervised)
        flat_gold = torch.masked_select(gold, supervised)
        flat_weights = (
            None if weights is None else torch.masked_select(weights, supervised)
        )
        output["loss"] = self._chunked_cross_entropy(
            flat_hidden, flat_gold, flat_weights
        )
        output["n_supervised"] = int(supervised.sum().item())
        return output

    def encode_content(
        self,
        tokens: torch.Tensor,
        segment_ids: torch.Tensor,
        padding_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return the window latent as a one-slot segment tensor.

        Downstream evaluation encodes one position per row, so the single
        bottleneck vector is that position's embedding.
        """
        latent = self.encode_latent(tokens, segment_ids, padding_mask)
        mask = torch.ones((tokens.shape[0], 1), dtype=torch.bool, device=tokens.device)
        return latent.unsqueeze(1), mask

    @torch.no_grad()
    def embed(
        self,
        tokens: torch.Tensor,
        segment_ids: torch.Tensor,
        padding_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self.encode_content(tokens, segment_ids, padding_mask)

    def extra_repr(self) -> str:
        return (
            f"d_model={self.config.d_model}, "
            f"n_layers={self.config.n_layers}, "
            f"n_decoder_layers={self.config.n_decoder_layers}, "
            f"latent_dim={self.config.latent_dim}"
        )


def build_attention_autoencoder(
    config: AttentionAutoencoderConfig,
) -> AttentionAutoencoderModel:
    """Convenience constructor mirrored by the model factory."""
    return AttentionAutoencoderModel(config)
