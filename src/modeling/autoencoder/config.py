from __future__ import annotations

from dataclasses import dataclass

from src.modeling.encoder.config import EncoderConfig

DCA_LOCATIONS = ("encoder", "decoder", "both")
DCA_VARIANTS = ("dca", "grn")


@dataclass
class AttentionAutoencoderConfig(EncoderConfig):
    """Transformer autoencoder with a single attention-pooled window bottleneck.

    The bare defaults are the six-million-parameter reference recipe: a
    256-wide encoder and decoder with four and three blocks, compressing each
    packed window into one 512-dimensional vector. The production recipe
    (8 enc + 4 dec, 512 wide, 256-d latent, DCA and checkpointing on) is
    :func:`production_config`.
    """

    arch: str = "attention_autoencoder"
    d_model: int = 256
    n_heads: int = 8
    n_layers: int = 4
    d_ff: int = 1024
    latent_dim: int = 512
    n_decoder_layers: int = 3
    dca: bool = False
    dca_locations: str = "encoder"
    dca_past_layers_k: int = 2
    dca_variant: str = "dca"
    grad_checkpointing: bool = False
    loss_chunks: int = 1

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.arch != "attention_autoencoder":
            raise ValueError("arch must be 'attention_autoencoder'")
        if self.latent_dim <= 0:
            raise ValueError("latent_dim must be positive")
        if self.n_decoder_layers < 1:
            raise ValueError("n_decoder_layers must be at least 1")
        if self.loss_chunks < 1:
            raise ValueError("loss_chunks must be at least 1")
        if self.dca_locations not in DCA_LOCATIONS:
            raise ValueError(f"dca_locations must be one of {DCA_LOCATIONS}")
        if self.dca_variant not in DCA_VARIANTS:
            raise ValueError(f"dca_variant must be one of {DCA_VARIANTS}")
        if self.dca_past_layers_k < 0:
            raise ValueError("dca_past_layers_k must be non-negative")
        return

    def routes(self, location: str) -> bool:
        """Whether DCA is enabled for ``"encoder"`` or ``"decoder"``."""
        if not self.dca:
            return False
        return self.dca_locations in (location, "both")

    @property
    def dca_slots(self) -> int:
        """Stack size handed to each GRN (full stack when ``k == 0``)."""
        if self.dca_past_layers_k == 0:
            return max(self.n_layers, self.n_decoder_layers) + 1
        return self.dca_past_layers_k + 2


def production_config(**overrides) -> AttentionAutoencoderConfig:
    """The 8:4 production recipe: 512 wide, 256-d latent, DCA, checkpointing.

    The bottleneck stays a hard boundary: ``dca_locations="both"`` gives the
    encoder and the decoder two disjoint stacks, never one shared stack.
    """
    values = {
        "d_model": 512,
        "n_heads": 8,
        "n_layers": 8,
        "n_decoder_layers": 4,
        "d_ff": 1024,
        "latent_dim": 256,
        "dca": True,
        "dca_locations": "both",
        "dca_past_layers_k": 2,
        "dca_variant": "dca",
        "grad_checkpointing": True,
        "loss_chunks": 8,
    }
    values.update(overrides)
    return AttentionAutoencoderConfig(**values)
