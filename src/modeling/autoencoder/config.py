from __future__ import annotations

from dataclasses import dataclass

from src.modeling.encoder.config import EncoderConfig


@dataclass
class AttentionAutoencoderConfig(EncoderConfig):
    """Transformer autoencoder with a single attention-pooled window bottleneck.

    The defaults are the six-million-parameter recipe: a 256-wide encoder and
    decoder with four and three blocks respectively, compressing each packed
    window into one 512-dimensional vector.
    """

    arch: str = "attention_autoencoder"
    d_model: int = 256
    n_heads: int = 8
    n_layers: int = 4
    d_ff: int = 1024
    latent_dim: int = 512
    n_decoder_layers: int = 3

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.arch != "attention_autoencoder":
            raise ValueError("arch must be 'attention_autoencoder'")
        if self.latent_dim <= 0:
            raise ValueError("latent_dim must be positive")
        if self.n_decoder_layers < 1:
            raise ValueError("n_decoder_layers must be at least 1")
        return
