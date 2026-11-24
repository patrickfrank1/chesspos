from __future__ import annotations

from dataclasses import dataclass

from src.modeling.encoder.config import EncoderConfig


@dataclass
class AttentionAutoencoderConfig(EncoderConfig):
    """Transformer autoencoder with a single attention-pooled window bottleneck.

    The defaults are the deep, narrow-bottleneck recipe: a 512-wide encoder
    with six blocks and sixteen heads compresses each packed window into one
    512-dimensional vector, decoded by three blocks.
    """

    arch: str = "attention_autoencoder"
    d_model: int = 512
    n_heads: int = 16
    n_layers: int = 6
    d_ff: int = 1536
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
