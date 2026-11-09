from __future__ import annotations

from dataclasses import dataclass

from src.modeling.encoder.config import EncoderConfig


@dataclass
class PerceiverIOConfig(EncoderConfig):
    """Perceiver IO architecture with separate content and state bottlenecks."""

    arch: str = "perceiver_io"
    n_content_latents: int = 16
    n_state_latents: int = 1
    n_local_latent_layers: int = 2
    n_cross_segment_layers: int = 2
    n_decoder_layers: int = 2
    content_readout: str = "attention"
    decoder_reconstruction_weight: float = 0.0
    content_pooling: bool = True
    aux_heads: bool = False
    phase_classes: int = 3

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.arch != "perceiver_io":
            raise ValueError("arch must be 'perceiver_io'")
        if self.n_content_latents <= 0:
            raise ValueError("n_content_latents must be positive")
        if self.n_state_latents <= 0:
            raise ValueError("n_state_latents must be positive")
        for name in (
            "n_local_latent_layers",
            "n_cross_segment_layers",
            "n_decoder_layers",
        ):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} must be non-negative")
        if self.content_readout not in ("attention", "first_latent"):
            raise ValueError("content_readout must be 'attention' or 'first_latent'")
        if self.decoder_reconstruction_weight < 0:
            raise ValueError("decoder_reconstruction_weight must be non-negative")
        if self.phase_classes <= 0:
            raise ValueError("phase_classes must be positive")
        return
