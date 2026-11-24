"""Masked attention autoencoder with a single window-level bottleneck."""

from src.modeling.autoencoder.bottleneck import AttentionBottleneck
from src.modeling.autoencoder.config import AttentionAutoencoderConfig
from src.modeling.autoencoder.decoder import LatentDecoder
from src.modeling.autoencoder.encoder import PackedSegmentEmbedding, SequenceEncoder
from src.modeling.autoencoder.model import (
    AttentionAutoencoderModel,
    build_attention_autoencoder,
    restoration_targets,
)

__all__ = [
    "AttentionAutoencoderConfig",
    "AttentionAutoencoderModel",
    "AttentionBottleneck",
    "LatentDecoder",
    "PackedSegmentEmbedding",
    "SequenceEncoder",
    "build_attention_autoencoder",
    "restoration_targets",
]
