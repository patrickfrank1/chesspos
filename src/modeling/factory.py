"""Model factory with backward-compatible architecture dispatch."""

from __future__ import annotations

from dataclasses import asdict, fields, is_dataclass
from typing import Any

from torch import nn

from src.modeling.perceiver.config import PerceiverIOConfig
from src.modeling.perceiver.model import PerceiverIOModel
from src.modeling.transformer import MaskedStreamTransformer, TransformerConfig

ARCH_TRANSFORMER = "transformer"
ARCH_PERCEIVER_IO = "perceiver_io"


def model_arch(config: dict[str, Any] | Any) -> str:
    """Return the architecture name encoded in a config or checkpoint dict."""
    if is_dataclass(config):
        return getattr(config, "arch", ARCH_TRANSFORMER)
    return config.get("arch", ARCH_TRANSFORMER)


def _known_fields(config_type: type) -> set[str]:
    return {field.name for field in fields(config_type)}


def model_config_from_dict(
    values: dict[str, Any],
) -> TransformerConfig | PerceiverIOConfig:
    """Build the appropriate config from a checkpoint ``model_config`` dict."""
    if values.get("arch", ARCH_TRANSFORMER) == ARCH_PERCEIVER_IO:
        known = _known_fields(PerceiverIOConfig)
        return PerceiverIOConfig(
            **{key: value for key, value in values.items() if key in known}
        )
    known = _known_fields(TransformerConfig)
    return TransformerConfig(
        **{key: value for key, value in values.items() if key in known}
    )


def build_model(
    config: TransformerConfig | PerceiverIOConfig | dict[str, Any],
) -> nn.Module:
    """Build a model from a config object or a config dictionary."""
    if isinstance(config, dict):
        config = model_config_from_dict(config)
    if isinstance(config, PerceiverIOConfig):
        return PerceiverIOModel(config)
    if isinstance(config, TransformerConfig):
        return MaskedStreamTransformer(config)
    raise TypeError(f"Unsupported model config type: {type(config)!r}")


def config_asdict(config: TransformerConfig | PerceiverIOConfig) -> dict[str, Any]:
    """Return a checkpoint-safe config mapping."""
    return asdict(config)
