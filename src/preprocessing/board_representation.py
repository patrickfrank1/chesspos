from __future__ import annotations

from typing import Tuple

import numpy as np

__all__ = [
    "boolean_to_byte_vector",
    "byte_to_boolean_vector",
]


def boolean_to_byte_vector(boolean_vector: np.ndarray) -> bytes:
    assert boolean_vector.dtype in [bool, int]
    uint8_packed_vector = np.packbits(boolean_vector, axis=-1)
    binary_vector = [bytes(vector.tolist()) for vector in uint8_packed_vector]
    return binary_vector


def byte_to_boolean_vector(byte_vector: list[bytes], original_shape: Tuple[int, int]):
    unpacked_array = np.frombuffer(byte_vector, dtype=np.uint8)
    unpacked_bits = np.unpackbits(unpacked_array)
    unpacked_bits = unpacked_bits[: np.prod(original_shape)].reshape(original_shape)
    return unpacked_bits
