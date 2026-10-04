import chess
import numpy as np
import pytest

from src.dataset.position_encoder import (
    PositionEncoder,
    get_encoder,
    register_encoder,
)


class TestGetEncoder:
    def test_get_unknown_encoder_raises_error(self):
        with pytest.raises(ValueError, match="Unknown encoding format"):
            get_encoder("unknown_format")


class TestRegisterEncoder:
    def test_register_custom_encoder(self):
        class CustomEncoder(PositionEncoder):
            @property
            def encoding_format(self):
                return "custom"

            @property
            def output_shape(self):
                return (10,)

            def encode(self, board):
                return np.zeros(10, dtype=np.int8)

            def encode_batch(self, boards):
                return np.zeros((len(boards), 10), dtype=np.int8)

            def decode(self, data):
                return chess.Board()

            def decode_batch(self, data):
                return [chess.Board() for _ in data]

        register_encoder("custom", CustomEncoder)
        encoder = get_encoder("custom")
        assert isinstance(encoder, CustomEncoder)
