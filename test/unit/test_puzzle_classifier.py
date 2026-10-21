from __future__ import annotations

import chess
import numpy as np
import pytest
import torch

from bdd import given, then, when
from src.dataset.token_stream import (
    PAD,
    SEP,
    VOCAB_SIZE,
    TokenStreamEncoder,
    pack_stream,
)
from src.modeling.puzzle_classifier import (
    FrozenEncoderClassifier,
    MeanPoolingClassifier,
    MeanPoolingClassifierConfig,
)
from src.modeling.transformer import MaskedStreamTransformer, TransformerConfig


def tiny_encoder() -> MaskedStreamTransformer:
    return MaskedStreamTransformer(
        TransformerConfig(
            vocab_size=VOCAB_SIZE,
            d_model=1,
            n_heads=1,
            n_layers=1,
            d_ff=2,
            dropout=0.0,
            max_seq_len=256,
            max_segments=8,
        )
    )


def batch_from_boards() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    encoder = TokenStreamEncoder()
    boards = [chess.Board(), chess.Board()]
    boards[1].push_uci("e2e4")
    streams = [pack_stream([encoder.encode(board)]) for board in boards]
    max_len = max(len(stream) for stream in streams)

    tokens = np.full((len(streams), max_len), PAD, dtype=np.int16)
    for i, stream in enumerate(streams):
        tokens[i, : len(stream)] = stream
    segment_ids = np.cumsum(tokens == SEP, axis=1, dtype=np.int64)
    padding_mask = tokens != PAD
    return (
        torch.from_numpy(tokens.astype(np.int64)),
        torch.from_numpy(segment_ids),
        torch.from_numpy(padding_mask),
    )


class TestMeanPoolingClassifier:
    def test_output_shape_and_masking(self):
        with given("a d_model=1 classifier and a batch with padded positions"):
            head = MeanPoolingClassifier(MeanPoolingClassifierConfig(d_model=1))
            with torch.no_grad():
                head.classifier.weight.fill_(1.0)
                head.classifier.bias.zero_()
            embeddings = torch.tensor(
                [[[1.0], [2.0], [100.0]], [[3.0], [100.0], [100.0]]]
            )
            mask = torch.tensor(
                [[True, True, False], [True, False, False]], dtype=torch.bool
            )

        with when("the classifier pools the sequence"):
            logits = head(embeddings, mask)

        with then("only valid positions contribute to the mean"):
            assert logits.shape == (2,)
            torch.testing.assert_close(logits, torch.tensor([1.5, 3.0]))

            changed = embeddings.clone()
            changed[0, 2, 0] = -999.0
            changed[1, 1, 0] = -999.0
            changed[1, 2, 0] = -999.0
            torch.testing.assert_close(head(changed, mask), logits)
        return

    @pytest.mark.parametrize("d_model", [1, 4, 16])
    def test_classifier_is_parameterized_by_d_model(self, d_model):
        with given(f"a classifier configured for d_model={d_model}"):
            head = MeanPoolingClassifier(MeanPoolingClassifierConfig(d_model=d_model))
            embeddings = torch.randn(2, 3, d_model)
            mask = torch.ones(2, 3, dtype=torch.bool)

        with when("the classifier runs on matching embeddings"):
            logits = head(embeddings, mask)

        with then("the head input dimension follows the config"):
            assert logits.shape == (2,)
            assert head.classifier.in_features == d_model
        return

    def test_missing_mask_treats_every_position_as_valid(self):
        with given("a classifier and a batch without a padding mask"):
            head = MeanPoolingClassifier(MeanPoolingClassifierConfig(d_model=1))
            embeddings = torch.tensor([[[1.0], [3.0]]])

        with when("the classifier runs"):
            logits = head(embeddings)

        with then("all positions are averaged"):
            assert logits.shape == (1,)
        return


class TestFrozenEncoderClassifier:
    def test_encoder_is_frozen_and_head_trains(self):
        with given("a toy d_model=1 encoder wrapped in a frozen classifier"):
            encoder = tiny_encoder()
            head = MeanPoolingClassifier(MeanPoolingClassifierConfig(d_model=1))
            model = FrozenEncoderClassifier(encoder, head)
            tokens, segment_ids, padding_mask = batch_from_boards()

        with when("a training step is run"):
            model.train()
            logits = model(tokens, segment_ids, padding_mask)
            logits.sum().backward()

        with then("only the head has gradients and the encoder stays in eval mode"):
            assert logits.shape == (2,)
            assert model.encoder.training is False
            assert all(not p.requires_grad for p in model.encoder.parameters())
            assert all(p.grad is None for p in model.encoder.parameters())
            assert model.head.classifier.weight.grad is not None
        return

    def test_forward_uses_the_encoder_embeddings(self):
        with given("a frozen classifier and a fixed input batch"):
            encoder = tiny_encoder()
            head = MeanPoolingClassifier(MeanPoolingClassifierConfig(d_model=1))
            model = FrozenEncoderClassifier(encoder, head)
            tokens, segment_ids, padding_mask = batch_from_boards()

        with when("the same batch is passed twice in eval mode"):
            model.eval()
            with torch.no_grad():
                first = model(tokens, segment_ids, padding_mask)
                second = model(tokens, segment_ids, padding_mask)

        with then("the deterministic output is identical"):
            torch.testing.assert_close(first, second)
        return
