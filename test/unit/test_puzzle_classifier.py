from __future__ import annotations

import chess
import numpy as np
import pytest
import torch
import torch.nn.functional as torch_functional

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


class TestTwoLayerHead:
    def test_hidden_layer_shapes_follow_the_config(self):
        with given("a two-layer head with d_model=4 and hidden_dim=3"):
            head = MeanPoolingClassifier(
                MeanPoolingClassifierConfig(d_model=4, hidden_dim=3)
            )
            embeddings = torch.randn(2, 5, 4)
            mask = torch.ones(2, 5, dtype=torch.bool)

        with when("the head runs on matching embeddings"):
            logits = head(embeddings, mask)

        with then(
            "the projection feeds d_model in and the classifier reads hidden_dim"
        ):
            assert head.projection.in_features == 4
            assert head.projection.out_features == 3
            assert head.classifier.in_features == 3
            assert logits.shape == (2,)
        return

    def test_forward_applies_the_hidden_layer_before_the_classifier(self):
        with given("a two-layer head with an explicit hidden activation"):
            torch.manual_seed(0)
            head = MeanPoolingClassifier(
                MeanPoolingClassifierConfig(d_model=2, hidden_dim=3)
            )
            embeddings = torch.tensor([[[0.5, -1.0], [2.0, 0.25]]])
            mask = torch.ones(1, 2, dtype=torch.bool)

        with when("the head pools and classifies the sequence"):
            logits = head(embeddings, mask)

        with then("the output is classifier(gelu(projection(mean)))"):
            pooled = embeddings.mean(dim=1)
            hidden = torch_functional.gelu(head.projection(pooled))
            expected = head.classifier(hidden).squeeze(-1)
            torch.testing.assert_close(logits, expected)
        return

    def test_masking_still_applies_with_a_hidden_layer(self):
        with given("a two-layer head and padded positions"):
            head = MeanPoolingClassifier(
                MeanPoolingClassifierConfig(d_model=2, hidden_dim=4)
            )
            embeddings = torch.tensor([[[1.0, 1.0], [2.0, 2.0], [99.0, 99.0]]])
            mask = torch.tensor([[True, True, False]], dtype=torch.bool)

        with when("the padded position changes"):
            logits = head(embeddings, mask)
            changed = embeddings.clone()
            changed[0, 2] = -999.0

        with then("the logits are unchanged"):
            torch.testing.assert_close(head(changed, mask), logits)
        return

    def test_single_layer_head_remains_available(self):
        with given("a head without a hidden dimension"):
            head = MeanPoolingClassifier(MeanPoolingClassifierConfig(d_model=8))

        with when("the head is inspected"):
            has_projection = head.projection is not None
            in_features = head.classifier.in_features

        with then("the projection is skipped and d_model feeds the classifier"):
            assert has_projection is False
            assert in_features == 8
        return

    @pytest.mark.parametrize("hidden_dim", [0, -2])
    def test_rejects_non_positive_hidden_dim(self, hidden_dim):
        with given(f"a hidden dimension of {hidden_dim}"):
            error = None

        with when("the configuration is validated"):
            try:
                MeanPoolingClassifierConfig(d_model=4, hidden_dim=hidden_dim)
            except ValueError as exc:
                error = exc

        with then("the invalid hidden dimension is rejected"):
            assert error is not None
        return

    def test_hidden_layer_receives_gradients(self):
        with given("a frozen encoder with a two-layer head"):
            encoder = tiny_encoder()
            head = MeanPoolingClassifier(
                MeanPoolingClassifierConfig(d_model=1, hidden_dim=4)
            )
            model = FrozenEncoderClassifier(encoder, head)
            tokens, segment_ids, padding_mask = batch_from_boards()

        with when("a training step is run"):
            model.train()
            logits = model(tokens, segment_ids, padding_mask)
            logits.sum().backward()

        with then("both head layers have gradients"):
            assert model.head.projection.weight.grad is not None
            assert model.head.classifier.weight.grad is not None
        return
