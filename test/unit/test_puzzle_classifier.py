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
from src.modeling.autoencoder.config import AttentionAutoencoderConfig
from src.modeling.autoencoder.model import AttentionAutoencoderModel
from src.modeling.frozen import embedding_dim, encoder_embeddings
from src.modeling.puzzle_classifier import (
    FrozenEncoderClassifier,
    PoolingClassifier,
    PoolingClassifierConfig,
)
from src.modeling.transformer import MaskedStreamTransformer, TransformerConfig


def tiny_encoder(
    d_model: int = 1,
    n_layers: int = 1,
    d_ff: int = 2,
) -> MaskedStreamTransformer:
    return MaskedStreamTransformer(
        TransformerConfig(
            vocab_size=VOCAB_SIZE,
            d_model=d_model,
            n_heads=1,
            n_layers=n_layers,
            d_ff=d_ff,
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


class TestMeanPooling:
    def test_output_shape_and_masking(self):
        with given("a d_model=1 classifier and a batch with padded positions"):
            head = PoolingClassifier(PoolingClassifierConfig(d_model=1))
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
            head = PoolingClassifier(PoolingClassifierConfig(d_model=d_model))
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
            head = PoolingClassifier(PoolingClassifierConfig(d_model=1))
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
            head = PoolingClassifier(PoolingClassifierConfig(d_model=1))
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
            head = PoolingClassifier(PoolingClassifierConfig(d_model=1))
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

    def test_full_feature_uses_the_non_isolated_trunk(self):
        with given("a frozen classifier using the full-trunk feature recipe"):
            encoder = tiny_encoder()
            head = PoolingClassifier(PoolingClassifierConfig(d_model=1))
            model = FrozenEncoderClassifier(encoder, head, feature="full")
            tokens, segment_ids, padding_mask = batch_from_boards()
            calls = []
            original_embed = encoder.embed
            original_encode = encoder.encode

            def embed(*args, **kwargs):
                calls.append("embed")
                return original_embed(*args, **kwargs)

            def encode(*args, **kwargs):
                calls.append("encode")
                return original_encode(*args, **kwargs)

            encoder.embed = embed
            encoder.encode = encode

        with when("the classifier scores a batch"):
            model(tokens, segment_ids, padding_mask)

        with then("encode is used instead of embed"):
            assert calls == ["encode"]
        return

    def test_rejects_an_unknown_feature_recipe(self):
        with given("an unsupported feature name"):
            encoder = tiny_encoder()
            head = PoolingClassifier(PoolingClassifierConfig(d_model=1))

        with when("the frozen classifier is constructed"):
            error = None
            try:
                FrozenEncoderClassifier(encoder, head, feature="content_state")
            except ValueError as exc:
                error = exc

        with then("a ValueError is raised"):
            assert isinstance(error, ValueError)
        return


class TestFineTuning:
    def test_unfreezing_the_last_block_requires_the_full_feature(self):
        with given("an unfrozen last block and the native feature recipe"):
            encoder = tiny_encoder()
            head = PoolingClassifier(PoolingClassifierConfig(d_model=1))

        with when("the classifier is constructed"):
            error = None
            try:
                FrozenEncoderClassifier(
                    encoder, head, feature="native", unfreeze_last_block=True
                )
            except ValueError as exc:
                error = exc

        with then("a ValueError explains the incompatible combination"):
            assert isinstance(error, ValueError)
        return

    def test_full_fine_tuning_requires_the_full_feature(self):
        with given("every encoder parameter unfrozen with the native recipe"):
            encoder = tiny_encoder()
            head = PoolingClassifier(PoolingClassifierConfig(d_model=1))

        with when("the classifier is constructed"):
            error = None
            try:
                FrozenEncoderClassifier(
                    encoder, head, feature="native", unfreeze_all=True
                )
            except ValueError as exc:
                error = exc

        with then("a ValueError is raised"):
            assert isinstance(error, ValueError)
        return

    def test_unfreezing_trains_only_the_last_block(self):
        with given("a two-layer encoder with the last block unfrozen"):
            encoder = tiny_encoder(n_layers=2)
            head = PoolingClassifier(PoolingClassifierConfig(d_model=1))
            model = FrozenEncoderClassifier(
                encoder, head, feature="full", unfreeze_last_block=True
            )

        with when("the trainable parameters are inspected"):
            trainable = {
                name
                for name, parameter in model.named_parameters()
                if parameter.requires_grad
            }

        with then("only the last block, final norm, and head remain trainable"):
            assert any(name.startswith("encoder.blocks.1.") for name in trainable)
            assert "encoder.ln_final.weight" in trainable
            assert not any(name.startswith("encoder.blocks.0.") for name in trainable)
            assert any(name.startswith("head.") for name in trainable)
            assert model.encoder.training is True
        return

    def test_all_encoder_parameters_are_trainable(self):
        with given("a fully unfrozen encoder with the full feature recipe"):
            encoder = tiny_encoder(d_model=4, n_layers=2, d_ff=8)
            head = PoolingClassifier(PoolingClassifierConfig(d_model=4))
            model = FrozenEncoderClassifier(
                encoder, head, feature="full", unfreeze_all=True
            )
            tokens, segment_ids, padding_mask = batch_from_boards()

        with when("a training step is run"):
            model.train()
            logits = model(tokens, segment_ids, padding_mask)
            logits.sum().backward()

        with then("the first block and embeddings receive gradients"):
            assert all(p.requires_grad for p in model.encoder.parameters())
            assert model.encoder.training is True
            assert encoder.blocks[0].qkv.weight.grad is not None
            assert encoder.blocks[0].qkv.weight.grad.abs().sum() > 0
        return

    def test_eval_mode_restores_the_frozen_behaviour_of_the_encoder(self):
        with given("a fully unfrozen classifier"):
            encoder = tiny_encoder(d_model=4, n_layers=2, d_ff=8)
            head = PoolingClassifier(PoolingClassifierConfig(d_model=4))
            model = FrozenEncoderClassifier(
                encoder, head, feature="full", unfreeze_all=True
            )

        with when("the model is switched to train and then eval"):
            model.train()
            train_mode = model.encoder.training
            model.eval()
            eval_mode = model.encoder.training

        with then("the encoder follows the requested mode"):
            assert train_mode is True
            assert eval_mode is False
        return

    def test_cannot_be_combined_with_last_block_unfreezing(self):
        with given("both unfreeze options enabled"):
            encoder = tiny_encoder()
            head = PoolingClassifier(PoolingClassifierConfig(d_model=1))

        with when("the classifier is constructed"):
            error = None
            try:
                FrozenEncoderClassifier(
                    encoder,
                    head,
                    feature="full",
                    unfreeze_last_block=True,
                    unfreeze_all=True,
                )
            except ValueError as exc:
                error = exc

        with then("a ValueError is raised"):
            assert isinstance(error, ValueError)
        return


class TestTwoLayerHead:
    def test_hidden_layer_shapes_follow_the_config(self):
        with given("a two-layer head with d_model=4 and hidden_dim=3"):
            head = PoolingClassifier(PoolingClassifierConfig(d_model=4, hidden_dim=3))
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
            head = PoolingClassifier(PoolingClassifierConfig(d_model=2, hidden_dim=3))
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
            head = PoolingClassifier(PoolingClassifierConfig(d_model=2, hidden_dim=4))
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
            head = PoolingClassifier(PoolingClassifierConfig(d_model=8))

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
                PoolingClassifierConfig(d_model=4, hidden_dim=hidden_dim)
            except ValueError as exc:
                error = exc

        with then("the invalid hidden dimension is rejected"):
            assert error is not None
        return

    def test_hidden_layer_receives_gradients(self):
        with given("a frozen encoder with a two-layer head"):
            encoder = tiny_encoder()
            head = PoolingClassifier(PoolingClassifierConfig(d_model=1, hidden_dim=4))
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


class TestAttentionPooling:
    def test_attention_weights_are_a_masked_softmax(self):
        with given("an attention head and a batch with one padded position"):
            head = PoolingClassifier(
                PoolingClassifierConfig(d_model=2, pooling="attention")
            )
            embeddings = torch.tensor([[[1.0, 0.0], [0.0, 1.0], [9.0, 9.0]]])
            mask = torch.tensor([[True, True, False]], dtype=torch.bool)

        with when("the attention weights are computed"):
            weights = head.attention_weights(embeddings, mask)

        with then("valid positions are normalized and padding gets no weight"):
            assert weights.shape == (1, 3)
            torch.testing.assert_close(weights.sum(dim=1), torch.ones(1))
            assert weights[0, 2].item() == 0.0
        return

    def test_pooled_vector_is_the_attention_weighted_sum(self):
        with given("an attention head and a masked batch"):
            head = PoolingClassifier(
                PoolingClassifierConfig(d_model=3, pooling="attention")
            )
            embeddings = torch.randn(2, 4, 3)
            mask = torch.tensor(
                [[True, True, False, False], [True, False, True, False]]
            )

        with when("the representation is pooled"):
            weights = head.attention_weights(embeddings, mask)
            pooled = head.pool(embeddings, mask)

        with then("it equals the weighted sum of the position embeddings"):
            expected = torch.bmm(weights.unsqueeze(1), embeddings).squeeze(1)
            torch.testing.assert_close(pooled, expected)
        return

    def test_padding_does_not_change_the_logits(self):
        with given("an attention head and a sequence with a padded position"):
            head = PoolingClassifier(
                PoolingClassifierConfig(d_model=2, pooling="attention")
            )
            embeddings = torch.tensor([[[1.0, 1.0], [2.0, 2.0], [99.0, 99.0]]])
            mask = torch.tensor([[True, True, False]], dtype=torch.bool)

        with when("the padded embedding changes"):
            logits = head(embeddings, mask)
            changed = embeddings.clone()
            changed[0, 2] = -999.0

        with then("the logits are unchanged"):
            torch.testing.assert_close(head(changed, mask), logits)
        return

    def test_query_is_a_learned_parameter(self):
        with given("an attention head"):
            head = PoolingClassifier(
                PoolingClassifierConfig(d_model=8, pooling="attention")
            )

        with when("the query is inspected"):
            query = head.query

        with then("it is a trainable d_model vector initialized to uniform attention"):
            assert query.shape == (8,)
            assert query.requires_grad is True
            assert torch.count_nonzero(query).item() == 0
        return

    def test_query_receives_gradients_through_the_frozen_encoder(self):
        with given("a frozen encoder with an attention head"):
            encoder = tiny_encoder()
            head = PoolingClassifier(
                PoolingClassifierConfig(d_model=1, hidden_dim=4, pooling="attention")
            )
            model = FrozenEncoderClassifier(encoder, head)
            tokens, segment_ids, padding_mask = batch_from_boards()

        with when("a training step is run"):
            model.train()
            logits = model(tokens, segment_ids, padding_mask)
            logits.sum().backward()

        with then("the query and head parameters have gradients"):
            assert model.head.query.grad is not None
            assert model.head.projection.weight.grad is not None
            assert model.head.classifier.weight.grad is not None
        return

    def test_fully_masked_rows_pool_to_zero(self):
        with given("an attention head with a fully padded row"):
            head = PoolingClassifier(
                PoolingClassifierConfig(d_model=2, pooling="attention")
            )
            embeddings = torch.randn(1, 3, 2)
            mask = torch.zeros(1, 3, dtype=torch.bool)

        with when("the representation is pooled"):
            weights = head.attention_weights(embeddings, mask)
            pooled = head.pool(embeddings, mask)

        with then("the row contributes no weight and pools to zero"):
            torch.testing.assert_close(weights, torch.zeros(1, 3))
            torch.testing.assert_close(pooled, torch.zeros(1, 2))
        return

    def test_mean_pooling_remains_the_config_default(self):
        with given("a head config without an explicit pooling mode"):
            config = PoolingClassifierConfig(d_model=4)

        with when("the head is built"):
            head = PoolingClassifier(config)

        with then("mean pooling is used and no query is allocated"):
            assert config.pooling == "mean"
            assert head.query is None
        return

    def test_rejects_unknown_pooling_modes(self):
        with given("an unsupported pooling mode"):
            error = None

        with when("the config is validated"):
            try:
                PoolingClassifierConfig(d_model=4, pooling="max")
            except ValueError as exc:
                error = exc

        with then("the mode is rejected"):
            assert error is not None
        return


def tiny_autoencoder(
    latent_dim: int = 4, d_model: int = 2
) -> AttentionAutoencoderModel:
    return AttentionAutoencoderModel(
        AttentionAutoencoderConfig(
            vocab_size=VOCAB_SIZE,
            d_model=d_model,
            n_heads=1,
            n_layers=1,
            d_ff=4,
            dropout=0.0,
            max_seq_len=256,
            max_segments=8,
            latent_dim=latent_dim,
            n_decoder_layers=1,
        )
    )


class TestAutoencoderBottleneck:
    def test_native_classifier_accepts_the_bottleneck_width(self):
        with given("an autoencoder whose bottleneck is wider than d_model"):
            encoder = tiny_autoencoder(latent_dim=4, d_model=2)
            head = PoolingClassifier(
                PoolingClassifierConfig(d_model=embedding_dim(encoder, "native"))
            )
            model = FrozenEncoderClassifier(encoder, head, feature="native")
            tokens, segment_ids, padding_mask = batch_from_boards()

        with when("the frozen classifier runs a training step"):
            model.train()
            logits = model(tokens, segment_ids, padding_mask)
            logits.sum().backward()

        with then("the bottleneck vectors are pooled by the head"):
            assert logits.shape == (2,)
            assert model.head.config.d_model == 4
            assert model.head.classifier.weight.grad is not None
            assert model.encoder.training is False
        return


class TestPerPositionRecipe:
    def test_embeds_each_position_as_its_own_window(self):
        with given("a two-position packed sequence and an autoencoder"):
            encoder = tiny_autoencoder(latent_dim=4, d_model=2)
            tokens, segment_ids, padding_mask = batch_from_boards()
            packed = pack_stream(
                [
                    TokenStreamEncoder().encode(chess.Board()),
                    TokenStreamEncoder().encode(chess.Board()),
                ]
            )
            tokens = torch.from_numpy(packed.astype(np.int64)).unsqueeze(0)
            segment_ids = torch.cumsum(tokens == SEP, dim=1)
            padding_mask = tokens != PAD
            single = [
                torch.from_numpy(
                    pack_stream([TokenStreamEncoder().encode(chess.Board())]).astype(
                        np.int64
                    )
                ).unsqueeze(0)
                for _ in range(2)
            ]

        with when("the per-position recipe runs"):
            embeddings, mask = encoder_embeddings(
                encoder, tokens, segment_ids, padding_mask, feature="native_positions"
            )

        with then("each position matches its own packed window"):
            assert embeddings.shape == (1, 2, 4)
            assert mask.tolist() == [[True, True]]
            with torch.no_grad():
                for index, window in enumerate(single):
                    ids = torch.cumsum(window == SEP, dim=1)
                    expected, _ = encoder.embed(window, ids)
                    torch.testing.assert_close(
                        embeddings[0, index], expected[0, 0], rtol=0, atol=1e-5
                    )
        return

    def test_classifier_pools_all_positions_of_a_puzzle(self):
        with given("a classifier using the per-position autoencoder recipe"):
            encoder = tiny_autoencoder(latent_dim=4, d_model=2)
            head = PoolingClassifier(
                PoolingClassifierConfig(d_model=embedding_dim(encoder, "native"))
            )
            model = FrozenEncoderClassifier(encoder, head, feature="native_positions")
            tokens, segment_ids, padding_mask = batch_from_boards()

        with when("the classifier runs a training step"):
            model.train()
            logits = model(tokens, segment_ids, padding_mask)
            logits.sum().backward()

        with then("the head trains on the pooled position embeddings"):
            assert logits.shape == (2,)
            assert model.head.classifier.weight.grad is not None
            assert model.encoder.training is False
        return


class TestAutoencoderFineTuning:
    def test_native_position_fine_tuning_trains_the_autoencoder(self):
        with given("an autoencoder classifier with every encoder parameter unfrozen"):
            encoder = tiny_autoencoder(latent_dim=4, d_model=2)
            head = PoolingClassifier(
                PoolingClassifierConfig(d_model=embedding_dim(encoder, "native"))
            )
            model = FrozenEncoderClassifier(
                encoder,
                head,
                feature="native_positions",
                unfreeze_all=True,
            )
            tokens, segment_ids, padding_mask = batch_from_boards()

        with when("a training step runs"):
            model.train()
            logits = model(tokens, segment_ids, padding_mask)
            logits.sum().backward()

        with then("gradients reach the per-position bottleneck path"):
            assert logits.shape == (2,)
            assert model.encoder.training is True
            assert any(
                parameter.grad is not None for parameter in model.encoder.parameters()
            )
        return
