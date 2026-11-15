from __future__ import annotations

import numpy as np
import torch

from bdd import given, then, when
from src.dataset.token_stream import VOCAB_SIZE, TokenStreamEncoder
from src.dataset.puzzle_variants import pack_segments
from src.modeling.score_head import (
    FrozenEncoderScore,
    ScoreHead,
    ScoreHeadConfig,
)
from src.modeling.transformer import MaskedStreamTransformer, TransformerConfig


def tiny_encoder(d_model: int = 1, d_ff: int = 2) -> MaskedStreamTransformer:
    return MaskedStreamTransformer(
        TransformerConfig(
            vocab_size=VOCAB_SIZE,
            d_model=d_model,
            n_heads=1,
            n_layers=2,
            d_ff=d_ff,
            dropout=0.0,
            max_seq_len=256,
            max_segments=8,
        )
    )


def batch_from_fen(fen: str = "") -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    import chess

    board = chess.Board(fen) if fen else chess.Board()
    board.push_san("e4")
    tokens_np, segment_ids_np = pack_segments([TokenStreamEncoder().encode(board)])
    tokens = torch.from_numpy(tokens_np.astype(np.int64)).unsqueeze(0)
    segment_ids = torch.from_numpy(segment_ids_np.astype(np.int64)).unsqueeze(0)
    return tokens, segment_ids, tokens.ne(0)


class TestScoreHead:
    def test_output_is_bounded_for_large_inputs(self):
        with given("a head receiving very large embeddings"):
            head = ScoreHead(ScoreHeadConfig(d_model=4, hidden_dim=8))
            embeddings = torch.randn(2, 5, 4) * 1e3
            mask = torch.ones(2, 5, dtype=torch.bool)

        with when("the forward pass runs"):
            scores = head(embeddings, mask)

        with then("every score stays inside [-1, 1]"):
            assert scores.shape == (2,)
            assert torch.all(scores >= -1.0)
            assert torch.all(scores <= 1.0)
        return

    def test_default_head_starts_near_zero(self):
        with given("a freshly initialised head"):
            head = ScoreHead(ScoreHeadConfig(d_model=8, hidden_dim=16))
            embeddings = torch.randn(4, 1, 8)

        with when("untrained scores are computed"):
            scores = head(embeddings)

        with then("predictions start close to a draw"):
            assert scores.abs().max() < 0.2
        return

    def test_padding_does_not_change_the_mean_pooled_score(self):
        with given("one valid position and a padded batch of the same position"):
            head = ScoreHead(ScoreHeadConfig(d_model=4, pooling="mean"))
            valid = torch.randn(1, 1, 4)
            padded = torch.cat([valid, torch.zeros(1, 4, 4)], dim=1)
            mask = torch.tensor([[True, False, False, False, False]])

        with when("both inputs are scored"):
            score_valid = head(valid)
            score_padded = head(padded, mask)

        with then("padding does not change the prediction"):
            torch.testing.assert_close(score_valid, score_padded)
        return

    def test_attention_pooling_masks_padding(self):
        with given("an attention head and one valid position with padding"):
            head = ScoreHead(ScoreHeadConfig(d_model=4, pooling="attention"))
            valid = torch.randn(1, 1, 4)
            padded = torch.cat([valid, torch.zeros(1, 3, 4)], dim=1)
            mask = torch.tensor([[True, False, False, False]])

        with when("both inputs are scored after a random query"):
            with torch.no_grad():
                head.query.normal_()
            score_valid = head(valid)
            score_padded = head(padded, mask)

        with then("the padded positions carry no weight"):
            torch.testing.assert_close(score_valid, score_padded)
        return

    def test_explicit_state_features_change_the_score(self):
        with given("a head that consumes three explicit state features"):
            head = ScoreHead(ScoreHeadConfig(d_model=4, hidden_dim=8, state_features=3))
            with torch.no_grad():
                head.classifier.weight.normal_(std=0.1)
            embeddings = torch.randn(1, 1, 4)
            white = torch.tensor([[1.0, 1.0, 0.0]])
            black = torch.tensor([[0.0, 1.0, 0.0]])

        with when("the same embedding is scored with different state vectors"):
            score_white = head(embeddings, features=white)
            score_black = head(embeddings, features=black)

        with then("the state vector reaches the classifier"):
            assert not torch.allclose(score_white, score_black)
        return

    def test_missing_state_features_are_rejected(self):
        with given("a head configured with state features and no features given"):
            head = ScoreHead(ScoreHeadConfig(d_model=4, state_features=3))
            embeddings = torch.randn(1, 1, 4)

        with when("the forward pass runs"):
            error = None
            try:
                head(embeddings)
            except ValueError as exc:
                error = exc

        with then("a ValueError is raised"):
            assert isinstance(error, ValueError)
        return

    def test_rejects_invalid_configuration(self):
        with given("invalid head configs"):
            cases = [
                dict(d_model=0),
                dict(d_model=4, pooling="max"),
                dict(d_model=4, dropout=1.0),
                dict(d_model=4, hidden_dim=0),
                dict(d_model=4, state_features=-1),
            ]

        with when("the configs are constructed"):
            errors = []
            for case in cases:
                try:
                    ScoreHeadConfig(**case)
                except ValueError:
                    errors.append(True)

        with then("every invalid config raises ValueError"):
            assert errors == [True] * len(cases)
        return


class TestFrozenEncoderScore:
    def test_encoder_is_frozen_and_head_trains(self):
        with given("a toy encoder wrapped for score regression"):
            encoder = tiny_encoder()
            head = ScoreHead(ScoreHeadConfig(d_model=1))
            model = FrozenEncoderScore(encoder, head)
            tokens, segment_ids, padding_mask = batch_from_fen()

        with when("a training step is run"):
            model.train()
            score = model(tokens, segment_ids, padding_mask)
            (score - 1.0).pow(2).mean().backward()

        with then("only the head receives gradients and the encoder stays in eval"):
            assert score.shape == (1,)
            assert model.encoder.training is False
            assert all(not p.requires_grad for p in model.encoder.parameters())
            assert all(p.grad is None for p in model.encoder.parameters())
            assert model.head.classifier.weight.grad is not None
        return

    def test_full_feature_uses_the_non_isolated_trunk(self):
        with given("a wrapper using the full-trunk feature recipe"):
            encoder = tiny_encoder()
            head = ScoreHead(ScoreHeadConfig(d_model=1))
            model = FrozenEncoderScore(encoder, head, feature="full")
            tokens, segment_ids, padding_mask = batch_from_fen()
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

        with when("the model scores a batch"):
            model(tokens, segment_ids, padding_mask)

        with then("encode is used instead of embed"):
            assert calls == ["encode"]
        return

    def test_unfreezing_the_last_block_requires_the_full_feature(self):
        with given("an unfrozen last block and the native feature recipe"):
            encoder = tiny_encoder()
            head = ScoreHead(ScoreHeadConfig(d_model=1))

        with when("the wrapper is constructed"):
            error = None
            try:
                FrozenEncoderScore(
                    encoder, head, feature="native", unfreeze_last_block=True
                )
            except ValueError as exc:
                error = exc

        with then("a ValueError explains the incompatible combination"):
            assert isinstance(error, ValueError)
        return

    def test_unfreezing_trains_only_the_last_block(self):
        with given("an unfrozen last block with the full feature recipe"):
            encoder = tiny_encoder()
            head = ScoreHead(ScoreHeadConfig(d_model=1))
            model = FrozenEncoderScore(
                encoder, head, feature="full", unfreeze_last_block=True
            )

        with when("the trainable parameters are inspected"):
            trainable = {
                name
                for name, parameter in model.named_parameters()
                if parameter.requires_grad
            }

        with then("only the last block, the final norm, and the head remain trainable"):
            assert any(name.startswith("encoder.blocks.1.") for name in trainable)
            assert "encoder.ln_final.weight" in trainable
            assert not any(name.startswith("encoder.blocks.0.") for name in trainable)
            assert any(name.startswith("head.") for name in trainable)
        return


class TestAttentionTokenPool:
    def test_attention_pool_trains_only_the_query_when_frozen(self):
        with given("a frozen encoder score model with attention token pooling"):
            encoder = tiny_encoder(d_model=4, d_ff=8)
            head = ScoreHead(ScoreHeadConfig(d_model=4))
            model = FrozenEncoderScore(
                encoder, head, feature="full", token_pooling="attention"
            )
            tokens, segment_ids, padding_mask = batch_from_fen()

        with when("a training step is run"):
            model.train()
            score = model(tokens, segment_ids, padding_mask)
            (score - 1.0).pow(2).mean().backward()

        with then("the query trains and the encoder stays frozen"):
            assert model.token_pool.query is not None
            assert model.token_pool.query.grad is not None
            assert all(p.grad is None for p in model.encoder.parameters())
        return

    def test_attention_pool_with_unfrozen_last_block_reaches_the_block(self):
        with given("an attention-pooled model with the last block unfrozen"):
            encoder = tiny_encoder(d_model=4, d_ff=8)
            head = ScoreHead(ScoreHeadConfig(d_model=4))
            model = FrozenEncoderScore(
                encoder,
                head,
                feature="full",
                unfreeze_last_block=True,
                token_pooling="attention",
            )
            tokens, segment_ids, padding_mask = batch_from_fen()

        with when("the loss is backpropagated"):
            score = model(tokens, segment_ids, padding_mask)
            (score - 1.0).pow(2).mean().backward()

        with then("gradients reach the last block through the attention pool"):
            assert encoder.blocks[1].qkv.weight.grad is not None
            assert encoder.blocks[1].qkv.weight.grad.abs().sum() > 0
        return

    def test_mean_token_pooling_keeps_the_native_embedding_path(self):
        with given("a model using the default mean token pooling"):
            encoder = tiny_encoder()
            head = ScoreHead(ScoreHeadConfig(d_model=1))
            model = FrozenEncoderScore(encoder, head, feature="full")
            tokens, segment_ids, padding_mask = batch_from_fen()
            calls = []
            original_encode = encoder.encode

            def encode(*args, **kwargs):
                calls.append("encode")
                return original_encode(*args, **kwargs)

            encoder.encode = encode

        with when("the model scores a batch"):
            model(tokens, segment_ids, padding_mask)

        with then("the encoder's own pooling is used"):
            assert calls == ["encode"]
            assert model.token_pool is None
        return

    def test_rejects_an_unknown_token_pooling_mode(self):
        with given("an unsupported token pooling mode"):
            encoder = tiny_encoder()
            head = ScoreHead(ScoreHeadConfig(d_model=1))

        with when("the model is constructed"):
            error = None
            try:
                FrozenEncoderScore(encoder, head, feature="full", token_pooling="max")
            except ValueError as exc:
                error = exc

        with then("a ValueError is raised"):
            assert isinstance(error, ValueError)
        return
