from __future__ import annotations

import copy

import chess
import numpy as np
import pytest
import torch

from bdd import given, then, when
from src.dataset.token_stream import (
    EMPTY,
    PAD,
    SEP,
    TURN_BLACK,
    TURN_WHITE,
    VOCAB_SIZE,
    TokenStreamEncoder,
    pack_stream,
    resolve_max_segments,
)
from src.modeling.factory import build_model, model_config_from_dict
from src.modeling.masking import MaskingConfig, mask_stream
from src.modeling.perceiver import PerceiverIOModel, PerceiverIOConfig
from src.modeling.transformer import MaskedStreamTransformer, TransformerConfig
from src.training.objectives import AuxiliaryLossConfig, compute_training_loss


def tiny_perceiver_config(**overrides) -> PerceiverIOConfig:
    values = {
        "vocab_size": VOCAB_SIZE,
        "d_model": 16,
        "n_heads": 2,
        "n_layers": 1,
        "d_ff": 32,
        "dropout": 0.0,
        "max_seq_len": 512,
        "max_segments": 8,
        "n_content_latents": 4,
        "n_state_latents": 1,
        "n_local_latent_layers": 1,
        "n_cross_segment_layers": 1,
        "n_decoder_layers": 1,
    }
    values.update(overrides)
    return PerceiverIOConfig(**values)


def game_segments(count: int = 3) -> list[np.ndarray]:
    encoder = TokenStreamEncoder()
    board = chess.Board()
    segments = []
    for _ in range(count):
        segments.append(encoder.encode(board))
        board.push(list(board.legal_moves)[0])
    return segments


def pack(segments: list[np.ndarray]) -> tuple[torch.Tensor, ...]:
    tokens_np = pack_stream(segments)
    tokens = torch.from_numpy(tokens_np.astype(np.int64))[None]
    segment_ids = torch.from_numpy(np.cumsum(tokens_np == SEP).astype(np.int64))[None]
    padding = torch.ones_like(tokens, dtype=torch.bool)
    return tokens, segment_ids, padding


def flip_turn(segments: list[np.ndarray]) -> list[np.ndarray]:
    flipped = [segment.copy() for segment in segments]
    for segment in flipped:
        segment[64] = TURN_BLACK if segment[64] == TURN_WHITE else TURN_WHITE
    return flipped


def masked_batch(segments: list[np.ndarray]) -> dict[str, torch.Tensor]:
    tokens_np = pack_stream(segments)
    masking = MaskingConfig(
        p_rect=1.0, p_random=0.0, p_board=0.0, p_span=0.0, p_skip_gram=0.0
    )
    masked, targets, weights = mask_stream(tokens_np, masking, np.random.default_rng(0))
    return {
        "tokens": torch.from_numpy(masked.astype(np.int64))[None],
        "segment_ids": torch.from_numpy(np.cumsum(masked == SEP).astype(np.int64))[
            None
        ],
        "padding_mask": torch.from_numpy(masked != PAD)[None],
        "targets": torch.from_numpy(targets.astype(np.int64))[None],
        "weights": torch.from_numpy(weights)[None],
    }


def content_mask(tokens: torch.Tensor) -> torch.Tensor:
    return (tokens >= EMPTY) & (tokens < TURN_WHITE)


def state_mask(tokens: torch.Tensor) -> torch.Tensor:
    return (tokens >= TURN_WHITE) & (tokens < VOCAB_SIZE)


class TestPerceiverViews:
    def test_content_embedding_ignores_state_tokens(self):
        with given("a Perceiver and a multi-position window"):
            model = PerceiverIOModel(tiny_perceiver_config()).eval()
            segments = game_segments()
            tokens, segment_ids, padding = pack(segments)
            flipped_tokens, flipped_ids, flipped_padding = pack(flip_turn(segments))

        with when("content embeddings are computed for both state variants"):
            with torch.no_grad():
                content, _ = model.encode_content(tokens, segment_ids, padding)
                flipped, _ = model.encode_content(
                    flipped_tokens, flipped_ids, flipped_padding
                )

        with then("content embeddings are exactly invariant to state tokens"):
            assert torch.allclose(content, flipped, atol=1e-6)
        return

    def test_state_embedding_reacts_to_state_tokens(self):
        with given("a Perceiver and a multi-position window"):
            model = PerceiverIOModel(tiny_perceiver_config()).eval()
            segments = game_segments()
            tokens, segment_ids, padding = pack(segments)
            flipped_tokens, flipped_ids, flipped_padding = pack(flip_turn(segments))

        with when("state embeddings are computed for both state variants"):
            with torch.no_grad():
                state, _ = model.encode_state(tokens, segment_ids, padding)
                flipped, _ = model.encode_state(
                    flipped_tokens, flipped_ids, flipped_padding
                )

        with then("state embeddings change when the turn changes"):
            assert (state - flipped).abs().max().item() > 1e-5
        return

    def test_cross_segment_attention_changes_contextual_latents(self):
        with given("a Perceiver and one board embedded alone and in a window"):
            model = PerceiverIOModel(tiny_perceiver_config()).eval()
            segments = game_segments()
            window = pack(segments)
            alone = pack([segments[0]])

        with when("the content embedding is computed for both contexts"):
            with torch.no_grad():
                alone_content, _ = model.encode_content(*alone)
                window_content, _ = model.encode_content(*window)

        with then("the window context changes the first position's embedding"):
            difference = (alone_content[:, 0] - window_content[:, 0]).abs()
            assert difference.max().item() > 1e-5
        return

    def test_single_segment_has_no_foreign_context(self):
        with given("one segment alone and the same segment with padded siblings"):
            model = PerceiverIOModel(tiny_perceiver_config()).eval()
            segment = game_segments(1)[0]
            alone_tokens, alone_ids, alone_padding = pack([segment])
            padded_tokens = torch.zeros(
                (2, alone_tokens.shape[1]), dtype=alone_tokens.dtype
            )
            padded_tokens[0] = alone_tokens[0]
            padded_ids = torch.zeros_like(padded_tokens)
            padded_ids[0] = alone_ids[0]
            padded_mask = torch.zeros_like(padded_tokens, dtype=torch.bool)
            padded_mask[0] = alone_padding[0]

        with when("the content embedding is computed for both batches"):
            with torch.no_grad():
                alone, _ = model.encode_content(alone_tokens, alone_ids, alone_padding)
                padded, mask = model.encode_content(
                    padded_tokens, padded_ids, padded_mask
                )

        with then("the padded siblings do not affect the first segment"):
            assert torch.allclose(alone[:, 0], padded[0, 0], atol=1e-6)
            assert mask[0, 0] and not mask[0, 1:].any()
        return

    def test_forward_shapes_and_segment_mask(self):
        with given("a Perceiver and a three-segment window"):
            model = PerceiverIOModel(tiny_perceiver_config()).eval()
            segments = game_segments()
            tokens, segment_ids, padding = pack(segments)

        with when("the model runs forward and encodes views"):
            with torch.no_grad():
                output = model(tokens, segment_ids, padding)
                content, state, mask = model.encode_views(tokens, segment_ids, padding)

        with then("the hidden states and view tensors have the expected shapes"):
            assert output["hidden_states"].shape == tokens.shape + (16,)
            assert output["logits"].shape == tokens.shape + (VOCAB_SIZE,)
            assert content.shape == (1, 8, 16)
            assert state.shape == (1, 8, 16)
            assert mask.shape == (1, 8)
            assert int(mask.sum()) == 3
        return

    def test_decoder_content_logits_ignore_state_tokens(self):
        with given("a Perceiver and two windows differing only in state"):
            model = PerceiverIOModel(tiny_perceiver_config()).eval()
            segments = game_segments()
            tokens, segment_ids, padding = pack(segments)
            flipped_tokens, flipped_ids, flipped_padding = pack(flip_turn(segments))

        with when("decoded hidden states are computed"):
            with torch.no_grad():
                hidden = model(tokens, segment_ids, padding)["hidden_states"]
                flipped = model(flipped_tokens, flipped_ids, flipped_padding)[
                    "hidden_states"
                ]

        with then("content slot logits are unchanged"):
            content = content_mask(tokens)
            assert torch.allclose(hidden[content], flipped[content], atol=1e-6)
        return

    def test_decoder_state_logits_react_to_state_tokens(self):
        with given("a Perceiver and two windows differing only in state"):
            model = PerceiverIOModel(tiny_perceiver_config()).eval()
            segments = game_segments()
            tokens, segment_ids, padding = pack(segments)
            flipped_tokens, flipped_ids, flipped_padding = pack(flip_turn(segments))

        with when("decoded hidden states are computed"):
            with torch.no_grad():
                hidden = model(tokens, segment_ids, padding)["hidden_states"]
                flipped = model(flipped_tokens, flipped_ids, flipped_padding)[
                    "hidden_states"
                ]

        with then("state slot logits change with the state tokens"):
            state = state_mask(tokens)
            assert (hidden[state] - flipped[state]).abs().max().item() > 1e-5
        return


class TestPerceiverLoss:
    def test_reconstruction_loss_is_disabled_by_default(self):
        with given("a Perceiver with the default reconstruction weight"):
            model = PerceiverIOModel(tiny_perceiver_config()).eval()
            tokens, segment_ids, padding = pack(game_segments())
            targets = torch.full_like(tokens, -100)

        with when("the forward loss is computed"):
            with torch.no_grad():
                output = model(tokens, segment_ids, padding, targets=targets)

        with then("there is no reconstruction term"):
            assert "reconstruction_loss" not in output
            assert output["loss"].item() == pytest.approx(0.0)
        return

    def test_reconstruction_loss_is_positive_when_enabled(self):
        with given("a Perceiver with reconstruction enabled and visible targets"):
            model = PerceiverIOModel(
                tiny_perceiver_config(decoder_reconstruction_weight=0.5)
            ).eval()
            tokens, segment_ids, padding = pack(game_segments())
            targets = torch.full_like(tokens, -100)

        with when("the forward loss is computed"):
            with torch.no_grad():
                output = model(tokens, segment_ids, padding, targets=targets)

        with then("the reconstruction term is finite and positive"):
            assert "reconstruction_loss" in output
            assert output["reconstruction_loss"].item() > 0
            assert torch.isfinite(output["loss"])
        return


class TestPerceiverIntegration:
    def test_max_segments_are_derived_from_seq_len(self):
        with given("a 2048-token sequence budget"):
            expected = (2048 - 1) // (67 + 1)

        with when("the segment capacity is resolved"):
            resolved = resolve_max_segments(2048)
            config = PerceiverIOConfig(
                d_model=16,
                n_heads=2,
                n_layers=1,
                d_ff=32,
                max_seq_len=2048,
            )

        with then("both use the true segment capacity and not an over-allocation"):
            assert expected == 30
            assert resolved == expected
            assert config.max_segments == expected
        return

    def test_config_validation(self):
        with given("invalid Perceiver configurations"):
            invalid = (
                {"n_content_latents": 0},
                {"n_state_latents": 0},
                {"content_readout": "mean"},
                {"decoder_reconstruction_weight": -1.0},
            )

        with when("each config is constructed"):
            errors = []
            for overrides in invalid:
                try:
                    tiny_perceiver_config(**overrides)
                except ValueError as error:
                    errors.append(error)

        with then("every invalid config is rejected"):
            assert len(errors) == len(invalid)
        return

    def test_compute_training_loss_runs_with_perceiver(self):
        with given("a masked window, a Perceiver, and auxiliary losses enabled"):
            model = PerceiverIOModel(tiny_perceiver_config())
            batch = masked_batch(game_segments(8))
            config = AuxiliaryLossConfig(
                inv_weight=0.5,
                var_weight=1.0,
                cov_weight=0.1,
                nce_weight=0.1,
                alone_var_weight=1.0,
                alone_cov_weight=0.1,
                alone_targets_per_window=2,
                warmup_steps=0,
                content_pooling=True,
            )

        with when("the shared training loss is computed and backpropagated"):
            loss, components = compute_training_loss(model, batch, config, step=10)
            loss.backward()

        with then("the loss is finite and gradients reach the model"):
            assert torch.isfinite(loss)
            assert "mlm" in components
            assert any(parameter.grad is not None for parameter in model.parameters())
        return

    def test_factory_selects_architecture(self):
        with given("transformer and Perceiver configs"):
            transformer = TransformerConfig(
                d_model=16,
                n_heads=2,
                n_layers=1,
                d_ff=32,
                max_seq_len=512,
                max_segments=8,
            )
            perceiver = tiny_perceiver_config()

        with when("the factory builds each model"):
            legacy_model = build_model(transformer)
            perceiver_model = build_model(perceiver)

        with then("the correct classes are returned"):
            assert isinstance(legacy_model, MaskedStreamTransformer)
            assert isinstance(perceiver_model, PerceiverIOModel)
        return

    def test_legacy_checkpoint_config_defaults_to_transformer(self):
        with given("a checkpoint config without an arch field"):
            values = {
                "d_model": 16,
                "n_heads": 2,
                "n_layers": 1,
                "d_ff": 32,
                "max_seq_len": 512,
                "max_segments": 8,
            }

        with when("the checkpoint config is reconstructed"):
            config = model_config_from_dict(values)

        with then("it is a legacy TransformerConfig"):
            assert isinstance(config, TransformerConfig)
        return

    def test_checkpoint_roundtrip(self):
        with given("a Perceiver and a copy with a different initialization"):
            config = tiny_perceiver_config()
            model = PerceiverIOModel(config)
            restored = PerceiverIOModel(copy.deepcopy(config))
            tokens, segment_ids, padding = pack(game_segments())

        with when("the state dict is transferred"):
            restored.load_state_dict(model.state_dict())

        with then("both models produce identical embeddings"):
            model.eval()
            restored.eval()
            with torch.no_grad():
                expected, _ = model.encode_content(tokens, segment_ids, padding)
                actual, _ = restored.encode_content(tokens, segment_ids, padding)
            assert torch.allclose(expected, actual, atol=1e-6)
        return
