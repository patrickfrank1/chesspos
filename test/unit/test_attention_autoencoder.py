from __future__ import annotations

import copy

import chess
import numpy as np
import pytest
import torch
import torch.nn.functional as F

from bdd import given, then, when
from src.dataset.token_stream import (
    PAD,
    SEP,
    VOCAB_SIZE,
    TokenStreamEncoder,
    pack_stream,
)
from src.evaluation.core.embeddings import embed_tokens
from src.modeling.autoencoder import (
    AttentionAutoencoderConfig,
    AttentionAutoencoderModel,
    build_attention_autoencoder,
    restoration_targets,
)
from src.modeling.factory import build_model, model_config_from_dict
from src.modeling.masking import IGNORE_INDEX, MaskingConfig, mask_stream
from src.training.objectives import AuxiliaryLossConfig, compute_training_loss
from src.training.window_dataset import WindowDataset, WindowDatasetConfig


def tiny_config(**overrides) -> AttentionAutoencoderConfig:
    values = {
        "vocab_size": VOCAB_SIZE,
        "d_model": 16,
        "n_heads": 2,
        "n_layers": 1,
        "n_decoder_layers": 1,
        "d_ff": 32,
        "dropout": 0.0,
        "max_seq_len": 1024,
        "max_segments": 8,
        "latent_dim": 8,
    }
    values.update(overrides)
    return AttentionAutoencoderConfig(**values)


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


def pad_window(
    tokens: torch.Tensor, segment_ids: torch.Tensor, extra: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    padded_tokens = torch.full(
        (tokens.shape[0], tokens.shape[1] + extra), PAD, dtype=tokens.dtype
    )
    padded_tokens[:, : tokens.shape[1]] = tokens
    padded_ids = torch.zeros_like(padded_tokens)
    padded_ids[:, : tokens.shape[1]] = segment_ids
    padding = torch.zeros_like(padded_tokens, dtype=torch.bool)
    padding[:, : tokens.shape[1]] = True
    return padded_tokens, padded_ids, padding


def masked_batch(
    segments: list[np.ndarray], seed: int = 0
) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
    tokens_np = pack_stream(segments)
    masking = MaskingConfig(
        p_rect=1.0, p_random=0.0, p_board=0.0, p_span=0.0, p_skip_gram=0.0
    )
    masked, targets, weights = mask_stream(
        tokens_np, masking, np.random.default_rng(seed)
    )
    batch = {
        "tokens": torch.from_numpy(masked.astype(np.int64))[None],
        "segment_ids": torch.from_numpy(np.cumsum(masked == SEP).astype(np.int64))[
            None
        ],
        "padding_mask": torch.from_numpy(masked != PAD)[None],
        "targets": torch.from_numpy(targets.astype(np.int64))[None],
        "weights": torch.from_numpy(weights)[None],
    }
    return batch, torch.from_numpy(tokens_np.astype(np.int64))[None]


class TestBottleneck:
    def test_window_produces_one_latent_vector(self):
        with given("an autoencoder and a three-position packed window"):
            model = AttentionAutoencoderModel(tiny_config()).eval()
            tokens, segment_ids, padding = pack(game_segments())

        with when("the window is encoded"):
            with torch.no_grad():
                latent = model.encode_latent(tokens, segment_ids, padding)

        with then("the bottleneck is a single vector per window"):
            assert latent.shape == (1, 8)
            assert torch.isfinite(latent).all()
        return

    def test_padding_does_not_change_the_latent(self):
        with given("the same window with and without trailing padding"):
            model = AttentionAutoencoderModel(tiny_config()).eval()
            tokens, segment_ids, padding = pack(game_segments())
            padded = pad_window(tokens, segment_ids, extra=69)

        with when("both variants are encoded"):
            with torch.no_grad():
                plain = model.encode_latent(tokens, segment_ids, padding)
                extended = model.encode_latent(*padded)

        with then("padded positions are excluded from the bottleneck"):
            assert torch.allclose(plain, extended, atol=1e-6)
        return


class TestDecoder:
    def test_logits_cover_every_input_position(self):
        with given("an autoencoder and a packed window"):
            model = AttentionAutoencoderModel(tiny_config()).eval()
            tokens, segment_ids, padding = pack(game_segments())

        with when("the window is decoded"):
            with torch.no_grad():
                output = model(tokens, segment_ids, padding)

        with then("the decoder emits the same sequence shape plus vocabulary"):
            assert output["logits"].shape == tokens.shape + (VOCAB_SIZE,)
            assert output["hidden_states"].shape == tokens.shape + (16,)
            assert output["latent"].shape == (1, 8)
        return

    def test_decoder_uses_the_bottleneck(self):
        with given("two different windows and their latents"):
            model = AttentionAutoencoderModel(tiny_config()).eval()
            first = pack(game_segments())
            second = pack(game_segments(4))
            with torch.no_grad():
                first_latent = model.encode_latent(*first)
                second_latent = model.encode_latent(*second)

        with when("the latents are swapped before decoding the first window"):
            tokens, segment_ids, padding = first
            with torch.no_grad():
                own = model.decode(first_latent, tokens, segment_ids, padding)
                swapped = model.decode(second_latent, tokens, segment_ids, padding)

        with then("the decoded sequence changes, so the latent is not ignored"):
            assert not torch.allclose(own, swapped, atol=1e-5)
        return

    def test_decoder_is_position_aware(self):
        with given("a window and its latent"):
            model = AttentionAutoencoderModel(tiny_config()).eval()
            tokens, segment_ids, padding = pack(game_segments(2))
            with torch.no_grad():
                latent = model.encode_latent(tokens, segment_ids, padding)
                hidden = model.decode(latent, tokens, segment_ids, padding)

        with when("two positions sharing the same latent are compared"):
            first = hidden[0, 1]
            second = hidden[0, 2]

        with then("slot embeddings make the positions distinguishable"):
            assert (first - second).abs().max().item() > 1e-5
        return


class TestReconstructionLoss:
    def test_gold_sequence_is_the_unmasked_input(self):
        with given("a masked window with known masked positions"):
            tokens = torch.tensor([[3, 4, 1, 17, 2, 0]])
            targets = torch.tensor([[-100, -100, 5, -100, -100, -100]])

        with when("the gold sequence is restored"):
            restored = restoration_targets(tokens, targets)

        with then("masked positions are replaced and visible ones are kept"):
            assert restored.tolist() == [[3, 4, 5, 17, 2, 0]]
        return

    def test_loss_supervises_every_non_pad_token(self):
        with given("a masked window"):
            model = AttentionAutoencoderModel(tiny_config())
            batch, tokens_np = masked_batch(game_segments())

        with when("the reconstruction loss is computed"):
            output = model(
                batch["tokens"],
                batch["segment_ids"],
                padding_mask=batch["padding_mask"],
                targets=batch["targets"],
                weights=torch.ones_like(batch["weights"]),
            )

        with then("all non-pad positions are supervised"):
            assert output["n_supervised"] == int((tokens_np != PAD).sum())
            assert output["loss"].item() > 0
        return

    def test_padding_does_not_change_the_loss(self):
        with given("a masked window with and without trailing padding"):
            model = AttentionAutoencoderModel(tiny_config()).eval()
            batch, _ = masked_batch(game_segments())
            padded_tokens, padded_ids, padded_padding = pad_window(
                batch["tokens"], batch["segment_ids"], extra=69
            )
            padded_targets = torch.full_like(padded_tokens, IGNORE_INDEX)
            padded_targets[:, : batch["targets"].shape[1]] = batch["targets"]

        with when("both variants are scored"):
            with torch.no_grad():
                plain = model(
                    batch["tokens"],
                    batch["segment_ids"],
                    padding_mask=batch["padding_mask"],
                    targets=batch["targets"],
                    weights=torch.ones_like(batch["weights"]),
                )
                padded = model(
                    padded_tokens,
                    padded_ids,
                    padding_mask=padded_padding,
                    targets=padded_targets,
                    weights=torch.ones_like(padded_tokens, dtype=torch.float32),
                )

        with then("padded positions are ignored by the loss"):
            assert plain["loss"].item() == pytest.approx(padded["loss"].item())
        return

    def test_forward_without_targets_returns_logits_only(self):
        with given("an autoencoder and a window without targets"):
            model = AttentionAutoencoderModel(tiny_config()).eval()
            tokens, segment_ids, padding = pack(game_segments())

        with when("the window is forwarded"):
            with torch.no_grad():
                output = model(tokens, segment_ids, padding)

        with then("no loss is reported"):
            assert "loss" not in output
            assert "logits" in output
        return

    def test_gradients_flow_through_the_bottleneck(self):
        with given("a masked window and an autoencoder"):
            model = AttentionAutoencoderModel(tiny_config())
            batch, _ = masked_batch(game_segments())

        with when("the loss is backpropagated"):
            output = model(
                batch["tokens"],
                batch["segment_ids"],
                padding_mask=batch["padding_mask"],
                targets=batch["targets"],
                weights=torch.ones_like(batch["weights"]),
            )
            output["loss"].backward()

        with then("both the encoder embeddings and the bottleneck receive grads"):
            embedding = model.encoder.embedding.token_embedding
            assert embedding.weight.grad is not None
            assert model.bottleneck.query.grad is not None
        return


class TestEmbeddingInterface:
    def test_embed_returns_one_vector_per_row(self):
        with given("an autoencoder and one position per row"):
            model = AttentionAutoencoderModel(tiny_config()).eval()
            tokens, segment_ids, padding = pack(game_segments(1))

        with when("the embedding is requested"):
            with torch.no_grad():
                pooled, mask = model.embed(tokens, segment_ids, padding)

        with then("the result matches the evaluation harness shape"):
            assert pooled.shape == (1, 1, 8)
            assert mask.tolist() == [[True]]
        return

    def test_evaluation_helper_normalizes_autoencoder_embeddings(self):
        with given("an autoencoder and a batch of single positions"):
            model = AttentionAutoencoderModel(tiny_config()).eval()
            segments = game_segments(3)

        with when("the shared embedding helper runs"):
            vectors = embed_tokens(model, segments, "cpu", batch_size=2)

        with then("one unit-norm vector per position is returned"):
            assert vectors.shape == (3, 8)
            norms = vectors.norm(dim=-1)
            assert torch.allclose(norms, torch.ones(3), atol=1e-5)
        return


class TestConfiguration:
    def test_default_config_is_the_six_million_recipe(self):
        with given("the default autoencoder configuration"):
            config = AttentionAutoencoderConfig()

        with when("the model is built"):
            model = build_attention_autoencoder(config)
            parameters = sum(p.numel() for p in model.parameters())

        with then("it lands in the intended six-million parameter range"):
            assert 5_000_000 <= parameters <= 7_000_000
        return

    def test_config_validation(self):
        with given("invalid autoencoder configurations"):
            invalid = (
                {"latent_dim": 0},
                {"n_decoder_layers": 0},
            )

        with when("each config is constructed"):
            errors = []
            for overrides in invalid:
                try:
                    tiny_config(**overrides)
                except ValueError as error:
                    errors.append(error)

        with then("every invalid config is rejected"):
            assert len(errors) == len(invalid)
        return

    def test_factory_selects_autoencoder(self):
        with given("an autoencoder config"):
            config = tiny_config()

        with when("the factory builds the model"):
            model = build_model(config)

        with then("the autoencoder class is returned"):
            assert isinstance(model, AttentionAutoencoderModel)
        return

    def test_checkpoint_config_roundtrip(self):
        with given("a checkpoint config with the autoencoder arch"):
            values = {
                "arch": "attention_autoencoder",
                "d_model": 16,
                "n_heads": 2,
                "n_layers": 1,
                "n_decoder_layers": 1,
                "d_ff": 32,
                "max_seq_len": 1024,
                "max_segments": 8,
                "latent_dim": 8,
            }

        with when("the checkpoint config is reconstructed"):
            config = model_config_from_dict(values)

        with then("it is the autoencoder config"):
            assert isinstance(config, AttentionAutoencoderConfig)
        return

    def test_checkpoint_roundtrip(self):
        with given("an autoencoder and a fresh copy"):
            config = tiny_config()
            model = AttentionAutoencoderModel(config)
            restored = AttentionAutoencoderModel(copy.deepcopy(config))
            tokens, segment_ids, padding = pack(game_segments())

        with when("the state dict is transferred"):
            restored.load_state_dict(model.state_dict())

        with then("both produce identical latents"):
            model.eval()
            restored.eval()
            with torch.no_grad():
                expected = model.encode_latent(tokens, segment_ids, padding)
                actual = restored.encode_latent(tokens, segment_ids, padding)
            assert torch.allclose(expected, actual, atol=1e-6)
        return

    def test_loss_is_the_weighted_cross_entropy_of_the_gold_sequence(self):
        with given("known logits, tokens, and targets"):
            model = AttentionAutoencoderModel(tiny_config()).eval()
            tokens = torch.tensor([[3, 4, 1, 17, 2]])
            targets = torch.tensor([[-100, -100, 5, -100, -100]])
            logits = torch.randn(1, 5, VOCAB_SIZE)
            padding = torch.ones_like(tokens, dtype=torch.bool)
            gold = restoration_targets(tokens, targets)

        with when("the model loss is evaluated on those logits"):
            loss = model.loss_from_logits(logits, tokens, targets, padding)

        with then("it matches full-sequence cross entropy"):
            expected = F.cross_entropy(logits[0], gold[0])
            assert loss.item() == pytest.approx(expected.item(), rel=1e-5)
        return


class TestTrainingIntegration:
    def test_full_reconstruction_weights_supervise_visible_tokens(self):
        with given("a window dataset configured for full reconstruction"):
            games = [game_segments(6)]
            config = WindowDatasetConfig(
                games=games,
                max_seq_len=1024,
                max_segments=8,
                masking=MaskingConfig(
                    p_rect=1.0, p_random=0.0, p_board=0.0, p_span=0.0, p_skip_gram=0.0
                ),
                seed=0,
                force_mode="rect",
                balanced_modes=False,
                full_reconstruction_weights=True,
            )

        with when("one window is sampled"):
            sample = WindowDataset(config)[0]
            gold = np.where(
                sample["targets"] != IGNORE_INDEX, sample["targets"], sample["tokens"]
            )
            visible = gold != PAD

        with then("every non-pad position carries a training weight"):
            assert (sample["weights"][visible] > 0).all()
            assert (sample["weights"][~visible] == 0).all()
        return

    def test_masked_only_weighting_leaves_visible_tokens_unweighted(self):
        with given("a window dataset with the default masked-only weighting"):
            games = [game_segments(6)]
            config = WindowDatasetConfig(
                games=games,
                max_seq_len=1024,
                max_segments=8,
                masking=MaskingConfig(
                    p_rect=1.0, p_random=0.0, p_board=0.0, p_span=0.0, p_skip_gram=0.0
                ),
                seed=0,
                force_mode="rect",
                balanced_modes=False,
            )

        with when("one window is sampled"):
            sample = WindowDataset(config)[0]
            visible = sample["targets"] == IGNORE_INDEX

        with then("unmasked positions have zero weight"):
            assert (sample["weights"][visible] == 0).all()
        return

    def test_compute_training_loss_runs_with_autoencoder(self):
        with given("a masked window, an autoencoder, and a reconstruction-only loss"):
            model = AttentionAutoencoderModel(tiny_config())
            batch, _ = masked_batch(game_segments(8))
            batch["weights"] = torch.ones_like(batch["weights"])
            config = AuxiliaryLossConfig(
                mlm_weight=1.0,
                inv_weight=0.0,
                var_weight=0.0,
                cov_weight=0.0,
                nce_weight=0.0,
                alone_var_weight=0.0,
                alone_cov_weight=0.0,
                warmup_steps=0,
            )

        with when("the shared training loss is computed and backpropagated"):
            loss, components = compute_training_loss(model, batch, config, step=0)
            loss.backward()

        with then("the reconstruction loss is finite and reaches the model"):
            assert torch.isfinite(loss)
            assert "mlm" in components
            assert any(parameter.grad is not None for parameter in model.parameters())
        return


class TestFlashCompatibleAttention:
    def test_autoencoder_uses_the_varlen_attention_path(self, monkeypatch):
        with given("an autoencoder and a spy on the flash-compatible kernel"):
            model = AttentionAutoencoderModel(tiny_config()).eval()
            tokens, segment_ids, padding = pack(game_segments())
            calls = []
            import src.modeling.shared.layers as shared_layers

            original = shared_layers.varlen_attention

            def spy(*args, **kwargs):
                calls.append(kwargs)
                return original(*args, **kwargs)

            monkeypatch.setattr(shared_layers, "varlen_attention", spy)

        with when("the window is encoded and decoded"):
            with torch.no_grad():
                model(tokens, segment_ids, padding)

        with then("every block runs through the no-dense-mask attention helper"):
            assert len(calls) == 1 + 1
        return

    def test_flash_path_matches_the_dense_reference(self):
        with given("an autoencoder and a padded window"):
            model = AttentionAutoencoderModel(tiny_config()).eval()
            tokens, segment_ids, padding = pack(game_segments())
            padded_tokens, padded_ids, padded_mask = pad_window(
                tokens, segment_ids, extra=5
            )

        with when("the latents are compared with and without padding"):
            with torch.no_grad():
                plain = model.encode_latent(tokens, segment_ids, padding)
                padded = model.encode_latent(padded_tokens, padded_ids, padded_mask)

        with then("the varlen stream ignores the padded tail exactly"):
            assert torch.allclose(plain, padded, atol=1e-6)
        return
