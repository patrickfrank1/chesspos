from __future__ import annotations

import chess
import numpy as np
import pytest
import torch

from bdd import given, then, when
from src.dataset.token_stream import (
    CLS,
    PAD,
    SEP,
    TokenStreamEncoder,
    pack_stream,
)
from src.modeling.masking import MaskingConfig, mask_stream
from src.modeling.transformer import MaskedStreamTransformer, TransformerConfig
from src.training.objectives import (
    AuxiliaryLossConfig,
    alignment_loss,
    auxiliary_losses,
    compute_training_loss,
    extract_alone_views,
    info_nce_loss,
    select_target_segments,
    vicreg_terms,
)


def tiny_model() -> MaskedStreamTransformer:
    return MaskedStreamTransformer(
        TransformerConfig(
            vocab_size=52,
            d_model=16,
            n_heads=2,
            n_layers=1,
            d_ff=32,
            dropout=0.0,
            max_seq_len=256,
            max_segments=8,
        )
    )


def tiny_batch() -> dict[str, torch.Tensor]:
    encoder = TokenStreamEncoder()
    board = chess.Board()
    segments = []
    for _ in range(3):
        segments.append(encoder.encode(board))
        board.push(list(board.legal_moves)[0])
    tokens = pack_stream(segments)
    masking = MaskingConfig(
        p_rect=1.0, p_random=0.0, p_board=0.0, p_span=0.0, p_skip_gram=0.0
    )
    masked, targets, weights = mask_stream(tokens, masking, np.random.default_rng(0))
    segment_ids = np.cumsum(masked == SEP)
    return {
        "tokens": torch.from_numpy(masked.astype(np.int64))[None],
        "segment_ids": torch.from_numpy(segment_ids.astype(np.int64))[None],
        "padding_mask": torch.from_numpy(masked != PAD)[None],
        "targets": torch.from_numpy(targets.astype(np.int64))[None],
        "weights": torch.from_numpy(weights)[None],
    }


class TestVICReg:
    def test_variance_penalizes_collapsed_dimensions(self):
        with given("a batch of identical embeddings"):
            embeddings = torch.ones(8, 4)

        with when("the VICReg variance term is computed"):
            variance, covariance = vicreg_terms(embeddings, gamma=1.0, eps=1e-4)

        with then("every dimension is penalised for collapsing"):
            assert variance == pytest.approx(1.0, abs=0.02)
            assert covariance == pytest.approx(0.0)
        return

    def test_variance_is_zero_at_target_scale(self):
        with given("standardised independent embeddings"):
            torch.manual_seed(0)
            embeddings = torch.randn(512, 8)
            embeddings = (embeddings - embeddings.mean(0)) / embeddings.std(0)

        with when("the VICReg terms are computed"):
            variance, covariance = vicreg_terms(embeddings, gamma=1.0, eps=1e-4)

        with then("neither term penalises the batch"):
            assert variance == pytest.approx(0.0, abs=1e-3)
            assert covariance < 0.05
        return

    def test_covariance_penalizes_correlated_dimensions(self):
        with given("two identical embedding dimensions"):
            base = torch.randn(256, 1)
            embeddings = torch.cat([base, base, torch.randn(256, 1)], dim=1)

        with when("the VICReg covariance term is computed"):
            _, covariance = vicreg_terms(embeddings)

        with then("the correlated pair is penalised"):
            assert covariance > 0.1
        return


class TestInfoNCE:
    def test_aligned_pairs_have_lower_loss_than_shuffled_pairs(self):
        with given("a batch of embeddings and their aligned copies"):
            torch.manual_seed(0)
            embeddings = torch.randn(16, 8)

        with when("InfoNCE is computed for aligned and shuffled pairs"):
            aligned = info_nce_loss(embeddings, embeddings, temperature=0.1)
            shuffled = info_nce_loss(
                embeddings, embeddings[torch.randperm(16)], temperature=0.1
            )

        with then("the true pairs are easier than random pairs"):
            assert aligned < shuffled
        return


class TestContextViews:
    def test_extracts_only_the_target_segment_from_a_window(self):
        with given("a packed window with three segments"):
            segments = [np.arange(3) + 10, np.arange(4) + 20, np.arange(5) + 30]
            tokens = torch.from_numpy(pack_stream(segments).astype(np.int64))[None]
            segment_ids = torch.from_numpy(
                np.cumsum(pack_stream(segments) == SEP).astype(np.int64)
            )[None]
            padding = torch.ones_like(tokens, dtype=torch.bool)
            target_ids = torch.tensor([1])

        with when("the alone view for the middle segment is extracted"):
            alone_tokens, alone_segments, alone_padding = extract_alone_views(
                tokens, segment_ids, padding, target_ids
            )

        with then("the view contains only that segment with CLS/SEP framing"):
            expected = pack_stream([segments[1]])
            assert alone_tokens.shape == (1, len(expected))
            assert alone_tokens[0].tolist() == expected.tolist()
            assert alone_tokens[0, 0] == CLS
            assert alone_tokens[0, -1] == SEP
            assert alone_padding.all()
            assert alone_segments[0, -1] == 1
            assert torch.all(alone_segments[0, :-1] == 0)
        return

    def test_pads_windows_with_different_segment_lengths(self):
        with given("two windows whose target segments have different lengths"):
            short = pack_stream([np.arange(3) + 10])
            long = pack_stream([np.arange(4) + 20])
            tokens = torch.zeros((2, len(long)), dtype=torch.long)
            tokens[0, : len(short)] = torch.from_numpy(short.astype(np.int64))
            tokens[1, : len(long)] = torch.from_numpy(long.astype(np.int64))
            segment_ids = torch.zeros((2, len(long)), dtype=torch.long)
            segment_ids[0, : len(short)] = torch.from_numpy(
                np.cumsum(short == SEP).astype(np.int64)
            )
            segment_ids[1, : len(long)] = torch.from_numpy(
                np.cumsum(long == SEP).astype(np.int64)
            )
            padding = torch.zeros((2, len(long)), dtype=torch.bool)
            padding[0, : len(short)] = True
            padding[1, : len(long)] = True

        with when("the alone views are extracted"):
            alone_tokens, _, alone_padding = extract_alone_views(
                tokens, segment_ids, padding, torch.tensor([0, 0])
            )

        with then("the shorter view is padded and the longer one is intact"):
            assert alone_tokens.shape == (2, len(long))
            assert alone_tokens[0, : len(short)].tolist() == short.tolist()
            assert not alone_padding[0, len(short) :].any()
            assert alone_tokens[1].tolist() == long.tolist()
            assert alone_padding[1].all()
        return

    def test_selects_a_valid_segment_per_sample(self):
        with given("a segment mask with different valid counts"):
            mask = torch.tensor(
                [[True, True, False], [True, False, False], [True, True, True]]
            )

        with when("target segments are sampled"):
            targets = select_target_segments(
                mask, generator=torch.Generator().manual_seed(0)
            )

        with then("every selected segment is valid"):
            assert targets.shape == (3,)
            for row, target in enumerate(targets.tolist()):
                assert mask[row, target]
        return


class TestAuxiliaryLosses:
    def test_default_weights_match_the_objective_sweep(self):
        with given("a default auxiliary loss configuration"):
            config = AuxiliaryLossConfig()

        with then("MLM, strong VICReg, invariance and InfoNCE are enabled"):
            assert config.mlm_weight == 1.0
            assert config.inv_weight == 0.5
            assert config.var_weight == 1.0
            assert config.cov_weight == 0.1
            assert config.nce_weight == 0.1
            assert config.nce_temperature == 0.1
            assert config.warmup_steps == 200
            assert config.auxiliary_enabled() is True
        return

    def test_zero_weights_leave_only_mlm(self):
        with given("an MLM loss and all auxiliary weights at zero"):
            mlm = torch.tensor(2.0)
            config = AuxiliaryLossConfig(
                mlm_weight=2.0,
                inv_weight=0.0,
                var_weight=0.0,
                cov_weight=0.0,
                nce_weight=0.0,
            )
            embeddings = torch.randn(8, 4)
            targets = torch.randn(8, 4)

        with when("the total loss is assembled"):
            total, components = auxiliary_losses(
                mlm, embeddings, targets, targets, config
            )

        with then("the result is the weighted MLM loss only"):
            assert total == pytest.approx(4.0)
            assert set(components) == {"mlm"}
        return

    def test_zero_nce_weight_removes_the_nce_term(self):
        with given("invariance enabled and InfoNCE disabled"):
            mlm = torch.tensor(1.0)
            config = AuxiliaryLossConfig(
                mlm_weight=1.0,
                inv_weight=0.5,
                var_weight=0.0,
                cov_weight=0.0,
                nce_weight=0.0,
            )
            targets = torch.randn(8, 4)

        with when("the total loss is assembled"):
            _, components = auxiliary_losses(mlm, None, targets, targets, config)

        with then("only MLM and invariance contribute"):
            assert "nce" not in components
            assert "inv" in components
        return

    def test_nce_weight_includes_the_nce_term(self):
        with given("a positive InfoNCE weight"):
            mlm = torch.tensor(1.0)
            config = AuxiliaryLossConfig(
                mlm_weight=1.0,
                inv_weight=0.0,
                var_weight=0.0,
                cov_weight=0.0,
                nce_weight=0.2,
            )
            targets = torch.randn(8, 4)

        with when("the total loss is assembled"):
            _, components = auxiliary_losses(mlm, None, targets, targets, config)

        with then("the NCE component is present"):
            assert "nce" in components
        return

    def test_warmup_ramps_auxiliary_weights(self):
        with given("a warmup over ten steps and invariance enabled"):
            mlm = torch.tensor(1.0)
            config = AuxiliaryLossConfig(
                mlm_weight=1.0,
                inv_weight=1.0,
                var_weight=0.0,
                cov_weight=0.0,
                nce_weight=0.0,
                warmup_steps=10,
            )
            targets = torch.randn(8, 4)

        with when("the loss is assembled at the start, middle and end"):
            start, _ = auxiliary_losses(mlm, None, targets, targets, config, step=0)
            middle, _ = auxiliary_losses(mlm, None, targets, targets, config, step=5)
            end, _ = auxiliary_losses(mlm, None, targets, targets, config, step=10)

        with then("the invariance contribution ramps from zero to full"):
            inv = alignment_loss(targets, targets)
            assert start == pytest.approx(mlm.item())
            assert middle == pytest.approx(mlm.item() + 0.5 * inv.item())
            assert end == pytest.approx(mlm.item() + inv.item())
        return


class TestComputeTrainingLoss:
    def test_mlm_only_configuration_matches_the_model_loss(self):
        with given("a tiny model and batch with all auxiliary weights at zero"):
            model = tiny_model()
            batch = tiny_batch()
            config = AuxiliaryLossConfig(
                mlm_weight=1.0,
                inv_weight=0.0,
                var_weight=0.0,
                cov_weight=0.0,
                nce_weight=0.0,
            )

        with when("the training loss is computed"):
            total, components = compute_training_loss(model, batch, config)

        with then("it is exactly the MLM loss"):
            with torch.no_grad():
                reference = model(
                    batch["tokens"],
                    batch["segment_ids"],
                    padding_mask=batch["padding_mask"],
                    targets=batch["targets"],
                    weights=batch["weights"],
                )["loss"]
            assert total.item() == pytest.approx(reference.item())
            assert set(components) == {"mlm"}
        return

    def test_context_views_and_vicreg_terms_are_added(self):
        with given("a tiny model, batch, and geometry weights"):
            model = tiny_model()
            batch = tiny_batch()
            config = AuxiliaryLossConfig(
                mlm_weight=1.0,
                inv_weight=0.1,
                var_weight=0.1,
                cov_weight=0.01,
                nce_weight=0.0,
            )

        with when("the training loss is computed and backpropagated"):
            total, components = compute_training_loss(
                model,
                batch,
                config,
                generator=torch.Generator().manual_seed(0),
            )
            total.backward()

        with then("invariance, variance and covariance are present"):
            assert {"mlm", "inv", "var", "cov"} == set(components)
            assert "nce" not in components
            assert all(
                parameter.grad is not None
                for parameter in model.parameters()
                if parameter.requires_grad
            )
        return

    def test_info_nce_can_be_enabled_without_changing_the_interface(self):
        with given("a tiny model, batch, and a non-zero NCE weight"):
            model = tiny_model()
            batch = tiny_batch()
            config = AuxiliaryLossConfig(
                mlm_weight=1.0,
                inv_weight=0.0,
                var_weight=0.0,
                cov_weight=0.0,
                nce_weight=0.5,
            )

        with when("the training loss is computed"):
            total, components = compute_training_loss(
                model,
                batch,
                config,
                generator=torch.Generator().manual_seed(0),
            )

        with then("the NCE component is included"):
            assert "nce" in components
            assert total.requires_grad
        return
