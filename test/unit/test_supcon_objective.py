from __future__ import annotations

import chess
import numpy as np
import pytest
import torch

from bdd import given, then, when
from src.dataset.token_stream import SEP, TokenStreamEncoder, pack_stream
from src.modeling.masking import IGNORE_INDEX
from src.modeling.transformer import MaskedStreamTransformer, TransformerConfig
from src.training.contrastive import build_contrastive_masks
from src.training.objectives import (
    AuxiliaryLossConfig,
    compute_training_loss,
)


def clustered(
    segment_mask: torch.Tensor,
    start_index: torch.Tensor,
    seed: int = 0,
    **overrides,
) -> list[list[int]]:
    """Call the clustered selector with the Option 2 defaults."""
    options = {
        "count": 8,
        "temporal_radius": 3,
        "opening_ply_threshold": 0,
    }
    options.update(overrides)
    from src.training.objectives import select_clustered_target_segments

    return select_clustered_target_segments(
        segment_mask,
        start_index,
        generator=torch.Generator().manual_seed(seed),
        **options,
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


def three_segment_batch(start_index: int) -> dict[str, torch.Tensor]:
    """A one-window batch whose metadata claims the given starting ply."""
    encoder = TokenStreamEncoder()
    board = chess.Board()
    segments = []
    for _ in range(3):
        segments.append(encoder.encode(board))
        board.push(list(board.legal_moves)[0])
    stream = pack_stream(segments)
    tokens = torch.from_numpy(stream.astype(np.int64))[None]
    segment_ids = torch.from_numpy(np.cumsum(stream == SEP).astype(np.int64))[None]
    return {
        "tokens": tokens,
        "segment_ids": segment_ids,
        "padding_mask": torch.ones_like(tokens, dtype=torch.bool),
        "targets": torch.full_like(tokens, IGNORE_INDEX),
        "weights": torch.zeros_like(tokens, dtype=torch.float),
        "game_id": torch.tensor([0]),
        "start_index": torch.tensor([start_index]),
    }


def supcon_config(**overrides) -> AuxiliaryLossConfig:
    values = {
        "mlm_weight": 1.0,
        "inv_weight": 0.0,
        "var_weight": 0.0,
        "cov_weight": 0.0,
        "nce_weight": 0.2,
        "alone_var_weight": 0.0,
        "alone_cov_weight": 0.0,
        "warmup_steps": 0,
        "use_supcon": True,
        "opening_ply_threshold": 30,
    }
    values.update(overrides)
    return AuxiliaryLossConfig(**values)


class TestClusteredTargetSelection:
    def test_every_selected_target_has_a_neighbour_within_the_radius(self):
        with given("a 60-segment window with every segment valid"):
            mask = torch.ones(1, 60, dtype=torch.bool)
            start = torch.tensor([0])

        with when("eight targets are selected with a three-ply radius"):
            targets = clustered(mask, start, count=8, temporal_radius=3)

        with then("the targets are distinct and mutually clustered"):
            selected = targets[0]
            assert len(selected) == 8
            assert len(set(selected)) == 8
            assert all(
                any(other != target and abs(other - target) <= 3 for other in selected)
                for target in selected
            )
        return

    def test_opening_positions_are_never_selected(self):
        with given("a window starting at ply zero with a 30-ply threshold"):
            mask = torch.ones(1, 60, dtype=torch.bool)
            start = torch.tensor([0])

        with when("targets are selected"):
            targets = clustered(mask, start, opening_ply_threshold=30)

        with then("every selected position is past the opening"):
            assert targets[0]
            assert all(target >= 30 for target in targets[0])
        return

    def test_the_start_index_is_used_as_the_global_ply(self):
        with given("a window whose first segment is already at ply ten"):
            mask = torch.ones(1, 60, dtype=torch.bool)
            start = torch.tensor([10])

        with when("targets are selected with a 30-ply threshold"):
            targets = clustered(mask, start, opening_ply_threshold=30)

        with then("opening filtering uses start_index plus the segment index"):
            assert targets[0]
            assert all(start.item() + target >= 30 for target in targets[0])
        return

    def test_a_window_with_only_opening_positions_yields_no_targets(self):
        with given("a 20-segment window entirely inside the opening"):
            mask = torch.ones(1, 20, dtype=torch.bool)
            start = torch.tensor([0])

        with when("targets are selected with a 30-ply threshold"):
            targets = clustered(mask, start, opening_ply_threshold=30)

        with then("the window contributes no alone views"):
            assert targets == [[]]
        return

    def test_masked_out_segments_are_never_selected(self):
        with given("a window with a masked gap in the middle"):
            mask = torch.tensor([[True] * 20 + [False] * 10 + [True] * 30])
            start = torch.tensor([0])

        with when("targets are selected"):
            targets = clustered(mask, start)

        with then("every target is a valid segment"):
            assert targets[0]
            assert all(bool(mask[0, target]) for target in targets[0])
        return

    def test_selection_is_deterministic_for_a_fixed_generator(self):
        with given("a valid window and two identically seeded calls"):
            mask = torch.ones(1, 60, dtype=torch.bool)
            start = torch.tensor([0])

        with when("targets are selected twice"):
            first = clustered(mask, start, seed=7)
            second = clustered(mask, start, seed=7)

        with then("the selected clusters match"):
            assert first == second
        return


class TestSupConMasksWithoutOpeningIgnore:
    def test_clustered_targets_never_produce_views_without_a_positive(self):
        with given("eight clustered targets from one game"):
            mask = torch.ones(1, 60, dtype=torch.bool)
            start = torch.tensor([0])
            targets = clustered(mask, start, count=8, temporal_radius=3)
            plies = [start.item() + target for target in targets[0]]

        with when("the contrastive masks are built"):
            positive, ignore = build_contrastive_masks(
                [0] * len(plies), plies, temporal_radius=3
            )

        with then("every view has at least one positive"):
            assert positive.any(dim=1).all()
            assert ignore.diagonal().sum() == 0
        return

    def test_cross_game_opening_positions_are_ordinary_negatives(self):
        with given("two opening positions from different games"):
            game_ids = ["a", "b"]
            plies = [10, 10]

        with when("the contrastive masks are built"):
            positive, ignore = build_contrastive_masks(
                game_ids, plies, temporal_radius=3
            )

        with then("they are neither positives nor ignored"):
            assert not positive[0, 1]
            assert not ignore[0, 1]
        return


class TestRemovedOptions:
    def test_turn_flip_option_is_removed(self):
        with given("the Option 2 configuration"):
            config = AuxiliaryLossConfig()

        with then("there is no turn-flip flag to add degenerate positives"):
            assert "turn_flip" not in AuxiliaryLossConfig.__dataclass_fields__
            assert not hasattr(config, "turn_flip")
        return

    def test_ignore_opening_negatives_option_is_removed(self):
        with given("the Option 2 configuration"):
            config = AuxiliaryLossConfig()

        with then("opening positions are excluded at selection instead"):
            assert (
                "ignore_opening_negatives"
                not in AuxiliaryLossConfig.__dataclass_fields__
            )
            assert not hasattr(config, "ignore_opening_negatives")
        return


class TestSupConTrainingIntegration:
    def test_opening_only_batches_skip_supcon(self):
        with given("a batch window entirely inside the opening"):
            model = tiny_model()
            batch = three_segment_batch(start_index=0)
            config = supcon_config()

        with when("the training loss is computed"):
            total, components = compute_training_loss(model, batch, config)

        with then("only the MLM term is produced"):
            assert "supcon" not in components
            assert torch.isfinite(total)
        return

    def test_supcon_uses_isolated_window_and_alone_views(self):
        with given("a non-opening batch window"):
            model = tiny_model()
            batch = three_segment_batch(start_index=40)
            config = supcon_config()
            calls: list[int] = []
            original = model.encode_views

            def spy(*args, **kwargs):
                calls.append(1)
                return original(*args, **kwargs)

            model.encode_views = spy

        with when("the training loss is computed"):
            total, components = compute_training_loss(model, batch, config)

        with then("the window and alone views each use content isolation"):
            assert "supcon" in components
            assert len(calls) == 2
            assert torch.isfinite(total)
        return


class TestRemovedCliFlags:
    def test_turn_flip_flag_is_gone(self, monkeypatch):
        with given("the removed --turn-flip flag"):
            monkeypatch.setattr("sys.argv", ["train_masked_transformer", "--turn-flip"])

        with when("the arguments are parsed"):
            from src.run.train_masked_transformer import parse_args

            with pytest.raises(SystemExit):
                parse_args()
        return

    def test_ignore_opening_negatives_flag_is_gone(self, monkeypatch):
        with given("the removed --ignore-opening-negatives flag"):
            monkeypatch.setattr(
                "sys.argv",
                ["train_masked_transformer", "--ignore-opening-negatives"],
            )

        with when("the arguments are parsed"):
            from src.run.train_masked_transformer import parse_args

            with pytest.raises(SystemExit):
                parse_args()
        return
