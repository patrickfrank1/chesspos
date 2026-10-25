from __future__ import annotations

import chess
import numpy as np

from bdd import given, then, when
from src.dataset.token_stream import CLS, SEP, TokenStreamEncoder, pack_stream
from src.modeling.masking import (
    IGNORE_INDEX,
    MaskingConfig,
    mask_stream,
    segment_bounds,
)
from src.training.window_dataset import build_mode_pattern


def packed_stream(n_boards: int = 5) -> np.ndarray:
    encoder = TokenStreamEncoder()
    board = chess.Board()
    segments = []
    for _ in range(n_boards):
        segments.append(encoder.encode(board))
        board.push(list(board.legal_moves)[0])
    return pack_stream(segments)


def rectangle_config(**kwargs) -> MaskingConfig:
    defaults = dict(
        p_random=0.0,
        p_board=0.0,
        p_span=0.0,
        p_rect=1.0,
        rect_min_rate=0.30,
        rect_max_rate=0.30,
    )
    defaults.update(kwargs)
    return MaskingConfig(**defaults)


class TestRectangleMasking:
    def test_masks_thirty_percent_of_each_board(self):
        with given("a packed stream and rectangle-only masking at 30%"):
            tokens = packed_stream()
            config = rectangle_config()

        with when("the stream is masked"):
            masked, targets, weights = mask_stream(
                tokens, config, np.random.default_rng(0)
            )

        with then("each segment has 19 masked squares and only square offsets"):
            for start, end in segment_bounds(tokens):
                offsets = np.flatnonzero(targets[start:end] != IGNORE_INDEX)
                assert len(offsets) == 19
                assert np.all(offsets < 64)
                assert np.all(weights[start + offsets] == 1.0)
            assert np.all(masked[tokens == SEP] == SEP)
        return

    def test_synchronizes_the_same_squares_across_segments(self):
        with given("rectangle-only masking on a multi-segment window"):
            tokens = packed_stream()
            config = rectangle_config()

        with when("the stream is masked"):
            _, targets, _ = mask_stream(tokens, config, np.random.default_rng(0))

        with then("every segment masks the same square offsets"):
            masks = []
            for start, end in segment_bounds(tokens):
                masks.append(
                    tuple(np.flatnonzero(targets[start:end] != IGNORE_INDEX).tolist())
                )
            assert all(mask == masks[0] for mask in masks)
        return

    def test_leaves_structural_and_state_tokens_unmasked(self):
        with given("rectangle-only masking"):
            tokens = packed_stream()
            config = rectangle_config()

        with when("the stream is masked"):
            _, targets, _ = mask_stream(tokens, config, np.random.default_rng(0))

        with then("CLS, SEP, turn, castling and en-passant are never targets"):
            structural = np.flatnonzero((tokens == CLS) | (tokens == SEP))
            assert np.all(targets[structural] == IGNORE_INDEX)
            for start, end in segment_bounds(tokens):
                assert np.all(targets[start + 64 : start + 67] == IGNORE_INDEX)
        return

    def test_rate_stays_within_the_configured_range(self):
        with given("rectangle-only masking with the 30-50% range"):
            tokens = packed_stream()
            config = rectangle_config(rect_min_rate=0.30, rect_max_rate=0.50)

        with when("several streams are masked"):
            counts = []
            for seed in range(10):
                _, targets, _ = mask_stream(tokens, config, np.random.default_rng(seed))
                for start, end in segment_bounds(tokens):
                    counts.append(int((targets[start:end] != IGNORE_INDEX).sum()))

        with then("every board stays within 30-50% coverage"):
            assert all(19 <= count <= 32 for count in counts)
        return

    def test_rectangle_is_the_only_default_training_mode(self):
        with given("a rectangle-only masking config"):
            config = rectangle_config()

        with when("the balanced mode pattern is built"):
            pattern = build_mode_pattern(config)

        with then("every slot selects rectangle masking"):
            assert set(pattern) == {"rect"}
        return


class TestRectangleConfigValidation:
    def test_invalid_rate_range_is_rejected(self):
        with given("a rectangle range whose minimum exceeds its maximum"):
            error = None

        with when("the config is constructed"):
            try:
                MaskingConfig(rect_min_rate=0.6, rect_max_rate=0.4)
            except ValueError as exc:
                error = exc

        with then("a clear error is raised"):
            assert error is not None
        return
