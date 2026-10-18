import numpy as np
import pytest
import torch

from bdd import given, then, when
from src.dataset.token_stream import SEP
from src.modeling.masking import IGNORE_INDEX, MaskingConfig, mask_stream
from src.modeling.transformer import MaskedStreamTransformer, TransformerConfig
from src.training.window_dataset import collate_windows


def make_stream(board_factory, n_boards=6):
    from src.dataset.token_stream import TokenStreamEncoder, pack_stream

    encoder = TokenStreamEncoder()
    segments = [encoder.encode(board_factory(i)) for i in range(n_boards)]
    return pack_stream(segments, add_cls=True)


def random_game_factory(seed):
    import chess

    def factory(_):
        rng = np.random.default_rng(seed)
        board = chess.Board()
        for _ in range(int(rng.integers(20, 60))):
            moves = list(board.legal_moves)
            board.push(moves[int(rng.integers(len(moves)))])
            if board.is_game_over():
                board = chess.Board()
        return board

    return factory


def segment_flags(targets, sep_positions):
    """Per-segment flags telling whether every content token is masked."""
    starts = [1] + [int(s) + 1 for s in sep_positions[:-1]]
    flags = []
    for i, start in enumerate(starts):
        end = int(sep_positions[i])
        flags.append(all(targets[p] != IGNORE_INDEX for p in range(start, end)))
    return flags


class TestMaskStream:
    def test_random_mode_masks_segment_tokens(self):
        from src.dataset.token_stream import TURN_WHITE

        with given("a packed token stream and random-only masking"):
            tokens = make_stream(random_game_factory(1))
            config = MaskingConfig(
                p_random=1.0, p_board=0.0, p_span=0.0, random_rate=0.3
            )

        with when("the stream is masked"):
            masked, targets, weights = mask_stream(
                tokens, config, np.random.default_rng(0)
            )

        with then("only non-structural tokens are masked and targets keep originals"):
            changed = np.flatnonzero(targets != IGNORE_INDEX)
            assert changed.size > 0
            assert np.all(tokens[changed] >= TURN_WHITE)
            structural = np.flatnonzero((tokens == 3) | (tokens == 2))
            assert np.all(targets[structural] == IGNORE_INDEX)
            assert np.all(masked[structural] == tokens[structural])
            assert np.all(targets[changed] == tokens[changed])
        return

    def test_random_mode_rate(self):
        from src.dataset.token_stream import TURN_WHITE

        with given("a packed token stream and random-only masking at rate 0.3"):
            tokens = make_stream(random_game_factory(2))
            config = MaskingConfig(p_random=1.0, random_rate=0.3)

        with when("the stream is masked"):
            _, targets, _ = mask_stream(tokens, config, np.random.default_rng(0))

        with then("a fraction of the maskable tokens is masked"):
            n_maskable = int((tokens >= TURN_WHITE).sum())
            n_masked = int((targets != IGNORE_INDEX).sum())
            assert 0.2 * n_maskable <= n_masked <= 0.4 * n_maskable
        return

    def test_board_mode_masks_board_span_contiguous_segments(self):
        with given("a packed stream of 10 boards and board-only masking"):
            tokens = make_stream(random_game_factory(3), n_boards=10)
            config = MaskingConfig(p_random=0.0, p_board=1.0, p_span=0.0)

        with when("the stream is masked"):
            masked, targets, weights = mask_stream(
                tokens, config, np.random.default_rng(0)
            )

        with then("four contiguous full segments are masked with board-mode weights"):
            sep_positions = np.flatnonzero(tokens == SEP)
            flags = segment_flags(targets, sep_positions)
            flagged = np.flatnonzero(flags)
            assert len(flagged) == 4
            assert len(flagged) == flagged[-1] - flagged[0] + 1
            assert np.all(masked[np.flatnonzero(tokens == SEP)] == SEP)
            assert np.all(weights[targets != IGNORE_INDEX] == 2.0)
            assert np.all(weights[targets == IGNORE_INDEX] == 0.0)
        return

    def test_board_mode_masks_all_boards_when_stream_is_short(self):
        with given(
            "a packed stream of 3 boards, board-only masking, and no context gate"
        ):
            tokens = make_stream(random_game_factory(8), n_boards=3)
            config = MaskingConfig(
                p_random=0.0,
                p_board=1.0,
                p_span=0.0,
                board_span=4,
                min_context_segments=1,
            )

        with when("the stream is masked"):
            _, targets, _ = mask_stream(tokens, config, np.random.default_rng(0))

        with then("every board is fully masked"):
            sep_positions = np.flatnonzero(tokens == SEP)
            flags = segment_flags(targets, sep_positions)
            assert sum(flags) == 3
        return

    def test_board_mode_falls_back_to_random_without_context(self):
        with given("a packed stream with fewer than 8 boards and board-only masking"):
            tokens = make_stream(random_game_factory(7), n_boards=4)
            config = MaskingConfig(p_random=0.0, p_board=1.0, p_span=0.0)

        with when("the stream is masked"):
            _, targets, weights = mask_stream(tokens, config, np.random.default_rng(0))

        with then("masking falls back to random mode with random-mode weights"):
            changed = targets != IGNORE_INDEX
            assert changed.any()
            assert not any(segment_flags(targets, np.flatnonzero(tokens == SEP)))
            assert np.all(weights[changed] == 1.0)
        return

    def test_span_mode_falls_back_to_random_without_context(self):
        with given("a packed stream with fewer than 8 boards and span-only masking"):
            tokens = make_stream(random_game_factory(9), n_boards=5)
            config = MaskingConfig(p_random=0.0, p_board=0.0, p_span=1.0)

        with when("the stream is masked"):
            _, targets, weights = mask_stream(tokens, config, np.random.default_rng(0))

        with then("masking falls back to random mode with random-mode weights"):
            changed = targets != IGNORE_INDEX
            assert changed.any()
            assert np.all(weights[changed] == 1.0)
        return

    def test_span_mode_masks_contiguous_segments(self):
        with given("a packed token stream and span masking of 2 to 3 segments"):
            tokens = make_stream(random_game_factory(4), n_boards=8)
            config = MaskingConfig(
                p_random=0.0, p_board=0.0, p_span=1.0, min_span=2, max_span=3
            )

        with when("the stream is masked"):
            _, targets, _ = mask_stream(tokens, config, np.random.default_rng(0))

        with then("one contiguous run of 2 to 3 segments is masked"):
            sep_positions = np.flatnonzero(tokens == SEP)
            flags = segment_flags(targets, sep_positions)
            assert any(flags)
            flagged = np.flatnonzero(flags)
            assert len(flagged) == flagged[-1] - flagged[0] + 1
            assert 2 <= len(flagged) <= 3
        return

    def test_bert_replacement_distribution(self):
        with given("a long packed token stream and random masking at rate 0.5"):
            tokens = make_stream(random_game_factory(5), n_boards=40)
            config = MaskingConfig(p_random=1.0, random_rate=0.5)

        with when("the stream is masked"):
            masked, targets, _ = mask_stream(tokens, config, np.random.default_rng(0))

        with then("replacements follow the 80/10/10 mask/random/keep split"):
            changed = np.flatnonzero((targets != IGNORE_INDEX) & (targets >= 38))
            assert changed.size > 200
            mask_frac = float((masked[changed] == 1).mean())
            random_frac = float(
                ((masked[changed] >= 38) & (masked[changed] != targets[changed])).mean()
            )
            keep_frac = float((masked[changed] == targets[changed]).mean())
            assert mask_frac == pytest.approx(0.8, abs=0.1)
            assert random_frac == pytest.approx(0.1, abs=0.05)
            assert keep_frac == pytest.approx(0.1, abs=0.05)
        return

    def test_scalar_replacement_stays_in_token_class(self):
        from src.dataset.token_stream import (
            CASTLE_BASE,
            EP_BASE,
            PIECE_SQUARE_BASE,
            TURN_WHITE,
        )

        with given("many masked streams and random-only masking at rate 0.5"):
            config = MaskingConfig(
                p_random=1.0, p_board=0.0, p_span=0.0, random_rate=0.5
            )

        with when("every replaced token is bucketed by its token class"):
            replaced = {
                TURN_WHITE: [],
                CASTLE_BASE: [],
                EP_BASE: [],
                PIECE_SQUARE_BASE: [],
            }
            for seed in range(60):
                tokens = make_stream(random_game_factory(seed), n_boards=8)
                masked, targets, _ = mask_stream(
                    tokens, config, np.random.default_rng(seed)
                )
                for pos in np.flatnonzero(targets != IGNORE_INDEX):
                    low, _ = next(
                        (lo, hi)
                        for lo, hi in ((4, 6), (6, 22), (22, 38), (38, 806))
                        if lo <= targets[pos] < hi
                    )
                    if masked[pos] != 1 and masked[pos] != targets[pos]:
                        replaced[low].append(int(masked[pos]))

        with then("each class only receives replacements from its own range"):
            assert replaced[TURN_WHITE], "expected random-replaced turn tokens"
            assert replaced[CASTLE_BASE], "expected random-replaced castle tokens"
            assert all(4 <= v < 6 for v in replaced[TURN_WHITE])
            assert all(6 <= v < 22 for v in replaced[CASTLE_BASE])
            assert all(22 <= v < 38 for v in replaced[EP_BASE])
            assert all(38 <= v < 806 for v in replaced[PIECE_SQUARE_BASE])
        return

    def test_deterministic_with_same_seed(self):
        with given("a packed token stream and default masking"):
            tokens = make_stream(random_game_factory(6))
            config = MaskingConfig()

        with when("the stream is masked twice with the same seed"):
            a = mask_stream(tokens, config, np.random.default_rng(42))
            b = mask_stream(tokens, config, np.random.default_rng(42))

        with then("both runs produce identical masks, targets, and weights"):
            assert np.array_equal(a[0], b[0])
            assert np.array_equal(a[1], b[1])
            assert np.array_equal(a[2], b[2])
        return

    def test_random_mode_uses_random_mode_weight(self):
        with given("a long packed stream and random-only masking"):
            tokens = make_stream(random_game_factory(10), n_boards=12)
            config = MaskingConfig(p_random=1.0, p_board=0.0, p_span=0.0)

        with when("the stream is masked"):
            _, targets, weights = mask_stream(tokens, config, np.random.default_rng(0))

        with then("every masked position carries the random-mode weight"):
            assert np.all(weights[targets != IGNORE_INDEX] == 1.0)
        return

    def test_mode_weights_are_configurable(self):
        with given("a long packed stream and board-only masking with custom weights"):
            tokens = make_stream(random_game_factory(11), n_boards=10)
            config = MaskingConfig(
                p_random=0.0,
                p_board=1.0,
                p_span=0.0,
                mode_weights={"random": 1.0, "board": 3.5, "span": 2.0},
            )

        with when("the stream is masked"):
            _, targets, weights = mask_stream(tokens, config, np.random.default_rng(0))

        with then("masked positions carry the configured board-mode weight"):
            assert np.all(weights[targets != IGNORE_INDEX] == 3.5)
        return


TINY_CONFIG = TransformerConfig(
    d_model=32,
    n_heads=4,
    n_layers=2,
    d_ff=64,
    dropout=0.0,
    max_seq_len=64,
    max_segments=16,
)


class TestTransformerConfig:
    def test_default_architecture_is_scaled_up(self):
        with given("a transformer config with no explicit dimensions"):
            config = TransformerConfig()

        with then("the default architecture is the scaled-up configuration"):
            assert config.d_model == 512
            assert config.n_heads == 16
            assert config.n_layers == 8
            assert config.d_ff == 2048
        return


def tiny_batch(n_segments=3, batch=2, seed=0):
    tokens = make_stream(random_game_factory(seed), n_boards=n_segments)
    samples = [
        {
            "tokens": tokens[: 10 + 7 * i],
            "targets": np.full(10 + 7 * i, IGNORE_INDEX, dtype=np.int16),
        }
        for i in range(batch)
    ]
    return collate_windows(samples)


class TestTransformer:
    def test_forward_shapes_and_loss(self):
        with given("a tiny masked stream transformer in eval mode and a batch"):
            model = MaskedStreamTransformer(TINY_CONFIG)
            model.eval()
            batch = tiny_batch()

        with when("the model runs a forward pass with targets"):
            output = model(
                batch["tokens"],
                batch["segment_ids"],
                padding_mask=batch["padding_mask"],
                targets=batch["targets"],
            )

        with then("logits span the vocabulary and the loss is finite"):
            assert output["logits"].shape == (*batch["tokens"].shape, 806)
            assert torch.isfinite(output["loss"])
        return

    def test_padding_does_not_change_real_outputs(self):
        with given("a tiny model in eval mode and a padded batch"):
            model = MaskedStreamTransformer(TINY_CONFIG)
            model.eval()
            batch = tiny_batch()
            tokens = batch["tokens"]

        with when("each row is re-run alone without padding"):
            with torch.no_grad():
                out_full = model(tokens, batch["segment_ids"], batch["padding_mask"])[
                    "logits"
                ]
                singles = []
                for row in range(tokens.shape[0]):
                    length = int(batch["padding_mask"][row].sum())
                    out_single = model(
                        tokens[row : row + 1, :length],
                        batch["segment_ids"][row : row + 1, :length],
                        batch["padding_mask"][row : row + 1, :length],
                    )["logits"]
                    singles.append(out_single[0])

        with then("padded and unpadded outputs agree on real positions"):
            for row, single in enumerate(singles):
                length = int(batch["padding_mask"][row].sum())
                assert torch.allclose(out_full[row, :length], single, atol=1e-5)
        return

    def test_loss_ignores_unmasked_and_pad(self):
        with given("a tiny model in eval mode and a batch with no supervised targets"):
            model = MaskedStreamTransformer(TINY_CONFIG)
            model.eval()
            batch = tiny_batch()
            targets = torch.full_like(batch["targets"], IGNORE_INDEX)

        with when("the model runs a forward pass with those targets"):
            with torch.no_grad():
                loss = model(
                    batch["tokens"],
                    batch["segment_ids"],
                    batch["padding_mask"],
                    targets,
                )["loss"]

        with then("the loss is zero"):
            assert loss.item() == 0.0
        return

    def test_weighted_loss_matches_manual_weighted_mean(self):
        with given("synthetic logits, targets, and per-target weights"):
            logits = torch.zeros(1, 3, 10)
            logits[0, 0, 3] = 100.0
            logits[0, 1, 5] = 100.0
            targets = torch.tensor([[2, 5, IGNORE_INDEX]])
            weights = torch.tensor([[10.0, 1.0, 0.0]])

        with when("the weighted and unweighted losses are computed"):
            weighted = MaskedStreamTransformer._loss(logits, targets, weights)
            unweighted = MaskedStreamTransformer._loss(logits, targets)

        with then("the weighted loss equals the manually computed weighted mean"):
            ce0 = torch.nn.functional.cross_entropy(
                logits[0, 0].unsqueeze(0), targets[0, 0].unsqueeze(0)
            )
            ce1 = torch.nn.functional.cross_entropy(
                logits[0, 1].unsqueeze(0), targets[0, 1].unsqueeze(0)
            )
            expected = (10.0 * ce0 + 1.0 * ce1) / 11.0
            assert weighted.item() == pytest.approx(expected.item(), rel=1e-4)
            assert weighted.item() != pytest.approx(unweighted.item(), rel=1e-3)
        return

    def test_forward_accepts_weights(self):
        with given("a tiny model in eval mode and a collated batch with weights"):
            model = MaskedStreamTransformer(TINY_CONFIG)
            model.eval()
            batch = tiny_batch()
            weights = (batch["targets"] != IGNORE_INDEX).float() + 1.0

        with when("the model runs a forward pass with those weights"):
            with torch.no_grad():
                output = model(
                    batch["tokens"],
                    batch["segment_ids"],
                    padding_mask=batch["padding_mask"],
                    targets=batch["targets"],
                    weights=weights,
                )

        with then("the loss is finite"):
            assert torch.isfinite(output["loss"])
        return

    def test_segment_embedding_has_effect(self):
        with given("a tiny model in eval mode and a batch"):
            model = MaskedStreamTransformer(TINY_CONFIG)
            model.eval()
            batch = tiny_batch()

        with when("the model is run with original and shifted segment ids"):
            shifted = batch["segment_ids"] + 1
            with torch.no_grad():
                base = model(
                    batch["tokens"], batch["segment_ids"], batch["padding_mask"]
                )["hidden_states"]
                other = model(batch["tokens"], shifted, batch["padding_mask"])[
                    "hidden_states"
                ]

        with then("the hidden states differ"):
            assert not torch.allclose(base, other)
        return

    def test_embed_pools_segments_correctly(self):
        with given("a tiny model in eval mode and a four-segment batch"):
            model = MaskedStreamTransformer(TINY_CONFIG)
            model.eval()
            batch = tiny_batch(n_segments=4, batch=2)

        with when("the model runs forward and pools segment embeddings"):
            tokens = batch["tokens"]
            segments = batch["segment_ids"]
            with torch.no_grad():
                hidden = model(tokens, segments, batch["padding_mask"])["hidden_states"]
                pooled, mask = model.embed(tokens, segments, batch["padding_mask"])

        with then("each pooled segment averages its real content positions"):
            row = 0
            valid = batch["padding_mask"][row]
            real_segments = sorted(set(segments[row][valid].tolist()))
            for s in real_segments:
                positions = (
                    valid & segments[row].eq(s) & tokens[row].ne(2) & tokens[row].ne(3)
                )
                expected = hidden[row][positions].mean(dim=0)
                assert torch.allclose(pooled[row, s], expected, atol=1e-6)
            assert mask[row, : len(real_segments)].all()
            assert not mask[row, len(real_segments) :].any()
        return
