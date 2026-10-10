import numpy as np
import pytest
import torch

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


class TestMaskStream:
    def test_random_mode_masks_only_piece_tokens(self):
        tokens = make_stream(random_game_factory(1))
        config = MaskingConfig(p_random=1.0, p_board=0.0, p_span=0.0, random_rate=0.3)
        rng = np.random.default_rng(0)
        masked, targets = mask_stream(tokens, config, rng)

        changed = np.flatnonzero(targets != IGNORE_INDEX)
        assert changed.size > 0
        assert np.all(tokens[changed] >= 38)
        structural = np.flatnonzero((tokens == 3) | (tokens == 2))
        assert np.all(targets[structural] == IGNORE_INDEX)
        assert np.all(masked[structural] == tokens[structural])
        assert np.all(targets[changed] == tokens[changed])

    def test_random_mode_rate(self):
        tokens = make_stream(random_game_factory(2))
        config = MaskingConfig(p_random=1.0, random_rate=0.3)
        rng = np.random.default_rng(0)
        _, targets = mask_stream(tokens, config, rng)
        n_piece = int((tokens >= 38).sum())
        n_masked = int((targets != IGNORE_INDEX).sum())
        assert 0.2 * n_piece <= n_masked <= 0.4 * n_piece

    def test_board_mode_masks_exactly_one_segment(self):
        from src.dataset.token_stream import SEP

        tokens = make_stream(random_game_factory(3))
        config = MaskingConfig(p_random=0.0, p_board=1.0, p_span=0.0)
        rng = np.random.default_rng(0)
        masked, targets = mask_stream(tokens, config, rng)

        sep_positions = np.flatnonzero(tokens == SEP)
        starts = [1] + [int(s) + 1 for s in sep_positions[:-1]]
        fully_masked = 0
        untouched = 0
        for i, start in enumerate(starts):
            end = int(sep_positions[i])
            content = range(start, end)
            masked_flags = [targets[p] != IGNORE_INDEX for p in content]
            if all(masked_flags):
                fully_masked += 1
            elif not any(masked_flags):
                untouched += 1
        assert fully_masked == 1
        assert untouched == len(starts) - 1
        assert np.all(masked[np.flatnonzero(tokens == SEP)] == SEP)

    def test_span_mode_masks_contiguous_segments(self):
        from src.dataset.token_stream import SEP

        tokens = make_stream(random_game_factory(4), n_boards=8)
        config = MaskingConfig(
            p_random=0.0, p_board=0.0, p_span=1.0, min_span=2, max_span=3
        )
        rng = np.random.default_rng(0)
        _, targets = mask_stream(tokens, config, rng)

        sep_positions = np.flatnonzero(tokens == SEP)
        starts = [1] + [int(s) + 1 for s in sep_positions[:-1]]
        flags = []
        for i, start in enumerate(starts):
            end = int(sep_positions[i])
            flags.append(all(targets[p] != IGNORE_INDEX for p in range(start, end)))
        assert any(flags)
        flagged = np.flatnonzero(flags)
        assert len(flagged) == flagged[-1] - flagged[0] + 1
        assert 2 <= len(flagged) <= 3

    def test_bert_replacement_distribution(self):
        tokens = make_stream(random_game_factory(5), n_boards=40)
        config = MaskingConfig(p_random=1.0, random_rate=0.5)
        rng = np.random.default_rng(0)
        masked, targets = mask_stream(tokens, config, rng)
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

    def test_deterministic_with_same_seed(self):
        tokens = make_stream(random_game_factory(6))
        config = MaskingConfig()
        a = mask_stream(tokens, config, np.random.default_rng(42))
        b = mask_stream(tokens, config, np.random.default_rng(42))
        assert np.array_equal(a[0], b[0])
        assert np.array_equal(a[1], b[1])


TINY_CONFIG = TransformerConfig(
    d_model=32,
    n_heads=4,
    n_layers=2,
    d_ff=64,
    dropout=0.0,
    max_seq_len=64,
    max_segments=16,
)


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
        model = MaskedStreamTransformer(TINY_CONFIG)
        model.eval()
        batch = tiny_batch()
        output = model(
            batch["tokens"],
            batch["segment_ids"],
            padding_mask=batch["padding_mask"],
            targets=batch["targets"],
        )
        assert output["logits"].shape == (*batch["tokens"].shape, 806)
        assert torch.isfinite(output["loss"])

    def test_padding_does_not_change_real_outputs(self):
        model = MaskedStreamTransformer(TINY_CONFIG)
        model.eval()
        batch = tiny_batch()
        tokens = batch["tokens"]
        with torch.no_grad():
            out_full = model(tokens, batch["segment_ids"], batch["padding_mask"])[
                "logits"
            ]
            for row in range(tokens.shape[0]):
                length = int(batch["padding_mask"][row].sum())
                out_single = model(
                    tokens[row : row + 1, :length],
                    batch["segment_ids"][row : row + 1, :length],
                    batch["padding_mask"][row : row + 1, :length],
                )["logits"]
                assert torch.allclose(out_full[row, :length], out_single[0], atol=1e-5)

    def test_loss_ignores_unmasked_and_pad(self):
        model = MaskedStreamTransformer(TINY_CONFIG)
        model.eval()
        batch = tiny_batch()
        targets = torch.full_like(batch["targets"], IGNORE_INDEX)
        with torch.no_grad():
            loss = model(
                batch["tokens"], batch["segment_ids"], batch["padding_mask"], targets
            )["loss"]
        assert loss.item() == 0.0

    def test_segment_embedding_has_effect(self):
        model = MaskedStreamTransformer(TINY_CONFIG)
        model.eval()
        batch = tiny_batch()
        shifted = batch["segment_ids"] + 1
        with torch.no_grad():
            base = model(batch["tokens"], batch["segment_ids"], batch["padding_mask"])[
                "hidden_states"
            ]
            other = model(batch["tokens"], shifted, batch["padding_mask"])[
                "hidden_states"
            ]
        assert not torch.allclose(base, other)

    def test_embed_pools_segments_correctly(self):
        model = MaskedStreamTransformer(TINY_CONFIG)
        model.eval()
        batch = tiny_batch(n_segments=4, batch=2)
        tokens = batch["tokens"]
        segments = batch["segment_ids"]
        with torch.no_grad():
            hidden = model(tokens, segments, batch["padding_mask"])["hidden_states"]
            pooled, mask = model.embed(tokens, segments, batch["padding_mask"])

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
