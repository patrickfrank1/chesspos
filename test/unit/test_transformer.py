import numpy as np
import pytest
import torch
from torch import nn

from bdd import given, then, when
from src.dataset.token_stream import CLS, EMPTY, N_SLOTS, SEGMENT_LEN, SEP, VOCAB_SIZE
from src.modeling.masking import IGNORE_INDEX, MaskingConfig, mask_stream
from src.modeling.transformer import (
    MaskedStreamTransformer,
    TransformerBlock,
    TransformerConfig,
    slot_indices,
)
from src.modeling.varlen import cu_seqlens, flatten_valid, scatter_valid
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
        from src.dataset.token_stream import EMPTY

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
            assert np.all(tokens[changed] >= EMPTY)
            structural = np.flatnonzero((tokens == 3) | (tokens == 2))
            assert np.all(targets[structural] == IGNORE_INDEX)
            assert np.all(masked[structural] == tokens[structural])
            assert np.all(targets[changed] == tokens[changed])
        return

    def test_random_mode_rate(self):
        from src.dataset.token_stream import EMPTY

        with given("a packed token stream and random-only masking at rate 0.3"):
            tokens = make_stream(random_game_factory(2))
            config = MaskingConfig(p_random=1.0, random_rate=0.3)

        with when("the stream is masked"):
            _, targets, _ = mask_stream(tokens, config, np.random.default_rng(0))

        with then("a fraction of the maskable tokens is masked"):
            n_maskable = int((tokens >= EMPTY).sum())
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
            changed = np.flatnonzero((targets != IGNORE_INDEX) & (targets >= EMPTY))
            assert changed.size > 200
            mask_frac = float((masked[changed] == 1).mean())
            random_frac = float(
                (
                    (masked[changed] >= EMPTY) & (masked[changed] != targets[changed])
                ).mean()
            )
            keep_frac = float((masked[changed] == targets[changed]).mean())
            assert mask_frac == pytest.approx(0.8, abs=0.1)
            assert random_frac == pytest.approx(0.1, abs=0.05)
            assert keep_frac == pytest.approx(0.1, abs=0.05)
        return

    def test_scalar_replacement_stays_in_token_class(self):
        from src.dataset.token_stream import (
            CASTLE_BASE,
            EMPTY,
            EP_BASE,
            TURN_WHITE,
        )

        with given("many masked streams and random-only masking at rate 0.5"):
            config = MaskingConfig(
                p_random=1.0, p_board=0.0, p_span=0.0, random_rate=0.5
            )

        with when("every replaced token is bucketed by its token class"):
            replaced = {
                EMPTY: [],
                TURN_WHITE: [],
                CASTLE_BASE: [],
                EP_BASE: [],
            }
            for seed in range(60):
                tokens = make_stream(random_game_factory(seed), n_boards=8)
                masked, targets, _ = mask_stream(
                    tokens, config, np.random.default_rng(seed)
                )
                for pos in np.flatnonzero(targets != IGNORE_INDEX):
                    low, _ = next(
                        (lo, hi)
                        for lo, hi in (
                            (EMPTY, TURN_WHITE),
                            (TURN_WHITE, CASTLE_BASE),
                            (CASTLE_BASE, EP_BASE),
                            (EP_BASE, VOCAB_SIZE),
                        )
                        if lo <= targets[pos] < hi
                    )
                    if masked[pos] != 1 and masked[pos] != targets[pos]:
                        replaced[low].append(int(masked[pos]))

        with then("each class only receives replacements from its own range"):
            assert replaced[EMPTY], "expected random-replaced slot tokens"
            assert replaced[TURN_WHITE], "expected random-replaced turn tokens"
            assert replaced[CASTLE_BASE], "expected random-replaced castle tokens"
            assert all(EMPTY <= v < TURN_WHITE for v in replaced[EMPTY])
            assert all(TURN_WHITE <= v < CASTLE_BASE for v in replaced[TURN_WHITE])
            assert all(CASTLE_BASE <= v < EP_BASE for v in replaced[CASTLE_BASE])
            assert all(EP_BASE <= v < VOCAB_SIZE for v in replaced[EP_BASE])
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
    max_seq_len=512,
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


def supervised_batch():
    """A tiny batch with real supervised targets at six positions."""
    batch = tiny_batch()
    targets = batch["targets"].clone()
    for row, positions in ((0, (2, 5, 8)), (1, (1, 4, 9))):
        for position in positions:
            targets[row, position] = batch["tokens"][row, position]
    batch["targets"] = targets
    return batch


class TestTransformer:
    def test_forward_with_targets_returns_loss_without_logits(self):
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

        with then("the loss is finite and no full logits tensor is materialized"):
            assert "logits" not in output
            assert torch.isfinite(output["loss"])
            assert output["hidden_states"].shape == (
                *batch["tokens"].shape,
                TINY_CONFIG.d_model,
            )
        return

    def test_forward_without_targets_returns_full_logits(self):
        with given("a tiny masked stream transformer in eval mode and a batch"):
            model = MaskedStreamTransformer(TINY_CONFIG)
            model.eval()
            batch = tiny_batch()

        with when("the model runs a forward pass without targets"):
            output = model(
                batch["tokens"],
                batch["segment_ids"],
                padding_mask=batch["padding_mask"],
            )

        with then("logits span the padded sequence and the vocabulary"):
            assert output["logits"].shape == (*batch["tokens"].shape, VOCAB_SIZE)
            assert torch.isfinite(output["logits"]).all()
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

    def test_head_runs_only_on_supervised_positions(self):
        with given("a tiny model whose head records how many rows it sees"):
            model = MaskedStreamTransformer(TINY_CONFIG)
            model.eval()
            batch = supervised_batch()

            class RecordingHead(nn.Module):
                def __init__(self, inner: nn.Module):
                    super().__init__()
                    self.inner = inner
                    self.row_counts: list[int] = []

                def forward(self, x: torch.Tensor) -> torch.Tensor:
                    self.row_counts.append(x.shape[0])
                    return self.inner(x)

            recording = RecordingHead(model.head)
            model.head = recording

        with when("the model runs a forward pass with targets"):
            with torch.no_grad():
                model(
                    batch["tokens"],
                    batch["segment_ids"],
                    padding_mask=batch["padding_mask"],
                    targets=batch["targets"],
                )

        with then("the head saw exactly the supervised rows in a single call"):
            supervised = int(batch["targets"].ne(IGNORE_INDEX).sum())
            assert recording.row_counts == [supervised]
        return

    def test_weighted_loss_matches_manual_weighted_mean(self):
        with given(
            "a tiny model with a recording head, targets, and per-target weights"
        ):
            model = MaskedStreamTransformer(TINY_CONFIG)
            model.eval()
            batch = supervised_batch()
            weights = (batch["targets"] != IGNORE_INDEX).float() + 1.0

            recorded: dict[str, torch.Tensor] = {}

            class RecordingHead(nn.Module):
                def __init__(self, inner: nn.Module):
                    super().__init__()
                    self.inner = inner

                def forward(self, x: torch.Tensor) -> torch.Tensor:
                    logits = self.inner(x)
                    recorded["logits"] = logits
                    return logits

            model.head = RecordingHead(model.head)

        with when("the weighted loss is computed"):
            with torch.no_grad():
                weighted = model(
                    batch["tokens"],
                    batch["segment_ids"],
                    padding_mask=batch["padding_mask"],
                    targets=batch["targets"],
                    weights=weights,
                )["loss"]

        with then(
            "it equals the manually computed weighted mean over the recorded logits"
        ):
            logits = recorded["logits"]
            targets = batch["targets"][batch["targets"] != IGNORE_INDEX]
            w = weights[batch["targets"] != IGNORE_INDEX]
            per_token = torch.nn.functional.cross_entropy(
                logits, targets, reduction="none"
            )
            expected = (per_token * w).sum() / w.sum()
            assert weighted.item() == pytest.approx(expected.item(), rel=1e-4)
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


def content_positions(tokens, segment_ids, padding_mask):
    """Positions of non-structural, non-pad tokens grouped by segment id."""
    row = tokens[0]
    groups: dict[int, list[int]] = {}
    for position in torch.nonzero(padding_mask[0]).flatten().tolist():
        if int(row[position]) in (CLS, SEP):
            continue
        groups.setdefault(int(segment_ids[0][position]), []).append(position)
    return groups


def swapped_tokens_batch(batch):
    """A copy of the batch with two different content tokens of one segment swapped."""
    tokens = batch["tokens"].clone()
    for positions in content_positions(
        tokens, batch["segment_ids"], batch["padding_mask"]
    ).values():
        for i, left in enumerate(positions):
            for right in positions[i + 1 :]:
                if tokens[0, left] != tokens[0, right]:
                    tokens[0, left], tokens[0, right] = (
                        tokens[0, right],
                        tokens[0, left],
                    )
                    return tokens
    raise AssertionError("no swappable content token pair found")


class TestSlotEmbedding:
    def test_model_has_slot_embedding(self):
        with given("a transformer config"):
            config = TransformerConfig()

        with then("the model embeds one slot per content position plus structure"):
            model = MaskedStreamTransformer(config)
            assert model.slot_embedding.weight.shape == (
                SEGMENT_LEN + 1,
                config.d_model,
            )
        return

    def test_slot_indices_match_segment_layout(self):
        with given("a collated batch and the slot index helper"):
            batch = tiny_batch(n_segments=4, batch=1)

        with when("slot indices are computed for every token"):
            slots = slot_indices(batch["tokens"], batch["segment_ids"])

        with then(
            "structural tokens get the structural index and content tokens are "
            "indexed by their offset within the segment"
        ):
            tokens = batch["tokens"][0]
            segment_ids = batch["segment_ids"][0]
            for position in torch.nonzero(batch["padding_mask"][0]).flatten().tolist():
                token = int(tokens[position])
                if token in (CLS, SEP):
                    assert int(slots[0, position]) == SEGMENT_LEN
                else:
                    group_start = int(
                        torch.nonzero(segment_ids == segment_ids[position])[0]
                    )
                    expected = position - group_start - 1
                    assert int(slots[0, position]) == expected
                    assert 0 <= expected < SEGMENT_LEN
        return

    def test_fully_masked_segment_gets_distinct_per_square_states(self):
        with given("a board-masked stream of 10 boards and a default-config model"):
            tokens = make_stream(random_game_factory(3), n_boards=10)
            masked, targets, weights = mask_stream(
                tokens,
                MaskingConfig(p_random=0.0, p_board=1.0, p_span=0.0),
                np.random.default_rng(0),
            )
            batch = collate_windows(
                [{"tokens": masked, "targets": targets, "weights": weights}]
            )
            model = MaskedStreamTransformer(TINY_CONFIG)
            model.eval()

        with when("the model runs a forward pass over the masked stream"):
            with torch.no_grad():
                hidden = model(
                    batch["tokens"], batch["segment_ids"], batch["padding_mask"]
                )["hidden_states"]

        with then("masked squares within one segment receive distinct hidden states"):
            sep_positions = np.flatnonzero(tokens == SEP)
            flags = segment_flags(targets, sep_positions)
            fully_masked = int(np.flatnonzero(flags)[0])
            start = 1 if fully_masked == 0 else int(sep_positions[fully_masked - 1]) + 1
            square_positions = range(start, start + N_SLOTS)
            states = hidden[0, list(square_positions)]
            assert not torch.allclose(states[0], states[1])
            assert not torch.allclose(states[0], states[-1])
        return

    def test_swapping_tokens_within_a_segment_changes_outputs(self):
        with given("a tiny default-config model and a collated batch"):
            model = MaskedStreamTransformer(TINY_CONFIG)
            model.eval()
            batch = tiny_batch()
            swapped = swapped_tokens_batch(batch)

        with when("the model runs on the original and the swapped batch"):
            with torch.no_grad():
                base = model(
                    batch["tokens"], batch["segment_ids"], batch["padding_mask"]
                )["hidden_states"]
                other = model(swapped, batch["segment_ids"], batch["padding_mask"])[
                    "hidden_states"
                ]

        with then("the hidden states at the swapped positions differ"):
            assert not torch.allclose(base, other)
        return

    def test_slot_indices_stay_in_bounds_for_ragged_padded_batches(self):
        with given("a collated batch with ragged row lengths"):
            batch = tiny_batch(n_segments=4, batch=2)

        with when("slot indices are computed for every position including padding"):
            slots = slot_indices(batch["tokens"], batch["segment_ids"])

        with then("every index is a valid slot embedding row"):
            assert slots.shape == batch["tokens"].shape
            assert int(slots.min()) >= 0
            assert int(slots.max()) <= SEGMENT_LEN
        return


VARLEN_MASK = [
    [1, 1, 1, 1, 0, 0],
    [1, 1, 0, 0, 0, 0],
    [1, 1, 1, 1, 1, 1],
]


class TestVarlenAttention:
    def test_cu_seqlens_from_padding_mask(self):
        with given("a padding mask for a batch of three rows"):
            padding_mask = torch.tensor(VARLEN_MASK, dtype=torch.bool)

        with when("cumulative segment offsets are computed"):
            cu, max_seqlen = cu_seqlens(padding_mask)

        with then("offsets delimit each row's valid tokens in row-major order"):
            assert cu.tolist() == [0, 4, 6, 12]
            assert max_seqlen == 6
            assert cu.dtype == torch.int32
        return

    def test_flatten_valid_gathers_rows_in_order(self):
        with given("a padded hidden tensor and its padding mask"):
            x = torch.arange(18.0).reshape(3, 6, 1)
            padding_mask = torch.tensor(VARLEN_MASK, dtype=torch.bool)

        with when("valid tokens are flattened"):
            flat = flatten_valid(x, padding_mask)

        with then("flat holds only the valid tokens in row-major order"):
            assert flat.shape == (12, 1)
            expected = [
                value
                for row, value_row in enumerate(x.reshape(3, 6).tolist())
                for kept, value in zip(VARLEN_MASK[row], value_row)
                if kept
            ]
            assert flat.ravel().tolist() == expected
        return

    def test_scatter_valid_restores_padded_layout(self):
        with given(
            "a padded hidden tensor, its padding mask, and the flattened valid tokens"
        ):
            x = torch.arange(18.0).reshape(3, 6, 1)
            padding_mask = torch.tensor(VARLEN_MASK, dtype=torch.bool)
            flat = flatten_valid(x, padding_mask)

        with when("the flat tokens are scattered back into a padded tensor"):
            restored = scatter_valid(flat, padding_mask)

        with then("valid positions are restored and pad positions are zeroed"):
            assert restored.shape == x.shape
            assert torch.equal(restored, x * padding_mask.unsqueeze(-1))
        return

    def test_block_attends_only_within_segments(self):
        with given("a tiny transformer block and a flat input of two segments"):
            block = TransformerBlock(TINY_CONFIG)
            block.eval()
            cu = torch.tensor([0, 4, 9], dtype=torch.int32)
            max_seqlen = 5
            x = torch.randn(9, TINY_CONFIG.d_model)

        with when("the block runs forward over the flat segments"):
            with torch.no_grad():
                out = block(x, cu, max_seqlen)

        with then("each segment's output equals running the block on it alone"):
            for start, end in ((0, 4), (4, 9)):
                with torch.no_grad():
                    alone = block(
                        x[start:end],
                        torch.tensor([0, end - start], dtype=torch.int32),
                        end - start,
                    )
                assert torch.allclose(out[start:end], alone, atol=1e-5)
        return

    def test_trunk_runs_blocks_on_flattened_valid_tokens(self):
        with given("a tiny model and a padded batch whose blocks record their inputs"):
            model = MaskedStreamTransformer(TINY_CONFIG)
            model.eval()
            batch = tiny_batch()
            seen = []

            def record(self, x, cu, max_seqlen):
                seen.append((x.dim(), int(cu[-1])))
                return x

            for layer in model.blocks:
                layer.forward = record.__get__(layer)

        with when("the trunk runs forward with a padding mask"):
            with torch.no_grad():
                model._trunk(
                    batch["tokens"], batch["segment_ids"], batch["padding_mask"]
                )

        with then("every block saw a flat tensor covering exactly the valid tokens"):
            assert len(seen) == len(model.blocks)
            valid_total = int(batch["padding_mask"].sum())
            for dim, covered in seen:
                assert dim == 2
                assert covered == valid_total
        return
