from __future__ import annotations

import chess
import numpy as np
import pytest
import pyarrow.parquet as pq
import torch
import torch.nn.functional as F

from bdd import given, then, when
from src.dataset.token_stream import (
    EMPTY,
    N_SLOTS,
    SEGMENT_LEN,
    SEP,
    TURN_BLACK,
    TURN_WHITE,
    VOCAB_SIZE,
    TokenStreamEncoder,
    pack_stream,
)
from src.modeling.transformer import (
    MaskedStreamTransformer,
    TransformerConfig,
    slot_indices,
)
from src.training.aux_heads import (
    AuxiliaryHeads,
    auxiliary_head_losses,
    segment_global_targets,
)
from src.training.contrastive import build_contrastive_masks, supcon_loss
from src.training.hard_negatives import (
    collect_hard_negatives,
    write_hard_negatives,
)
from src.training.objectives import (
    AuxiliaryLossConfig,
    extract_alone_views_multi,
    flip_turn_tokens,
)
from src.training.window_dataset import (
    WindowDataset,
    WindowDatasetConfig,
    collate_windows,
)


def tiny_model(*, aux_heads: bool = False) -> MaskedStreamTransformer:
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
            aux_heads=aux_heads,
        )
    )


def packed_segments(boards: list[chess.Board]):
    encoder = TokenStreamEncoder()
    segments = [encoder.encode(board) for board in boards]
    tokens_np = pack_stream(segments)
    tokens = torch.from_numpy(tokens_np.astype(np.int64))[None]
    segment_ids = torch.from_numpy(np.cumsum(tokens_np == SEP).astype(np.int64))[None]
    padding = torch.ones_like(tokens, dtype=torch.bool)
    return tokens, segment_ids, padding


def three_boards() -> list[chess.Board]:
    board = chess.Board()
    boards = []
    for _ in range(3):
        boards.append(board.copy())
        board.push(list(board.legal_moves)[0])
    return boards


class TestContentStatePooling:
    def test_content_pool_uses_only_square_tokens(self):
        with given("a packed board with arbitrary hidden states"):
            tokens, segment_ids, padding = packed_segments([chess.Board()])
            hidden = torch.randn(1, tokens.shape[1], 8)
            model = tiny_model()

        with when("the content and state pools are computed"):
            content, content_mask = model.pool_content(
                hidden, tokens, segment_ids, padding
            )
            state, state_mask = model.pool_state(hidden, tokens, segment_ids, padding)

        with then("content averages squares and state averages turn/castle/ep"):
            content_positions = (tokens[0] >= EMPTY) & (tokens[0] < TURN_WHITE)
            state_positions = (tokens[0] >= TURN_WHITE) & (tokens[0] < VOCAB_SIZE)
            expected_content = hidden[0][content_positions].mean(dim=0)
            expected_state = hidden[0][state_positions].mean(dim=0)
            assert torch.allclose(content[0], expected_content, atol=1e-6)
            assert torch.allclose(state[0], expected_state, atol=1e-6)
            assert content_mask[0, 0]
            assert state_mask[0, 0]
        return

    def test_padded_view_encoding_matches_independent_varlen_views(self):
        with given("three alone views packed into one padded batch"):
            tokens, segment_ids, padding = packed_segments(three_boards())
            model = tiny_model().eval()
            alone = extract_alone_views_multi(tokens, segment_ids, padding, [[0, 1, 2]])
            alone_tokens, alone_segments, alone_padding, _, _ = alone

        with when("the views are encoded together and individually"):
            batched_content, batched_state, _ = model.encode_views(
                alone_tokens, alone_segments, alone_padding
            )
            individual = []
            for index in range(alone_tokens.shape[0]):
                content, _ = model.encode_content(
                    alone_tokens[index : index + 1],
                    alone_segments[index : index + 1],
                    alone_padding[index : index + 1],
                )
                state, _ = model.encode_state(
                    alone_tokens[index : index + 1],
                    alone_segments[index : index + 1],
                    alone_padding[index : index + 1],
                )
                individual.append((content[0, 0], state[0, 0]))

        with then("each padded view matches its independent encoding"):
            for index, (content, state) in enumerate(individual):
                assert torch.allclose(batched_content[index], content, atol=1e-5)
                assert torch.allclose(batched_state[index], state, atol=1e-5)
        return

    def test_content_pool_is_insensitive_to_the_turn_hidden_state(self):
        with given("a packed board and hidden states"):
            tokens, segment_ids, padding = packed_segments([chess.Board()])
            hidden = torch.randn(1, tokens.shape[1], 8)
            model = tiny_model()

        with when("only the turn token's hidden state is changed"):
            changed = hidden.clone()
            turn_positions = (tokens[0] == TURN_WHITE) | (tokens[0] == TURN_BLACK)
            changed[0, turn_positions] += 100.0
            content, _ = model.pool_content(hidden, tokens, segment_ids, padding)
            changed_content, _ = model.pool_content(
                changed, tokens, segment_ids, padding
            )
            state, _ = model.pool_state(hidden, tokens, segment_ids, padding)
            changed_state, _ = model.pool_state(changed, tokens, segment_ids, padding)

        with then("content is unchanged while state changes"):
            assert torch.allclose(content, changed_content, atol=1e-6)
            assert not torch.allclose(state, changed_state)
        return


class TestTurnFlip:
    def test_flips_every_segment_turn_token_and_nothing_else(self):
        with given("three packed boards with alternating turns"):
            tokens, segment_ids, _ = packed_segments(three_boards())
            slots = slot_indices(tokens, segment_ids)
            turn_positions = (slots == N_SLOTS) & (
                (tokens == TURN_WHITE) | (tokens == TURN_BLACK)
            )

        with when("the turn tokens are flipped"):
            flipped = flip_turn_tokens(tokens, segment_ids)

        with then("only the turn tokens change and flipping twice restores them"):
            expected = torch.where(tokens == TURN_WHITE, TURN_BLACK, TURN_WHITE)
            assert torch.equal(flipped[turn_positions], expected[turn_positions])
            assert torch.equal(flipped[~turn_positions], tokens[~turn_positions])
            assert torch.equal(flip_turn_tokens(flipped, segment_ids), tokens)
        return


class TestSupCon:
    def test_multi_positive_loss_matches_the_definition(self):
        with given("five embeddings, one anchor, two positives, one ignored"):
            embeddings = torch.tensor(
                [
                    [1.0, 0.0],
                    [0.8, 0.2],
                    [0.2, 0.8],
                    [0.0, 1.0],
                    [0.6, 0.4],
                ]
            )
            positive = torch.zeros(5, 5, dtype=torch.bool)
            positive[0, 1] = True
            positive[0, 2] = True
            ignore = torch.zeros(5, 5, dtype=torch.bool)
            ignore[0, 3] = True

        with when("the SupCon loss is computed"):
            loss = supcon_loss(
                embeddings, positive, ignore_mask=ignore, temperature=0.1
            )

        with then("it equals the averaged log-probability of the positives"):
            z = F.normalize(embeddings, dim=-1)
            logits = z @ z.T / 0.1
            numerator = logits[0, [1, 2]].mean()
            denominator = torch.logsumexp(logits[0, [1, 2, 4]], dim=0)
            expected = -(numerator - denominator)
            assert loss.item() == pytest.approx(expected.item(), abs=1e-5)
        return

    def test_ignored_pairs_do_not_enter_the_denominator(self):
        with given("an ignored pair whose embedding is changed"):
            embeddings = torch.tensor([[1.0, 0.0], [0.9, 0.1], [0.0, 1.0], [0.2, 0.8]])
            positive = torch.zeros(4, 4, dtype=torch.bool)
            positive[0, 1] = True
            ignore = torch.zeros(4, 4, dtype=torch.bool)
            ignore[0, 2] = True

        with when("the loss is computed before and after changing the ignored row"):
            first = supcon_loss(embeddings, positive, ignore_mask=ignore)
            changed = embeddings.clone()
            changed[2] = torch.tensor([-1.0, 0.0])
            second = supcon_loss(changed, positive, ignore_mask=ignore)

        with then("the ignored row has no effect"):
            assert first.item() == pytest.approx(second.item(), abs=1e-6)
        return


class TestContrastiveMasks:
    def test_same_game_is_positive_within_radius_and_ignored_beyond_it(self):
        with given("five views from two games with known ply distances"):
            game_ids = ["a", "a", "a", "b", "b"]
            plies = [10, 12, 20, 10, 12]

        with when("the contrastive masks are built"):
            positive, ignore = build_contrastive_masks(
                game_ids, plies, temporal_radius=3
            )

        with then("near neighbours are positive and far same-game pairs ignored"):
            assert positive[0, 1]
            assert positive[3, 4]
            assert ignore[0, 2]
            assert ignore[1, 2]
            assert not positive[0, 2]
            assert not positive[0, 3]
            assert not ignore[0, 3]
            assert not positive[0, 4]
            assert not ignore[0, 4]
            assert not positive[0, 0]
            assert not ignore[0, 0]
        return

    def test_opening_positions_are_excluded_from_negative_mining(self):
        with given("two opening views and two non-opening views from other games"):
            game_ids = ["a", "a", "b", "b", "c"]
            plies = [10, 12, 10, 25, 25]
            opening = [True, True, True, False, False]

        with when("the contrastive masks are built with the opening heuristic"):
            positive, ignore = build_contrastive_masks(
                game_ids,
                plies,
                temporal_radius=3,
                opening=opening,
            )

        with then("openings stay positive within a game but never become negatives"):
            assert positive[0, 1]
            assert ignore[0, 2]
            assert ignore[2, 3]
            assert not ignore[3, 4]
            assert not positive[3, 4]
        return


class TestGlobalTargets:
    def test_start_position_targets(self):
        with given("the start position"):
            segment = TokenStreamEncoder().encode(chess.Board())

        with when("the global targets are computed"):
            targets = segment_global_targets(torch.from_numpy(segment)[None])

        with then("material, pieces, phase, turn, castle and ep are correct"):
            assert targets["material"].item() == pytest.approx(0.0)
            assert targets["piece_count"].item() == pytest.approx(32.0)
            assert targets["phase"].item() == 0
            assert targets["turn"].item() == 0
            assert targets["castle"].item() == 15
            assert targets["ep"].item() == 0
        return

    def test_material_and_endgame_targets(self):
        with given("a position missing one black pawn and a bare-kings endgame"):
            missing_pawn = chess.Board()
            missing_pawn.remove_piece_at(chess.E7)
            endgame = chess.Board("8/8/8/8/8/8/4K3/6k1 w - - 0 1")
            encoder = TokenStreamEncoder()
            segments = torch.from_numpy(
                np.stack([encoder.encode(missing_pawn), encoder.encode(endgame)])
            )

        with when("the global targets are computed"):
            targets = segment_global_targets(segments)

        with then("material is +1 and the endgame is classified as phase 2"):
            assert targets["material"].tolist() == pytest.approx([1.0, 0.0])
            assert targets["piece_count"].tolist() == pytest.approx([31.0, 2.0])
            assert targets["phase"].tolist() == [0, 2]
        return

    def test_en_passant_target_is_recovered(self):
        with given("a position with a legal en-passant capture on d6"):
            board = chess.Board(
                "rnbqkbnr/ppp1pppp/8/3pP3/8/8/PPPP1PPP/RNBQKBNR w KQkq d6 0 3"
            )
            segment = TokenStreamEncoder().encode(board)

        with when("the global targets are computed"):
            targets = segment_global_targets(torch.from_numpy(segment)[None])

        with then("the en-passant slot is non-zero"):
            assert targets["ep"].item() == 12
        return


class TestAuxiliaryHeads:
    def test_all_heads_produce_losses_and_gradients(self):
        with given("content/state embeddings and targets for four positions"):
            torch.manual_seed(0)
            content = torch.randn(4, 8, requires_grad=True)
            state = torch.randn(4, 8, requires_grad=True)
            heads = AuxiliaryHeads(d_model=8)
            targets = {
                "material": torch.tensor([0.0, 1.0, -1.0, 0.0]),
                "piece_count": torch.tensor([32.0, 31.0, 30.0, 20.0]),
                "phase": torch.tensor([0, 0, 1, 2]),
                "turn": torch.tensor([0, 1, 0, 1]),
                "castle": torch.tensor([15, 15, 0, 0]),
                "ep": torch.tensor([0, 12, 0, 0]),
            }
            weights = {name: 0.05 for name in targets}

        with when("the auxiliary head losses are computed and backpropagated"):
            total, components = auxiliary_head_losses(
                content, state, targets, heads, weights
            )
            total.backward()

        with then("every head contributes a finite loss and receives gradients"):
            assert set(components) == {
                "aux_material",
                "aux_piece_count",
                "aux_phase",
                "aux_turn",
                "aux_castle",
                "aux_ep",
            }
            assert torch.isfinite(total)
            assert content.grad is not None
            assert state.grad is not None
        return


class TestHardNegatives:
    def test_collects_and_writes_hard_negatives(self, tmp_path):
        with given("two non-opening windows from different games"):
            tokens_a, _, _ = packed_segments(three_boards())
            tokens_b, _, _ = packed_segments(three_boards())
            samples = []
            for game_id, tokens in enumerate((tokens_a, tokens_b)):
                tokens_np = tokens[0].numpy()
                samples.append(
                    {
                        "tokens": tokens_np,
                        "targets": np.full(tokens_np.shape, -100, dtype=np.int16),
                        "weights": np.zeros(tokens_np.shape, dtype=np.float32),
                        "game_id": game_id,
                        "start_index": 25,
                    }
                )
            batch = collate_windows(samples)
            model = tiny_model()
            config = AuxiliaryLossConfig(
                content_pooling=True,
                use_supcon=True,
                ignore_opening_negatives=True,
                opening_ply_threshold=20,
                alone_targets_per_window=2,
                nce_temperature=0.07,
                nce_weight=0.2,
            )

        with when("the hard negatives are collected and written"):
            records = collect_hard_negatives(
                model,
                batch,
                config,
                step=7,
                top_k=4,
                generator=torch.Generator().manual_seed(0),
            )
            path = write_hard_negatives(records, tmp_path, step=7)

        with then("the Parquet shard contains the ranked hard-negative rows"):
            assert records
            assert path is not None
            table = pq.read_table(path)
            assert table.num_rows == len(records)
            assert "anchor_fen" in table.column_names
            assert "negative_fen" in table.column_names
            assert all(not record["anchor_opening"] for record in records)
            assert all(not record["negative_opening"] for record in records)
            assert [record["rank"] for record in records] == list(
                range(1, len(records) + 1)
            )
        return


class TestWindowMetadata:
    def test_window_carries_game_id_and_start_index(self):
        with given("a long game and a window dataset"):
            board = chess.Board()
            encoder = TokenStreamEncoder()
            segments = []
            for _ in range(100):
                moves = list(board.legal_moves)
                board.push(moves[0])
                if board.is_game_over():
                    break
                segments.append(encoder.encode(board))
            dataset = WindowDataset(
                WindowDatasetConfig(
                    games=[segments],
                    seed=0,
                    max_seq_len=512,
                    max_segments=16,
                    p_single=0.0,
                    p_few=0.0,
                    balanced_modes=False,
                    apply_masking=False,
                )
            )

        with when("a window is sampled"):
            sample = dataset[0]

        with then("the game id and start index identify the source segment"):
            assert sample["game_id"] == 0
            assert 0 <= sample["start_index"] < len(segments)
            first_tokens = sample["tokens"][1 : 1 + SEGMENT_LEN]
            assert first_tokens.tolist() == segments[sample["start_index"]].tolist()
        return

    def test_collate_preserves_window_metadata(self):
        with given("two identical windows with metadata"):
            board = chess.Board()
            encoder = TokenStreamEncoder()
            segments = []
            for _ in range(20):
                board.push(list(board.legal_moves)[0])
                segments.append(encoder.encode(board))
            dataset = WindowDataset(
                WindowDatasetConfig(
                    games=[segments],
                    seed=0,
                    max_seq_len=512,
                    max_segments=16,
                    p_single=0.0,
                    p_few=0.0,
                    balanced_modes=False,
                    apply_masking=False,
                )
            )
            sample = dataset[0]

        with when("the windows are collated"):
            batch = collate_windows([sample, sample])

        with then("game ids and start indices become batch tensors"):
            assert batch["game_id"].tolist() == [0, 0]
            assert batch["start_index"].shape == (2,)
        return


class TestTrainingIntegration:
    def test_supcon_and_auxiliary_heads_are_used(self):
        with given("a tiny model with heads and a batch with window metadata"):
            model = tiny_model(aux_heads=True)
            tokens, segment_ids, padding = packed_segments(three_boards())
            targets = torch.full_like(tokens, -100)
            weights = torch.zeros_like(tokens, dtype=torch.float)
            batch = {
                "tokens": tokens,
                "segment_ids": segment_ids,
                "padding_mask": padding,
                "targets": targets,
                "weights": weights,
                "game_id": torch.tensor([0]),
                "start_index": torch.tensor([0]),
            }
            config = AuxiliaryLossConfig(
                mlm_weight=1.0,
                inv_weight=0.0,
                var_weight=0.0,
                cov_weight=0.0,
                nce_weight=0.2,
                alone_var_weight=0.0,
                alone_cov_weight=0.0,
                warmup_steps=0,
                use_supcon=True,
                turn_flip=True,
                temporal_radius=3,
                material_weight=0.05,
                piece_count_weight=0.05,
                phase_weight=0.05,
                turn_weight=0.05,
                castle_weight=0.05,
                ep_weight=0.05,
            )

        with when("the training loss is computed and backpropagated"):
            from src.training.objectives import compute_training_loss

            total, components = compute_training_loss(model, batch, config)
            total.backward()

        with then("SupCon and all auxiliary heads contribute"):
            assert {"mlm", "supcon", "aux_material", "aux_piece_count"}.issubset(
                set(components)
            )
            assert "aux_phase" in components
            assert "aux_turn" in components
            assert torch.isfinite(total)
        return


class TestTurnStateIsolation:
    def test_content_embedding_is_invariant_to_the_turn_token(self):
        with given("a board encoded with white and black to move"):
            tokens, segment_ids, padding = packed_segments([chess.Board()])
            flipped = flip_turn_tokens(tokens, segment_ids)
            model = tiny_model().eval()

        with when("the content pool is encoded for both turn states"):
            with torch.no_grad():
                content, _ = model.encode_content(tokens, segment_ids, padding)
                flipped_content, _ = model.encode_content(flipped, segment_ids, padding)

        with then("the content embedding is unchanged"):
            assert torch.allclose(content, flipped_content, atol=1e-6)
        return

    def test_content_views_ignore_state_but_state_views_do_not(self):
        with given("a board encoded with white and black to move"):
            tokens, segment_ids, padding = packed_segments([chess.Board()])
            flipped = flip_turn_tokens(tokens, segment_ids)
            model = tiny_model().eval()

        with when("the combined content/state views are encoded"):
            with torch.no_grad():
                content, state, _ = model.encode_views(tokens, segment_ids, padding)
                flipped_content, flipped_state, _ = model.encode_views(
                    flipped, segment_ids, padding
                )

        with then("content is invariant while state still changes"):
            assert torch.allclose(content, flipped_content, atol=1e-6)
            assert not torch.allclose(state, flipped_state, atol=1e-6)
        return

    def test_content_embedding_still_responds_to_board_content(self):
        with given("a board and a copy with one square changed"):
            tokens, segment_ids, padding = packed_segments([chess.Board()])
            changed = tokens.clone()
            changed[0, 1] = EMPTY if changed[0, 1] != EMPTY else EMPTY + 1
            model = tiny_model().eval()

        with when("the content pool is encoded for both boards"):
            with torch.no_grad():
                content, _ = model.encode_content(tokens, segment_ids, padding)
                changed_content, _ = model.encode_content(changed, segment_ids, padding)

        with then("the content embedding changes"):
            assert not torch.allclose(content, changed_content, atol=1e-6)
        return
