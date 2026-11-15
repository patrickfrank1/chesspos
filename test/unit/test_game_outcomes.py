from __future__ import annotations

import chess
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from bdd import given, then, when
from src.dataset.game_outcomes import (
    GameOutcome,
    ScoreDataset,
    ScorePlan,
    build_score_plan,
    collate_score_samples,
    decode_packed,
    flip_segment,
    load_game_records,
    material_balance,
    score_from_result,
    split_for_game,
    state_features,
)
from src.dataset.puzzle_variants import pack_segments
from src.dataset.token_stream import CLS, PAD, SEP, TokenStreamEncoder

SEGMENT_LEN = 67


def make_segment(fill: int) -> np.ndarray:
    return np.full(SEGMENT_LEN, fill, dtype=np.int16)


def make_outcome(
    game_id: str,
    n_positions: int,
    result: str = "1-0",
    offset: int = 10,
) -> GameOutcome:
    segments = [make_segment(offset + index) for index in range(n_positions)]
    return GameOutcome(
        game_id=game_id,
        segments=segments,
        result=result,
        white_elo=2000,
        black_elo=1900,
    )


class TestScoreFromResult:
    def test_maps_white_perspective_scores(self):
        with given("the three lichess result strings"):
            results = ["1-0", "0-1", "1/2-1/2"]

        with when("the scores are computed"):
            scores = [score_from_result(result) for result in results]

        with then("white wins are +1, black wins -1, draws 0"):
            assert scores == [1.0, -1.0, 0.0]
        return

    def test_rejects_missing_or_unknown_results(self):
        with given("empty and malformed result strings"):
            results = ["", "*", "white"]

        with when("the scores are computed"):
            errors = []
            for result in results:
                try:
                    score_from_result(result)
                except ValueError:
                    errors.append(result)

        with then("every unknown result raises"):
            assert errors == results
        return


class TestDecodePacked:
    def test_round_trips_packed_segments(self):
        with given("two packed position segments"):
            segments = [make_segment(10), make_segment(11)]
            packed, _ = pack_segments(segments)

        with when("the stream is decoded"):
            decoded = decode_packed(packed.tolist())

        with then("the segments come back unchanged"):
            assert len(decoded) == 2
            np.testing.assert_array_equal(decoded[0], segments[0])
            np.testing.assert_array_equal(decoded[1], segments[1])
        return

    def test_rejects_an_empty_stream(self):
        with given("an empty token stream"):
            packed: list[int] = []

        with when("it is decoded"):
            error = None
            try:
                decode_packed(packed)
            except ValueError as exc:
                error = exc

        with then("a ValueError is raised"):
            assert isinstance(error, ValueError)
        return


class TestFlipSegment:
    def test_flipping_twice_is_the_identity(self):
        with given("a real position segment"):
            board = chess.Board()
            board.push_san("e4")
            board.push_san("c5")
            board.push_san("Nf3")
            segment = TokenStreamEncoder().encode(board)

        with when("the segment is flipped twice"):
            restored = flip_segment(flip_segment(segment))

        with then("the original segment is recovered"):
            np.testing.assert_array_equal(restored, segment)
        return

    def test_flipped_segment_decodes_to_the_mirrored_board(self):
        with given("positions with castling rights and en passant"):
            boards = []
            with_rights = chess.Board("r3k2r/8/8/8/8/8/8/R3K2R w Kq - 0 1")
            boards.append(with_rights)
            ep_board = chess.Board()
            ep_board.push_san("e4")
            boards.append(ep_board)
            decorated = chess.Board()
            decorated.push_san("e4")
            decorated.push_san("c5")
            decorated.push_san("Nf3")
            boards.append(decorated)
            encoder = TokenStreamEncoder()

        with when("each segment is flipped and decoded"):
            decoded = [
                encoder.decode(flip_segment(encoder.encode(board))) for board in boards
            ]

        with then("the result matches Board.mirror()"):
            for expected, board in zip(decoded, boards):
                assert expected.fen().split()[:4] == board.mirror().fen().split()[:4]
        return


class TestMaterialBalance:
    def test_counts_white_minus_black_piece_values(self):
        with given("an equal position and one with an extra white rook"):
            equal = chess.Board()
            extra_rook = chess.Board("4k3/8/8/8/8/8/8/R3K3 w Q - 0 1")
            encoder = TokenStreamEncoder()

        with when("the material balance is computed from the segments"):
            equal_balance = material_balance(encoder.encode(equal))
            rook_balance = material_balance(encoder.encode(extra_rook))

        with then("the values follow pawn=1, knight/bishop=3, rook=5, queen=9"):
            assert equal_balance == 0
            assert rook_balance == 5
        return

    def test_detects_a_black_advantage(self):
        with given("a position where black has an extra queen"):
            board = chess.Board("4k3/8/8/8/8/8/8/4K2q w - - 0 1")
            encoder = TokenStreamEncoder()

        with when("the material balance is computed"):
            balance = material_balance(encoder.encode(board))

        with then("the balance is negative"):
            assert balance == -9
        return


class TestStateFeatures:
    def test_encodes_turn_castling_and_en_passant(self):
        with given("a position with full castling rights and a legal en passant"):
            board = chess.Board(
                "rnbqkbnr/ppp1pppp/8/3pP3/8/8/PPPP1PPP/RNBQKBNR w KQkq d6 0 3"
            )
            segment = TokenStreamEncoder().encode(board)

        with when("the explicit state features are computed"):
            features = state_features(segment)

        with then("turn, castling mask, and ep presence are exposed"):
            assert features.shape == (3,)
            assert features.dtype == np.float32
            assert features[0] == 1.0
            assert features[1] == 1.0
            assert features[2] == 1.0
        return


class TestBuildScorePlan:
    def test_respects_positions_per_game_and_the_ply_cap(self):
        with given("games with different lengths and a small ply cap"):
            records = [
                make_outcome("a", 10),
                make_outcome("b", 4),
                make_outcome("c", 30),
            ]

        with when("the plan is built"):
            plan = build_score_plan(records, positions_per_game=3, max_ply=8, seed=17)

        with then("three unique plies within the cap are sampled per game"):
            assert len(plan) == 9
            assert plan.ply.min() >= 1
            assert plan.ply.max() <= 8
            for game_index in range(3):
                plies = plan.ply[plan.game_index == game_index]
                assert len(plies) == 3
                assert len(set(plies.tolist())) == 3
        return

    def test_is_deterministic_for_a_seed_and_changes_with_another_seed(self):
        with given("many games and two seeds"):
            records = [make_outcome(f"g{index}", 20) for index in range(50)]

        with when("the plans are built"):
            first = build_score_plan(records, positions_per_game=4, seed=1)
            second = build_score_plan(records, positions_per_game=4, seed=1)
            other = build_score_plan(records, positions_per_game=4, seed=2)

        with then("the same seed reproduces the plan and a new seed differs"):
            np.testing.assert_array_equal(first.game_index, second.game_index)
            np.testing.assert_array_equal(first.ply, second.ply)
            assert not np.array_equal(first.ply, other.ply)
        return

    def test_can_select_every_position(self):
        with given("a game with five positions"):
            records = [make_outcome("a", 5)]

        with when("positions_per_game is None"):
            plan = build_score_plan(records, positions_per_game=None, max_ply=10)

        with then("every ply is in the plan"):
            assert plan.ply.tolist() == [1, 2, 3, 4, 5]
        return


class TestScoreDataset:
    def test_returns_one_position_ending_at_the_sampled_ply(self):
        with given("one game and a plan selecting ply 2"):
            records = [make_outcome("g", 4, result="0-1")]
            plan = ScorePlan(
                game_index=np.array([0], dtype=np.int32),
                ply=np.array([2], dtype=np.int16),
            )
            dataset = ScoreDataset(records, plan)

        with when("the item is read"):
            item = dataset[0]

        with then("the input is CLS + one segment + SEP with the game score"):
            assert item["tokens"].shape == (SEGMENT_LEN + 2,)
            assert item["tokens"][0] == CLS
            assert item["tokens"][-1] == SEP
            assert item["tokens"][1] == 11
            assert item["padding_mask"].all()
            assert item["target"] == -1.0
            assert item["ply"] == 2
            assert item["game_id"] == "g"
            assert "features" not in item
        return

    def test_window_uses_the_preceding_positions(self):
        with given("a dataset with a three-position window"):
            records = [make_outcome("g", 5)]
            plan = ScorePlan(
                game_index=np.array([0], dtype=np.int32),
                ply=np.array([4], dtype=np.int16),
            )
            dataset = ScoreDataset(records, plan, window=3)

        with when("the item is read"):
            item = dataset[0]

        with then("the input is the last three positions ending at the ply"):
            assert item["tokens"].shape == (3 * (SEGMENT_LEN + 1) + 1,)
            assert item["tokens"][1] == 11
            assert item["tokens"][SEGMENT_LEN + 2] == 12
            assert item["tokens"][2 * (SEGMENT_LEN + 1) + 1] == 13
            assert item["padding_mask"].all()
        return

    def test_exposes_explicit_state_features_when_requested(self):
        with given("a dataset built with state features enabled"):
            records = [make_outcome("g", 2)]
            plan = ScorePlan(
                game_index=np.array([0], dtype=np.int32),
                ply=np.array([1], dtype=np.int16),
            )
            dataset = ScoreDataset(records, plan, include_state_features=True)

        with when("the item is read"):
            item = dataset[0]

        with then("a three-float state vector is attached"):
            assert item["features"].shape == (3,)
            assert item["features"].dtype == np.float32
        return

    def test_rejects_an_invalid_window(self):
        with given("a window of zero"):
            records = [make_outcome("g", 2)]
            plan = ScorePlan(
                game_index=np.array([0], dtype=np.int32),
                ply=np.array([1], dtype=np.int16),
            )

        with when("the dataset is constructed"):
            error = None
            try:
                ScoreDataset(records, plan, window=0)
            except ValueError as exc:
                error = exc

        with then("a ValueError is raised"):
            assert isinstance(error, ValueError)
        return


class TestCollateSamples:
    def test_pads_tokens_and_stacks_targets(self):
        with given("samples with one and two segments"):
            batch = [
                {
                    "tokens": np.array([CLS, 1, 1, 1, SEP], dtype=np.int16),
                    "segment_ids": np.array([0, 0, 0, 0, 1], dtype=np.int64),
                    "padding_mask": np.ones(5, dtype=bool),
                    "target": 1.0,
                    "ply": 1,
                    "game_id": "a",
                },
                {
                    "tokens": np.array(
                        [CLS, 2, 2, 2, SEP, 3, 3, 3, SEP], dtype=np.int16
                    ),
                    "segment_ids": np.array(
                        [0, 0, 0, 0, 1, 1, 1, 1, 2], dtype=np.int64
                    ),
                    "padding_mask": np.ones(9, dtype=bool),
                    "target": -1.0,
                    "ply": 2,
                    "game_id": "b",
                },
            ]

        with when("the samples are collated"):
            collated = collate_score_samples(batch)

        with then("padded tensors and float targets are returned"):
            assert collated["tokens"].shape == (2, 9)
            assert collated["targets"].dtype.is_floating_point
            assert collated["targets"].tolist() == [1.0, -1.0]
            assert collated["plies"].tolist() == [1, 2]
            assert collated["game_ids"] == ["a", "b"]
            assert collated["tokens"][0, 5].item() == PAD
            assert collated["padding_mask"].tolist() == [
                [True, True, True, True, True, False, False, False, False],
                [True, True, True, True, True, True, True, True, True],
            ]
        return

    def test_collates_optional_state_features(self):
        with given("two samples carrying state features"):
            base = {
                "tokens": np.array([CLS, 1, 1, 1, SEP], dtype=np.int16),
                "segment_ids": np.array([0, 0, 0, 0, 1], dtype=np.int64),
                "padding_mask": np.ones(5, dtype=bool),
                "ply": 1,
            }
            batch = [
                {
                    **base,
                    "target": 1.0,
                    "game_id": "a",
                    "features": np.array([1.0, 1.0, 0.0], dtype=np.float32),
                },
                {
                    **base,
                    "target": 0.0,
                    "game_id": "b",
                    "features": np.array([0.0, 0.5, 1.0], dtype=np.float32),
                },
            ]

        with when("the samples are collated"):
            collated = collate_score_samples(batch)

        with then("a (batch, features) float tensor is returned"):
            assert collated["features"].shape == (2, 3)
            assert collated["features"].dtype.is_floating_point
        return


class TestLoadGameRecords:
    def test_loads_results_and_elo_and_drops_unknown_results(self, tmp_path):
        with given("a parquet shard with two valid results and one missing"):
            encoder = TokenStreamEncoder()
            board = chess.Board()
            board.push_san("e4")
            packed = pack_segments([encoder.encode(board)])[0].tolist()
            table = pa.table(
                {
                    "packed": pa.array([packed, packed, packed], pa.list_(pa.int16())),
                    "n_positions": pa.array([1, 1, 1], pa.int32()),
                    "game_id": pa.array(["a", "b", "c"], pa.string()),
                    "white_elo": pa.array([2000, 2100, 2200], pa.int32()),
                    "black_elo": pa.array([1900, 2050, 2150], pa.int32()),
                    "result": pa.array(["1-0", "1/2-1/2", ""], pa.string()),
                }
            )
            pq.write_table(table, tmp_path / "part.parquet")

        with when("the records are loaded"):
            records = load_game_records(tmp_path)

        with then("only rows with a known result are kept"):
            assert [record.game_id for record in records] == ["a", "b"]
            assert records[0].score == 1.0
            assert records[1].score == 0.0
            assert records[0].white_elo == 2000
            assert len(records[0].segments) == 1
        return

    def test_limit_stops_after_the_requested_number(self, tmp_path):
        with given("a shard with three games"):
            segment = make_segment(7)
            packed, _ = pack_segments([segment])
            table = pa.table(
                {
                    "packed": pa.array([packed.tolist()] * 3, pa.list_(pa.int16())),
                    "n_positions": pa.array([1, 1, 1], pa.int32()),
                    "game_id": pa.array(["a", "b", "c"], pa.string()),
                    "white_elo": pa.array([0, 0, 0], pa.int32()),
                    "black_elo": pa.array([0, 0, 0], pa.int32()),
                    "result": pa.array(["1-0", "0-1", "1/2-1/2"], pa.string()),
                }
            )
            pq.write_table(table, tmp_path / "part.parquet")

        with when("the loader is limited to two games"):
            records = load_game_records(tmp_path, limit=2)

        with then("only two games are returned"):
            assert len(records) == 2
            assert [record.game_id for record in records] == ["a", "b"]
        return


class TestSplitForGame:
    def test_split_is_deterministic_and_disjoint(self):
        with given("many game ids"):
            game_ids = [f"game-{index}" for index in range(20_000)]

        with when("train/val splits are computed"):
            splits = [split_for_game(game_id, val_ratio=0.1) for game_id in game_ids]
            repeated = [split_for_game(game_id, val_ratio=0.1) for game_id in game_ids]

        with then("the ratio is close to the target and the result is stable"):
            assert splits == repeated
            ratio = splits.count("val") / len(splits)
            assert 0.08 < ratio < 0.12
            assert set(splits) <= {"train", "val"}
        return

    def test_rejects_an_invalid_val_ratio(self):
        with given("a validation ratio outside (0, 1)"):
            ratio = 1.5

        with when("the split is computed"):
            error = None
            try:
                split_for_game("abc", val_ratio=ratio)
            except ValueError as exc:
                error = exc

        with then("a ValueError is raised"):
            assert isinstance(error, ValueError)
        return
