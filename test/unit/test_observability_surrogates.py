import chess
import numpy as np

from bdd import given, then, when
from src.evaluation.observability.samples import (
    SURROGATE_KEYS,
    board_surrogates,
    make_sample,
)
from src.evaluation.observability.surrogates import (
    SURROGATE_TASKS,
    classify_surrogates,
    matching_matrix,
    surrogate_class_labels,
)
from src.dataset.token_stream import TokenStreamEncoder


def test_start_position_surrogate_classes():
    with given("the starting position"):
        surrogates = board_surrogates(chess.Board())

    with when("semantic surrogate classes are derived"):
        classes = classify_surrogates(surrogates)

    with then("the classes match the known starting structure"):
        assert classes["material"] == "equal"
        assert classes["phase"] == "opening"
        assert classes["open_files"] == "none"
        assert classes["isolated_pawns"] == "none"
        assert classes["passed_pawns"] == "none"
        assert classes["doubled_pawns"] == "none"
        assert classes["pawn_islands"] == "low"
        assert classes["king_safety"] == "mid"
        assert classes["mobility"] == "low"
        assert classes["bishop_pair"] == "yes"
        assert classes["in_check"] == "no"
    return


def test_open_file_position_is_classified():
    with given("a position with an open a-file and no bishops"):
        board = chess.Board("4k3/8/8/8/8/8/4P3/4K3 w - - 0 1")
        surrogates = board_surrogates(board)

    with when("semantic surrogate classes are derived"):
        classes = classify_surrogates(surrogates)

    with then("the open file and bishop classes are correct"):
        assert classes["open_files"] == "some"
        assert classes["bishop_pair"] == "none"
    return


def test_material_advantage_classification():
    with given("positions with white and black material advantages"):
        white = chess.Board("4k3/8/8/8/8/8/8/3QK3 w - - 0 1")
        black = chess.Board("3qk3/8/8/8/8/8/8/4K3 w - - 0 1")

    with when("classes are derived"):
        white_classes = classify_surrogates(board_surrogates(white))
        black_classes = classify_surrogates(board_surrogates(black))

    with then("the material classes reflect the advantage"):
        assert white_classes["material"] == "white_advantage"
        assert black_classes["material"] == "black_advantage"
    return


def test_surrogate_class_labels_use_samples():
    with given("samples with different material balances"):
        encoder = TokenStreamEncoder()
        samples = [
            make_sample(
                encoder.encode(chess.Board()),
                "test",
                sample_id=0,
            ),
            make_sample(
                encoder.encode(chess.Board("4k3/8/8/8/8/8/8/3QK3 w - - 0 1")),
                "test",
                sample_id=1,
            ),
        ]

    with when("labels for the material task are extracted"):
        task = next(task for task in SURROGATE_TASKS if task.name == "material")
        labels = surrogate_class_labels(samples, task)

    with then("the labels are the semantic class names"):
        assert labels.tolist() == ["equal", "white_advantage"]
    return


def test_leave_one_out_matching_drops_the_target_column():
    with given("two samples and the material surrogate"):
        encoder = TokenStreamEncoder()
        samples = [
            make_sample(encoder.encode(chess.Board()), "test", sample_id=0),
            make_sample(
                encoder.encode(chess.Board("4k3/8/8/8/8/8/8/3QK3 w - - 0 1")),
                "test",
                sample_id=1,
            ),
        ]

    with when("the matching matrix excludes material"):
        matrix = matching_matrix(samples, exclude_key="material")

    with then("one surrogate column is removed"):
        assert matrix.shape == (2, len(SURROGATE_KEYS) - 1)
        assert np.isfinite(matrix).all()
    return


def test_edge_knight_and_inactive_bishop_classes():
    with given("a position with a knight on the a-file and blocked bishops"):
        edge = chess.Board("4k3/8/8/8/N7/8/8/4K3 w - - 0 1")
        start = chess.Board()

    with when("classes are derived"):
        edge_classes = classify_surrogates(board_surrogates(edge))
        start_classes = classify_surrogates(board_surrogates(start))

    with then("the new structural classes are correct"):
        assert edge_classes["knight_on_edge"] == "some"
        assert start_classes["knight_on_edge"] == "none"
        assert start_classes["inactive_bishop"] == "some"
    return


def test_king_attack_pressure_class():
    with given("a queen attacking the black king's escape squares"):
        board = chess.Board("4k3/4Q3/8/8/8/8/8/4K3 w - - 0 1")

    with when("the king-attack class is derived"):
        classes = classify_surrogates(board_surrogates(board))

    with then("the position is classified as a high king attack"):
        assert classes["king_attack"] == "high"
    return


def test_smothered_position_class():
    with given("a smothered-mate construction"):
        board = chess.Board("6rk/5Npp/8/8/8/8/8/6K1 b - - 0 1")

    with when("the smothered class is derived"):
        classes = classify_surrogates(board_surrogates(board))

    with then("the position is classified as smothered"):
        assert classes["smothered"] == "yes"
    return
