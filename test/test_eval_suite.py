import numpy as np
import torch

from src.dataset.token_stream import TokenStreamEncoder
from src.evaluation.eval_suite import full_eval
from src.modeling.transformer import MaskedStreamTransformer, TransformerConfig


def build_games(n_segments=20, seed=0):
    import chess

    rng = np.random.default_rng(seed)
    encoder = TokenStreamEncoder()
    board = chess.Board()
    segments = []
    for _ in range(n_segments):
        segments.append(encoder.encode(board))
        moves = list(board.legal_moves)
        board.push(moves[int(rng.integers(len(moves)))])
    return [segments]


TINY_CONFIG = TransformerConfig(
    d_model=32,
    n_heads=4,
    n_layers=2,
    d_ff=64,
    dropout=0.0,
    max_seq_len=64,
    max_segments=16,
)


class TestFullEval:
    def test_returns_expected_metrics(self):
        model = MaskedStreamTransformer(TINY_CONFIG)
        model.eval()
        games = build_games()
        metrics = full_eval(
            model,
            games,
            torch.device("cpu"),
            windows=2,
            batch_size=2,
            max_seq_len=64,
            max_segments=16,
            pool_size=16,
            counterfactual_n=8,
            seed=0,
        )
        expected = [
            "random_acc",
            "board_turn_acc",
            "board_castle_acc",
            "board_piece_acc",
            "board_prior_baseline",
            "span_acc",
            "mean_top10_cosine",
            "cos_to_move_child",
            "cos_to_corrupted",
            "test_r2_material",
            "test_acc_side_to_move",
            "pc1_piece_corr",
        ]
        for key in expected:
            assert key in metrics
        assert all(np.isfinite(value) for value in metrics.values())

    def test_restores_training_mode(self):
        model = MaskedStreamTransformer(TINY_CONFIG)
        model.train()
        games = build_games()
        full_eval(
            model,
            games,
            torch.device("cpu"),
            windows=1,
            batch_size=1,
            max_seq_len=64,
            max_segments=16,
            pool_size=8,
            counterfactual_n=0,
            seed=0,
        )
        assert model.training
