import numpy as np
import torch

from bdd import given, then, when
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
    max_seq_len=512,
    max_segments=16,
)


class TestEvalAutocast:
    """Eval forwards must run under bf16 autocast: flash-attn rejects fp32
    inputs, and the training loop keeps model weights in fp32 (autocast is
    per-call). Without the wrap, every full_eval on CUDA crashes with
    'FlashAttention only support fp16 and bf16 data type'."""

    @staticmethod
    def spy_forward(model, device):
        calls = []
        original_forward = model.forward
        original_embed = model.embed

        def wrapped(*args, **kwargs):
            calls.append(torch.is_autocast_enabled(device.type))
            return original_forward(*args, **kwargs)

        def wrapped_embed(*args, **kwargs):
            calls.append(torch.is_autocast_enabled(device.type))
            return original_embed(*args, **kwargs)

        model.forward = wrapped
        model.embed = wrapped_embed
        return calls

    def test_mlm_breakdown_no_autocast_by_default_on_cpu(self):
        with given("a tiny model and random games"):
            model = MaskedStreamTransformer(TINY_CONFIG)
            model.eval()
            games = build_games()
            device = torch.device("cpu")
            calls = self.spy_forward(model, device)

        with when("mlm_breakdown runs with the default bf16 setting"):
            from src.evaluation.eval_suite import mlm_breakdown

            prior = np.full(52, 1 / 52)
            mlm_breakdown(
                model,
                games,
                device,
                windows=1,
                batch_size=1,
                seed=0,
                max_seq_len=64,
                max_segments=16,
                prior=prior,
            )

        with then("the forward runs without autocast on cpu"):
            assert calls and not any(calls)
        return

    def test_mlm_breakdown_autocast_when_bf16_enabled(self):
        with given("a tiny model and random games"):
            model = MaskedStreamTransformer(TINY_CONFIG)
            model.eval()
            games = build_games()
            device = torch.device("cpu")
            calls = self.spy_forward(model, device)

        with when("mlm_breakdown runs with bf16=True"):
            from src.evaluation.eval_suite import mlm_breakdown

            prior = np.full(52, 1 / 52)
            mlm_breakdown(
                model,
                games,
                device,
                windows=1,
                batch_size=1,
                seed=0,
                max_seq_len=64,
                max_segments=16,
                prior=prior,
                bf16=True,
            )

        with then("the forward runs under autocast"):
            assert calls and all(calls)
        return

    def test_embed_segments_autocast_when_bf16_enabled(self):
        with given("a tiny model and token segments"):
            model = MaskedStreamTransformer(TINY_CONFIG)
            model.eval()
            segments = build_games()[0]
            device = torch.device("cpu")
            calls = self.spy_forward(model, device)

        with when("embed_segments runs with bf16=True"):
            from src.evaluation.eval_suite import embed_segments

            embed_segments(model, segments, device, bf16=True)

        with then("the embedding forward runs under autocast"):
            assert calls == [True]
        return

    def test_full_eval_threads_bf16_to_subcalls(self):
        with given("a tiny model, games, and a bf16 spy on every forward"):
            model = MaskedStreamTransformer(TINY_CONFIG)
            model.eval()
            games = build_games()
            device = torch.device("cpu")
            calls = self.spy_forward(model, device)

        with when("full_eval runs with bf16=True"):
            full_eval(
                model,
                games,
                device,
                windows=1,
                batch_size=1,
                max_seq_len=64,
                max_segments=16,
                pool_size=4,
                counterfactual_n=0,
                seed=0,
                bf16=True,
            )

        with then("every recorded forward saw autocast enabled"):
            assert calls and all(calls)
        return

    def test_full_eval_default_keeps_cpu_without_autocast(self):
        with given("a tiny model, games, and a bf16 spy on every forward"):
            model = MaskedStreamTransformer(TINY_CONFIG)
            model.eval()
            games = build_games()
            device = torch.device("cpu")
            calls = self.spy_forward(model, device)

        with when("full_eval runs with default bf16 settings"):
            full_eval(
                model,
                games,
                device,
                windows=1,
                batch_size=1,
                max_seq_len=64,
                max_segments=16,
                pool_size=4,
                counterfactual_n=0,
                seed=0,
            )

        with then("no forward saw autocast enabled on cpu"):
            assert calls and not any(calls)
        return


class TestFullEval:
    def test_returns_expected_metrics(self):
        with given("a tiny model in eval mode and random games"):
            model = MaskedStreamTransformer(TINY_CONFIG)
            model.eval()
            games = build_games()

        with when("the full evaluation runs"):
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

        with then("all expected metrics are present and finite"):
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
        return

    def test_restores_training_mode(self):
        with given("a tiny model left in training mode"):
            model = MaskedStreamTransformer(TINY_CONFIG)
            model.train()
            games = build_games()

        with when("the full evaluation runs"):
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

        with then("the model is returned to training mode"):
            assert model.training
        return
