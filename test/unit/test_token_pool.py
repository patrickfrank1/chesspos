from __future__ import annotations

import torch

from bdd import given, then, when
from src.dataset.token_stream import CLS, PAD, SEP, TURN_BLACK
from src.modeling.token_pool import TokenPool, TokenPoolConfig


def make_batch() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    tokens = torch.tensor(
        [
            [CLS, 10, 11, SEP, 12, 13, SEP, PAD, PAD],
            [CLS, 20, SEP, PAD, PAD, PAD, PAD, PAD, PAD],
        ],
        dtype=torch.long,
    )
    segment_ids = torch.tensor(
        [
            [0, 0, 0, 1, 1, 1, 2, 2, 2],
            [0, 0, 1, 1, 1, 1, 1, 1, 1],
        ],
        dtype=torch.long,
    )
    padding_mask = tokens.ne(PAD)
    hidden = torch.randn(2, 9, 4)
    return hidden, tokens, segment_ids, padding_mask


class TestMeanPooling:
    def test_mean_pool_averages_valid_tokens_per_segment(self):
        with given("a batch with two and one valid tokens in its segments"):
            hidden, tokens, segment_ids, padding_mask = make_batch()
            pool = TokenPool(TokenPoolConfig(d_model=4, mode="mean"))

        with when("the segments are pooled"):
            embeddings, mask = pool(hidden, tokens, segment_ids, padding_mask)

        with then("structural and padded tokens are excluded"):
            assert mask.tolist()[0][:3] == [True, True, False]
            assert mask.tolist()[1][:3] == [True, False, False]
            torch.testing.assert_close(
                embeddings[0, 0], (hidden[0, 1] + hidden[0, 2]) / 2
            )
            torch.testing.assert_close(
                embeddings[0, 1], (hidden[0, 4] + hidden[0, 5]) / 2
            )
            torch.testing.assert_close(embeddings[1, 0], hidden[1, 1])
            torch.testing.assert_close(embeddings[0, 2], torch.zeros(4))
        return

    def test_content_filter_excludes_state_tokens(self):
        with given("a segment whose last token is a turn token"):
            tokens = torch.tensor([[CLS, 10, 11, TURN_BLACK, SEP]], dtype=torch.long)
            segment_ids = torch.tensor([[0, 0, 0, 0, 1]], dtype=torch.long)
            padding_mask = tokens.ne(PAD)
            hidden = torch.randn(1, 5, 4)
            pool = TokenPool(
                TokenPoolConfig(d_model=4, mode="mean", token_filter="content")
            )

        with when("the segment is pooled"):
            embeddings, mask = pool(hidden, tokens, segment_ids, padding_mask)

        with then("the turn token is not part of the average"):
            torch.testing.assert_close(
                embeddings[0, 0], (hidden[0, 1] + hidden[0, 2]) / 2
            )
            assert mask.tolist()[0][0] is True
        return


class TestAttentionPooling:
    def test_zero_query_reproduces_mean_pooling(self):
        with given("an attention pool initialised at zero and a mean pool"):
            hidden, tokens, segment_ids, padding_mask = make_batch()
            attention = TokenPool(TokenPoolConfig(d_model=4, mode="attention"))
            mean = TokenPool(TokenPoolConfig(d_model=4, mode="mean"))

        with when("both pools run on the same batch"):
            attention_embeddings, attention_mask = attention(
                hidden, tokens, segment_ids, padding_mask
            )
            mean_embeddings, mean_mask = mean(hidden, tokens, segment_ids, padding_mask)

        with then("the attention pool starts exactly at the mean"):
            torch.testing.assert_close(attention_embeddings, mean_embeddings)
            assert attention_mask.tolist() == mean_mask.tolist()
        return

    def test_padding_and_structural_tokens_do_not_affect_the_result(self):
        with given("an attention pool with a random query"):
            hidden, tokens, segment_ids, padding_mask = make_batch()
            pool = TokenPool(TokenPoolConfig(d_model=4, mode="attention"))
            with torch.no_grad():
                pool.query.normal_()

        with when("a padded position and a separator are changed"):
            perturbed_hidden = hidden.clone()
            perturbed_hidden[0, 7] = 1e6
            perturbed_hidden[0, 8] = -1e6
            base, _ = pool(hidden, tokens, segment_ids, padding_mask)
            perturbed, _ = pool(perturbed_hidden, tokens, segment_ids, padding_mask)

        with then("the pooled vectors are unchanged"):
            torch.testing.assert_close(base, perturbed)
        return

    def test_attention_puts_weight_on_the_query_direction(self):
        with given("hidden states aligned with the x axis and a query on x"):
            hidden = torch.zeros(1, 3, 2)
            hidden[0, 1] = torch.tensor([10.0, 0.0])
            hidden[0, 2] = torch.tensor([0.0, 10.0])
            tokens = torch.tensor([[10, 11, SEP]], dtype=torch.long)
            segment_ids = torch.tensor([[0, 0, 1]], dtype=torch.long)
            padding_mask = torch.ones(1, 3, dtype=torch.bool)
            pool = TokenPool(TokenPoolConfig(d_model=2, mode="attention"))
            with torch.no_grad():
                pool.query.copy_(torch.tensor([1.0, 0.0]))

        with when("the segment is pooled"):
            embeddings, _ = pool(hidden, tokens, segment_ids, padding_mask)

        with then("the aligned token dominates the embedding"):
            assert embeddings[0, 0, 0] > 9.0
            assert embeddings[0, 0, 1] < 1.0
        return

    def test_query_receives_gradients(self):
        with given("an attention pool and a scalar loss on its output"):
            hidden, tokens, segment_ids, padding_mask = make_batch()
            pool = TokenPool(TokenPoolConfig(d_model=4, mode="attention"))

        with when("the loss is backpropagated"):
            embeddings, _ = pool(hidden, tokens, segment_ids, padding_mask)
            embeddings.sum().backward()

        with then("the query has a gradient"):
            assert pool.query.grad is not None
            assert pool.query.grad.abs().sum() > 0
        return


class TestConfig:
    def test_rejects_unknown_modes_and_filters(self):
        with given("invalid configurations"):
            cases = [
                dict(d_model=4, mode="max"),
                dict(d_model=4, token_filter="state"),
                dict(d_model=0),
            ]

        with when("the configs are constructed"):
            errors = []
            for case in cases:
                try:
                    TokenPoolConfig(**case)
                except ValueError:
                    errors.append(True)

        with then("each invalid config raises"):
            assert errors == [True, True, True]
        return
