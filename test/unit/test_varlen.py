import torch

from bdd import given, then, when
from src.modeling.varlen import (
    cu_seqlens,
    flatten_valid,
    group_cu_seqlens,
    scatter_valid,
    varlen_attention,
)


class TestFlattenValid:
    def test_matches_boolean_indexing_with_tail_padding(self):
        with given("a padded tensor and a tail-padded mask"):
            x = torch.randn(4, 6, 3)
            mask = torch.tensor(
                [
                    [1, 1, 1, 0, 0, 0],
                    [1, 1, 0, 0, 0, 0],
                    [1, 1, 1, 1, 1, 0],
                    [1, 0, 0, 0, 0, 0],
                ],
                dtype=torch.bool,
            )

        with when("the valid positions are flattened"):
            flat = flatten_valid(x, mask)

        with then("the result equals boolean indexing in row-major order"):
            assert torch.equal(flat, x[mask])
            assert flat.shape == (11, 3)
        return

    def test_matches_boolean_indexing_with_holes(self):
        with given("a mask with interior holes"):
            x = torch.randn(3, 5, 2)
            mask = torch.tensor(
                [
                    [1, 0, 1, 0, 1],
                    [0, 1, 0, 1, 0],
                    [1, 1, 0, 0, 1],
                ],
                dtype=torch.bool,
            )

        with when("the valid positions are flattened"):
            flat = flatten_valid(x, mask)

        with then("interior holes are still skipped in row-major order"):
            assert torch.equal(flat, x[mask])
        return


class TestScatterValid:
    def test_roundtrip_matches_boolean_assignment(self):
        with given("a padded tensor and an arbitrary mask"):
            x = torch.randn(3, 5, 2)
            mask = torch.tensor(
                [
                    [1, 0, 1, 0, 1],
                    [0, 1, 0, 1, 0],
                    [1, 1, 0, 0, 1],
                ],
                dtype=torch.bool,
            )

        with when("flattened values are scattered back"):
            out = scatter_valid(flatten_valid(x, mask), mask)

        with then("valid positions are restored and padding stays zero"):
            expected = torch.zeros_like(x)
            expected[mask] = x[mask]
            assert torch.equal(out, expected)
        return

    def test_handles_no_valid_tokens(self):
        with given("a mask with no valid positions"):
            x = torch.randn(2, 4, 3)
            mask = torch.zeros(2, 4, dtype=torch.bool)

        with when("flatten and scatter are applied"):
            flat = flatten_valid(x, mask)
            out = scatter_valid(flat, mask)

        with then("the results are empty and all-zero"):
            assert flat.shape == (0, 3)
            assert torch.equal(out, torch.zeros_like(x))
        return


class TestGroupCuSeqlens:
    def test_groups_by_row_and_segment_and_skips_empty_groups(self):
        with given("two rows with segment ids and holes in the valid mask"):
            segment_ids = torch.tensor(
                [
                    [0, 0, 0, 1, 1, 1, 1],
                    [0, 0, 1, 1, 1, 1, 1],
                ]
            )
            mask = torch.tensor(
                [
                    [1, 1, 0, 1, 1, 0, 1],
                    [1, 0, 1, 1, 1, 1, 0],
                ],
                dtype=torch.bool,
            )

        with when("the grouped cumulative offsets are built"):
            cu, max_len = group_cu_seqlens(mask, segment_ids, num_groups=2)

        with then("only non-empty segments are included in row-major order"):
            assert cu.tolist() == [0, 2, 5, 6, 10]
            assert max_len == 4
        return

    def test_uses_static_max_length_hint(self):
        with given("a grouped mask with a known upper bound"):
            segment_ids = torch.tensor([[0, 0, 0, 1, 1]])
            mask = torch.tensor([[1, 1, 0, 1, 0]], dtype=torch.bool)

        with when("a static maximum length is supplied"):
            _, max_len = group_cu_seqlens(
                mask, segment_ids, num_groups=2, max_len_hint=64
            )

        with then("the hint is returned without a device scalar sync"):
            assert max_len == 64
        return

    def test_returns_zero_offsets_when_nothing_is_valid(self):
        with given("an empty valid mask"):
            segment_ids = torch.zeros(2, 3, dtype=torch.long)
            mask = torch.zeros(2, 3, dtype=torch.bool)

        with when("the grouped cumulative offsets are built"):
            cu, max_len = group_cu_seqlens(mask, segment_ids, num_groups=2)

        with then("there is only the leading zero"):
            assert cu.tolist() == [0]
            assert max_len == 0
        return


class TestCuSeqlens:
    def test_max_length_is_a_static_upper_bound(self):
        with given("a mask whose longest valid row is shorter than the sequence"):
            mask = torch.tensor(
                [[1, 1, 1, 1, 0, 0, 0, 0], [1, 1, 0, 0, 0, 0, 0, 0]],
                dtype=torch.bool,
            )

        with when("the offsets are built"):
            cu, max_len = cu_seqlens(mask)

        with then("the maximum is the static sequence width and offsets are correct"):
            assert cu.tolist() == [0, 4, 6]
            assert max_len == 8
        return


class TestVarlenAttention:
    def test_supports_different_query_and_key_lengths(self):
        with given("query and key sequences with different segment lengths"):
            torch.manual_seed(0)
            q = torch.randn(5, 2, 4)
            k = torch.randn(8, 2, 4)
            v = torch.randn(8, 2, 4)
            cu_q = torch.tensor([0, 2, 3, 5], dtype=torch.int32)
            cu_k = torch.tensor([0, 3, 4, 8], dtype=torch.int32)

        with when("varlen attention runs with separate offsets"):
            out = varlen_attention(
                q,
                k,
                v,
                cu_q,
                2,
                cu_seqlens_k=cu_k,
                max_seqlen_k=4,
            )

        with then("each query segment attends only its paired key segment"):
            expected = []
            for q_start, q_end, k_start, k_end in zip(
                cu_q[:-1].tolist(),
                cu_q[1:].tolist(),
                cu_k[:-1].tolist(),
                cu_k[1:].tolist(),
            ):
                attended = torch.nn.functional.scaled_dot_product_attention(
                    q[q_start:q_end].transpose(0, 1).unsqueeze(0),
                    k[k_start:k_end].transpose(0, 1).unsqueeze(0),
                    v[k_start:k_end].transpose(0, 1).unsqueeze(0),
                )
                expected.append(attended.squeeze(0).transpose(0, 1))
            assert torch.allclose(out, torch.cat(expected, dim=0), atol=1e-6)
        return
