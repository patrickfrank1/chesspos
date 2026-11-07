import torch

from bdd import given, then, when
from src.modeling.varlen import flatten_valid, scatter_valid


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
