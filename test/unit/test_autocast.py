from __future__ import annotations

from contextlib import nullcontext

import torch

from bdd import given, then, when
from src.training.autocast import autocast_context


class TestAutocastContext:
    def test_rejects_an_unknown_mode(self):
        with given("an unsupported autocast mode"):
            mode = "fp8"

        with when("the context is requested"):
            error = None
            try:
                autocast_context(torch.device("cpu"), mode)
            except ValueError as exc:
                error = exc

        with then("a ValueError is raised"):
            assert isinstance(error, ValueError)
        return

    def test_disabled_is_a_no_op(self):
        with given("a CPU tensor and the none mode"):
            value = torch.randn(4)
            context = autocast_context(torch.device("cpu"), "none")

        with when("the context is entered"):
            with context:
                result = value * 2

        with then("the result keeps its dtype and the context is a nullcontext"):
            assert isinstance(context, nullcontext)
            assert result.dtype == value.dtype
            assert torch.allclose(result, value * 2)
        return

    def test_bf16_falls_back_to_no_op_without_cuda(self):
        with given("the bf16 mode on a CPU device"):
            value = torch.randn(4)
            context = autocast_context(torch.device("cpu"), "bf16")

        with when("the context is entered"):
            with context:
                result = value * 2

        with then("nothing is cast and the context is a nullcontext"):
            assert isinstance(context, nullcontext)
            assert result.dtype == value.dtype
        return
