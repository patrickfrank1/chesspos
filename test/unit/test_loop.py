import torch
from torch import nn

from bdd import given, then, when
from src.modeling.masking import IGNORE_INDEX
from src.training.loop import evaluate, use_fused_adamw


class FakeModel(nn.Module):
    """Stub with an identity head so argmax predictions are hand-crafted."""

    def __init__(self, hidden: torch.Tensor):
        super().__init__()
        self.hidden = hidden
        self.head = nn.Identity()

    def forward(
        self,
        tokens: torch.Tensor,
        segment_ids: torch.Tensor,
        padding_mask: torch.Tensor | None = None,
        targets: torch.Tensor | None = None,
        weights: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        return {"loss": torch.tensor(0.25), "hidden_states": self.hidden}


def fake_batch(hidden: torch.Tensor, targets: torch.Tensor) -> dict[str, torch.Tensor]:
    shape = targets.shape
    ones = torch.ones(shape, dtype=torch.long)
    return {
        "tokens": ones,
        "segment_ids": torch.zeros(shape, dtype=torch.long),
        "padding_mask": targets.ne(IGNORE_INDEX),
        "targets": targets,
    }, hidden


class TestEvaluate:
    def test_accuracy_without_full_logits(self):
        with given(
            "a stub model whose identity head turns one-hot hidden states into logits"
        ):
            vocab = 4
            hidden = torch.zeros(1, 3, vocab)
            hidden[0, 0, 2] = 1.0  # correct prediction for target 2
            hidden[0, 2, 1] = 1.0  # wrong prediction for target 3
            model = FakeModel(hidden)
            targets = torch.tensor([[2, IGNORE_INDEX, 3]])
            batch, _ = fake_batch(hidden, targets)

        with when("evaluate runs over a single batch"):
            loss, accuracy = evaluate(
                model, [batch], torch.device("cpu"), max_batches=1, bf16=False
            )

        with then("the stub loss is averaged and accuracy counts valid positions only"):
            assert loss == 0.25
            assert accuracy == 0.5
        return


class TestFusedAdamW:
    def test_disabled_without_cuda(self):
        with given("a CPU device"):
            device = torch.device("cpu")

        with when("fused AdamW support is queried"):
            fused = use_fused_adamw(device)

        with then("fused is disabled"):
            assert fused is False
        return

    def test_enabled_on_available_cuda(self, monkeypatch):
        with given("a CUDA device with CUDA reported available"):
            monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
            device = torch.device("cuda")

        with when("fused AdamW support is queried"):
            fused = use_fused_adamw(device)

        with then("fused is enabled"):
            assert fused is True
        return

    def test_disabled_when_cuda_unavailable(self, monkeypatch):
        with given("a CUDA device with CUDA reported unavailable"):
            monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
            device = torch.device("cuda")

        with when("fused AdamW support is queried"):
            fused = use_fused_adamw(device)

        with then("fused is disabled"):
            assert fused is False
        return
