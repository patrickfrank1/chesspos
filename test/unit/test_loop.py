import pytest
import torch
from torch import nn
from torch.optim.lr_scheduler import LambdaLR

from bdd import given, then, when
from src.modeling.masking import IGNORE_INDEX
from src.training.loop import (
    NceCollapseGuard,
    apply_compile,
    effective_rank_from_hidden,
    evaluate,
    lr_lambda,
    periodic_checkpoint_plan,
    scheduler_start_epoch,
    use_fused_adamw,
)


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


class TestApplyCompile:
    def test_disabled_returns_model_unchanged(self):
        with given("a model and compile disabled"):
            model = nn.Linear(2, 2)

        with when("compile is applied"):
            result = apply_compile(model, False)

        with then("the exact model object is returned"):
            assert result is model
        return

    def test_enabled_wraps_with_dynamic_shapes(self, monkeypatch):
        with given("a model, compile enabled, and torch.compile recorded"):
            model = nn.Linear(2, 2)
            calls: dict = {}

            def fake_compile(module, **kwargs):
                calls.update(kwargs)
                return ("compiled", module)

            monkeypatch.setattr(torch, "compile", fake_compile)

        with when("compile is applied"):
            result = apply_compile(model, True)

        with then("torch.compile wraps the model with dynamic shapes enabled"):
            assert result == ("compiled", model)
            assert calls.get("dynamic") is True
        return


class TestNceCollapseGuard:
    def test_high_initial_nce_does_not_trip_before_arming(self):
        with given("a guard that requires NCE to be healthy before it arms"):
            guard = NceCollapseGuard(threshold=1.5, patience=2, arm_below=1.0)

        with when("NCE is above threshold from initialization"):
            tripped = [guard.update(6.0) for _ in range(5)]

        with then("the guard never trips"):
            assert not any(tripped)
        return

    def test_sustained_nce_above_threshold_trips_after_patience(self):
        with given("an armed guard with patience two"):
            guard = NceCollapseGuard(threshold=1.5, patience=2, arm_below=1.0)
            guard.update(0.8)

        with when("NCE stays above threshold for two logged intervals"):
            first = guard.update(1.6)
            second = guard.update(1.6)

        with then("only the second interval trips the guard"):
            assert first is False
            assert second is True
        return

    def test_healthy_interval_resets_patience(self):
        with given("an armed guard that has seen one high-NCE interval"):
            guard = NceCollapseGuard(threshold=1.5, patience=2, arm_below=1.0)
            guard.update(0.8)
            guard.update(1.6)

        with when("a healthy interval is followed by another high interval"):
            guard.update(0.9)
            tripped = guard.update(1.6)

        with then("the high interval count restarted"):
            assert tripped is False
        return


class TestSchedulerStartEpoch:
    def test_resume_without_scheduler_state_starts_at_checkpoint_step(self):
        with given("a resume from step twenty thousand with no scheduler state"):
            start_step = 20_000
            has_scheduler_state = False

        with when("the LambdaLR start epoch is computed"):
            last_epoch = scheduler_start_epoch(start_step, has_scheduler_state)

        with then("the scheduler is initialized one step before the checkpoint"):
            assert last_epoch == 19_999
        return

    def test_resume_with_scheduler_state_uses_default(self):
        with given("a resume with saved scheduler state"):
            start_step = 20_000
            has_scheduler_state = True

        with when("the LambdaLR start epoch is computed"):
            last_epoch = scheduler_start_epoch(start_step, has_scheduler_state)

        with then("the saved scheduler state is allowed to take over"):
            assert last_epoch == -1
        return

    def test_fresh_run_uses_default(self):
        with given("a fresh run at step zero"):
            start_step = 0
            has_scheduler_state = False

        with when("the LambdaLR start epoch is computed"):
            last_epoch = scheduler_start_epoch(start_step, has_scheduler_state)

        with then("the scheduler starts normally"):
            assert last_epoch == -1
        return

    def test_scheduler_resumes_at_checkpoint_learning_rate(self):
        with given("an optimizer resuming at step twenty thousand"):
            parameter = nn.Parameter(torch.zeros(1))
            optimizer = torch.optim.AdamW([parameter], lr=2e-4)
            for group in optimizer.param_groups:
                group.setdefault("initial_lr", 2e-4)

        with when("a two-hundred-thousand-step cosine scheduler is constructed"):
            scheduler = LambdaLR(
                optimizer,
                lambda step: lr_lambda(step, 500, 200_000),
                last_epoch=scheduler_start_epoch(20_000, False),
            )
            learning_rate = scheduler.get_last_lr()[0]

        with then("the schedule continues near the peak instead of restarting warmup"):
            assert scheduler.last_epoch == 20_000
            assert 1.9e-4 < learning_rate < 2e-4
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


class TestTokenRank:
    def test_collapsed_states_have_rank_one(self):
        with given("identical token states"):
            hidden = torch.ones(64, 16)
            valid = torch.ones(64, dtype=torch.bool)

        with when("the token effective rank is computed"):
            rank = effective_rank_from_hidden(hidden, valid)

        with then("the rank is one"):
            assert rank == pytest.approx(1.0)
        return

    def test_random_states_have_a_high_rank(self):
        with given("random token states"):
            torch.manual_seed(0)
            hidden = torch.randn(512, 32)
            valid = torch.ones(512, dtype=torch.bool)

        with when("the token effective rank is computed"):
            rank = effective_rank_from_hidden(hidden, valid)

        with then("many dimensions are used"):
            assert rank > 10
        return

    def test_invalid_tokens_are_excluded(self):
        with given("random states followed by collapsed masked states"):
            torch.manual_seed(0)
            hidden = torch.cat([torch.randn(64, 16), torch.ones(64, 16)], dim=0)
            valid = torch.tensor([True] * 64 + [False] * 64)

        with when("the token effective rank is computed"):
            rank = effective_rank_from_hidden(hidden, valid)

        with then("only the valid states influence the rank"):
            assert rank > 10
        return

    def test_subsamples_to_the_token_budget(self):
        with given("more token states than the budget"):
            torch.manual_seed(0)
            hidden = torch.randn(1000, 16)
            valid = torch.ones(1000, dtype=torch.bool)

        with when("the rank is computed with a 100-token budget"):
            rank = effective_rank_from_hidden(hidden, valid, max_tokens=100)

        with then("the result is still a valid rank"):
            assert rank > 1
        return


class TestPeriodicCheckpointPlan:
    def test_eval_step_saves_before_evaluating(self):
        with given("a step where evaluation and saving are both due"):
            step = 2000
            steps = 100_000
            save_every = 2000
            eval_every = 2000

        with when("the periodic checkpoint plan is computed"):
            plan = periodic_checkpoint_plan(
                step,
                steps=steps,
                save_every=save_every,
                eval_every=eval_every,
            )

        with then("the checkpoint is saved before evaluation"):
            assert plan.save_before_eval is True
            assert plan.evaluate is True
            assert plan.save_after_eval is False
        return

    def test_eval_without_a_scheduled_save_still_saves_first(self):
        with given("an evaluation step that is not a scheduled save step"):
            step = 1000
            steps = 100_000
            save_every = 5000
            eval_every = 1000

        with when("the periodic checkpoint plan is computed"):
            plan = periodic_checkpoint_plan(
                step,
                steps=steps,
                save_every=save_every,
                eval_every=eval_every,
            )

        with then("a checkpoint is still written before evaluation"):
            assert plan.save_before_eval is True
            assert plan.evaluate is True
            assert plan.save_after_eval is False
        return

    def test_final_step_saves_again_after_evaluating(self):
        with given("the final training step, which is also an eval step"):
            step = 5000
            steps = 5000
            save_every = 2000
            eval_every = 2000

        with when("the periodic checkpoint plan is computed"):
            plan = periodic_checkpoint_plan(
                step,
                steps=steps,
                save_every=save_every,
                eval_every=eval_every,
            )

        with then("the checkpoint is refreshed with the final eval metrics"):
            assert plan.save_before_eval is True
            assert plan.evaluate is True
            assert plan.save_after_eval is True
        return

    def test_save_only_step_does_not_evaluate(self):
        with given("a scheduled save step that is not an eval step"):
            step = 4000
            steps = 100_000
            save_every = 2000
            eval_every = 3000

        with when("the periodic checkpoint plan is computed"):
            plan = periodic_checkpoint_plan(
                step,
                steps=steps,
                save_every=save_every,
                eval_every=eval_every,
            )

        with then("only the checkpoint save runs"):
            assert plan.save_before_eval is True
            assert plan.evaluate is False
            assert plan.save_after_eval is False
        return

    def test_regular_step_runs_no_periodic_work(self):
        with given("a step that is neither a save nor an eval step"):
            step = 1500
            steps = 100_000
            save_every = 2000
            eval_every = 2000

        with when("the periodic checkpoint plan is computed"):
            plan = periodic_checkpoint_plan(
                step,
                steps=steps,
                save_every=save_every,
                eval_every=eval_every,
            )

        with then("nothing periodic runs"):
            assert plan.save_before_eval is False
            assert plan.evaluate is False
            assert plan.save_after_eval is False
        return
