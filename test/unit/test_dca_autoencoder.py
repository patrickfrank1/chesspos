from __future__ import annotations

import chess
import numpy as np
import pytest
import torch

from bdd import then
from src.dataset.token_stream import (
    PAD,
    SEP,
    VOCAB_SIZE,
    TokenStreamEncoder,
    pack_stream,
)
from src.modeling.autoencoder import (
    AttentionAutoencoderConfig,
    AttentionAutoencoderModel,
    build_attention_autoencoder,
    production_config,
)
from src.modeling.checkpointing import autocast_for
from src.modeling.factory import config_asdict, model_config_from_dict
from src.modeling.masking import MaskingConfig, mask_stream
from src.modeling.shared.dca import DCAStackState, DCALayer, GRNv3
from src.modeling.shared.layers import VarlenTransformerLayer


def tiny_config(**overrides) -> AttentionAutoencoderConfig:
    values = {
        "vocab_size": VOCAB_SIZE,
        "d_model": 16,
        "n_heads": 2,
        "n_layers": 2,
        "n_decoder_layers": 2,
        "d_ff": 32,
        "dropout": 0.0,
        "max_seq_len": 1024,
        "max_segments": 8,
        "latent_dim": 8,
    }
    values.update(overrides)
    return AttentionAutoencoderConfig(**values)


def dca_config(**overrides) -> AttentionAutoencoderConfig:
    values = {
        "dca": True,
        "dca_locations": "both",
        "dca_past_layers_k": 2,
    }
    values.update(overrides)
    return tiny_config(**values)


def game_segments(count: int = 4) -> list[np.ndarray]:
    encoder = TokenStreamEncoder()
    board = chess.Board()
    segments = []
    for _ in range(count):
        segments.append(encoder.encode(board))
        board.push(list(board.legal_moves)[0])
    return segments


def masked_batch(seed: int = 0) -> dict[str, torch.Tensor]:
    tokens_np = pack_stream(game_segments())
    masking = MaskingConfig(
        p_rect=1.0, p_random=0.0, p_board=0.0, p_span=0.0, p_skip_gram=0.0
    )
    masked, targets, weights = mask_stream(
        tokens_np, masking, np.random.default_rng(seed)
    )
    return {
        "tokens": torch.from_numpy(masked.astype(np.int64))[None],
        "segment_ids": torch.from_numpy(np.cumsum(masked == SEP).astype(np.int64))[
            None
        ],
        "padding_mask": torch.from_numpy(masked != PAD)[None],
        "targets": torch.from_numpy(targets.astype(np.int64))[None],
        "weights": torch.from_numpy(weights)[None],
    }


def flat_stream(batch: dict[str, torch.Tensor]) -> torch.Tensor:
    from src.modeling.varlen import flatten_valid

    x = torch.randn(
        batch["tokens"].shape[0],
        batch["tokens"].shape[1],
        16,
        generator=torch.Generator().manual_seed(0),
    )
    return flatten_valid(x, batch["padding_mask"])


class TestStackBookkeeping:
    def test_stack_sums_to_the_residual_stream(self):
        # Given a stack in full and first+last-k modes
        x0 = torch.randn(6, 4)
        increments = [torch.randn(6, 4) for _ in range(5)]

        for past_k in (0, 1, 2):
            stack = DCAStackState(x0, past_k, past_k + 2 if past_k else 6)
            stream = x0
            for increment in increments:
                slots = stack.stack()
                with then("the slots sum to the current stream"):
                    assert torch.allclose(
                        torch.stack(slots).sum(dim=0), stream, atol=1e-6
                    )
                stream = stream + increment
                stack.update(increment)

    def test_recent_stack_is_zero_padded_early(self):
        # Given a first+last-2 stack at the first block
        x0 = torch.randn(5, 3)
        stack = DCAStackState(x0, 2, 4)

        # When the slots are requested before any increment exists
        slots = stack.stack()

        # Then the padding slots are exactly zero
        with then("missing recent slots contribute nothing"):
            assert len(slots) == 4
            assert torch.equal(slots[2], torch.zeros_like(x0))
            assert torch.equal(slots[3], torch.zeros_like(x0))
            assert torch.allclose(slots[0], x0)


class TestDCALayer:
    def test_dca_layer_reproduces_the_standard_block_at_init(self):
        # Given a standard block and its DCA retrofit
        torch.manual_seed(0)
        standard = VarlenTransformerLayer(16, 2, 32, 0.0).eval()
        dca = DCALayer.from_standard(standard).eval()

        # When both run on the same stream with a one-hot stack
        x = torch.randn(5, 16)
        cu = torch.tensor([0, 5], dtype=torch.int32)
        with torch.no_grad():
            expected = standard(x, cu, 5)
            increment = dca(x, x, x, x, cu, 5)

        # Then the DCA increment matches the standard block's residual delta
        with then("the emitted increment is the standard delta"):
            assert torch.allclose(increment, expected - x, atol=1e-5)

    def test_grn_composes_from_a_list_without_stacking(self, monkeypatch):
        # Given a GRN and a stack of slots
        grn = GRNv3(16, slots=4, outputs=3)
        slots = [torch.randn(7, 16) for _ in range(4)]

        def boom(*args, **kwargs):
            raise AssertionError("torch.stack must not be used on the stack")

        monkeypatch.setattr(torch, "stack", boom)

        # When the GRN runs
        composed = grn(slots)

        # Then it returns one tensor per output without materialising the stack
        with then("three composed inputs come back"):
            assert len(composed) == 3
            assert all(item.shape == (7, 16) for item in composed)

    def test_routing_receives_gradients_and_padding_slots_stay_dead(self):
        # Given a DCA encoder in training mode
        model = build_attention_autoencoder(dca_config()).train()
        batch = masked_batch()

        # When one forward/backward runs
        output = model(
            batch["tokens"],
            batch["segment_ids"],
            padding_mask=batch["padding_mask"],
            targets=batch["targets"],
            weights=batch["weights"],
        )
        output["loss"].backward()

        # Then the routing parameters learn, and the padded slots do not
        with then("proj and live bias rows have gradients"):
            first = model.encoder.grn[0]
            assert first.proj.weight.grad is not None
            assert first.proj.weight.grad.abs().sum() > 0
            assert first.bias.grad is not None
            assert first.bias.grad[0, 0].abs().sum() > 0
            assert torch.equal(
                first.bias.grad[0, 2:], torch.zeros_like(first.bias.grad[0, 2:])
            )


class TestModelWiring:
    def _retrofit(self, dca_model: AttentionAutoencoderModel, standard_model) -> None:
        params = dict(dca_model.named_parameters())
        with torch.no_grad():
            for name, tensor in standard_model.state_dict().items():
                if ".blocks." in name or name.endswith("grn.proj.weight"):
                    continue
                if name in params:
                    params[name].copy_(tensor)
            for standard_blocks, dca_blocks in (
                (standard_model.encoder.blocks, dca_model.encoder.blocks),
                (standard_model.decoder.blocks, dca_model.decoder.blocks),
            ):
                for standard_block, dca_block in zip(standard_blocks, dca_blocks):
                    dca_block.load_from_standard(standard_block)
            for module in (dca_model.encoder, dca_model.decoder):
                for grn in module.grn:
                    grn.proj.weight.zero_()
        return

    def test_dca_model_matches_the_baseline_at_step_zero(self):
        # Given a standard model and a DCA model retrofitted from it
        torch.manual_seed(0)
        standard = build_attention_autoencoder(tiny_config()).eval()
        dca = build_attention_autoencoder(dca_config()).eval()
        self._retrofit(dca, standard)
        batch = masked_batch(seed=3)

        # When both encode and score the same batch
        with torch.no_grad():
            standard_latent = standard.encode_latent(
                batch["tokens"], batch["segment_ids"], batch["padding_mask"]
            )
            dca_latent = dca.encode_latent(
                batch["tokens"], batch["segment_ids"], batch["padding_mask"]
            )
            standard_out = standard.head(
                standard(
                    batch["tokens"],
                    batch["segment_ids"],
                    padding_mask=batch["padding_mask"],
                )["hidden_states"]
            )
            dca_out = dca.head(
                dca(
                    batch["tokens"],
                    batch["segment_ids"],
                    padding_mask=batch["padding_mask"],
                )["hidden_states"]
            )

        # Then the retrofit is the identity at step zero
        with then("latents and logits agree"):
            assert torch.allclose(standard_latent, dca_latent, atol=1e-6)
            assert torch.allclose(standard_out, dca_out, atol=1e-5)

    def test_bottleneck_isolation_between_encoder_and_decoder(self):
        # Given encoder-only and both-sides DCA models
        torch.manual_seed(0)
        encoder_only = build_attention_autoencoder(
            dca_config(dca_locations="encoder")
        ).eval()
        both = build_attention_autoencoder(dca_config(dca_locations="both")).eval()
        with torch.no_grad():
            params = dict(both.named_parameters())
            for name, tensor in encoder_only.state_dict().items():
                if name.startswith("encoder.") or name.startswith("bottleneck."):
                    params[name].copy_(tensor)
        batch = masked_batch(seed=5)

        # When both encode the same window
        with torch.no_grad():
            latent_encoder = encoder_only.encode_latent(
                batch["tokens"], batch["segment_ids"], batch["padding_mask"]
            )
            latent_both = both.encode_latent(
                batch["tokens"], batch["segment_ids"], batch["padding_mask"]
            )

        # Then the decoder's routing cannot influence the latent
        with then("the latent is bit-identical"):
            assert torch.equal(latent_encoder, latent_both)

    def test_dca_routes_only_where_configured(self):
        # Given three location settings
        with then("routes() reflects the location flag"):
            assert tiny_config().routes("encoder") is False
            assert dca_config(dca_locations="encoder").routes("decoder") is False
            assert dca_config(dca_locations="decoder").routes("encoder") is False
            assert dca_config(dca_locations="both").routes("encoder") is True
            assert dca_config(dca_locations="both").routes("decoder") is True

    def test_every_dca_variant_runs_a_training_step(self):
        # Given the supported stack modes, variants and locations
        cases = (
            {"dca_past_layers_k": 0},
            {"dca_past_layers_k": 1},
            {"dca_variant": "grn"},
            {"dca_locations": "encoder"},
            {"dca_locations": "decoder"},
        )
        batch = masked_batch(seed=19)

        # When each one takes a forward/backward step
        losses = []
        for overrides in cases:
            model = build_attention_autoencoder(dca_config(**overrides)).train()
            out = model(
                batch["tokens"],
                batch["segment_ids"],
                padding_mask=batch["padding_mask"],
                targets=batch["targets"],
                weights=batch["weights"],
            )
            out["loss"].backward()
            losses.append(out["loss"].item())

        # Then every mode is finite
        with then("all variants train"):
            assert len(losses) == len(cases)
            assert all(torch.isfinite(torch.tensor(value)) for value in losses)


class TestConfiguration:
    def test_production_config_is_the_8_4_256_dca_recipe(self):
        # Given the production recipe
        config = production_config()

        # When the model is built
        model = build_attention_autoencoder(config)
        parameters = sum(p.numel() for p in model.parameters())

        # Then it is the measured 8:4 candidate shape with DCA and checkpointing
        with then("the recipe and parameter count match"):
            assert (config.n_layers, config.n_decoder_layers) == (8, 4)
            assert config.latent_dim == 256
            assert config.d_model == 512
            assert config.dca is True
            assert config.dca_locations == "both"
            assert config.grad_checkpointing is True
            assert 26_000_000 <= parameters <= 27_500_000

    def test_invalid_dca_settings_are_rejected(self):
        # Given invalid DCA settings
        invalid = (
            {"dca_locations": "bottleneck"},
            {"dca_variant": "dca-v2"},
            {"dca_past_layers_k": -1},
        )

        # When each config is built
        errors = []
        for overrides in invalid:
            try:
                tiny_config(**overrides)
            except ValueError as error:
                errors.append(error)

        # Then every invalid setting is rejected
        with then("all three fail"):
            assert len(errors) == len(invalid)

    def test_config_round_trip_keeps_dca_flags(self):
        # Given a DCA config
        config = dca_config(dca_variant="grn", grad_checkpointing=True)

        # When it is serialised and rebuilt
        restored = model_config_from_dict(config_asdict(config))

        # Then the routing flags survive
        with then("the rebuilt config is identical"):
            assert restored.dca is True
            assert restored.dca_locations == "both"
            assert restored.dca_variant == "grn"
            assert restored.grad_checkpointing is True


class TestCheckpointing:
    def _twin_models(self):
        torch.manual_seed(0)
        plain = build_attention_autoencoder(tiny_config()).train()
        torch.manual_seed(0)
        checkpointed = build_attention_autoencoder(
            tiny_config(grad_checkpointing=True)
        ).train()
        checkpointed.load_state_dict(plain.state_dict())
        return plain, checkpointed

    def test_gradient_checkpointing_is_numerically_inert(self):
        # Given two identical models, one checkpointed
        plain, checkpointed = self._twin_models()
        batch = masked_batch(seed=7)

        # When both take one optimizer step worth of gradients
        losses = []
        grads = []
        for model in (plain, checkpointed):
            model.zero_grad(set_to_none=True)
            output = model(
                batch["tokens"],
                batch["segment_ids"],
                padding_mask=batch["padding_mask"],
                targets=batch["targets"],
                weights=batch["weights"],
            )
            output["loss"].backward()
            losses.append(output["loss"].item())
            grads.append(model.encoder.blocks[0].qkv.weight.grad.clone())

        # Then the loss and gradients agree
        with then("checkpointing changes nothing"):
            assert abs(losses[0] - losses[1]) < 1e-6
            assert torch.allclose(grads[0], grads[1], atol=1e-5)

    def test_checkpointing_recomputes_each_block(self, monkeypatch):
        # Given a checkpointed model and a call counter on the blocks
        model = build_attention_autoencoder(
            tiny_config(grad_checkpointing=True)
        ).train()
        calls = {"n": 0}
        original = VarlenTransformerLayer.forward

        def counting(self, x, cu, max_seqlen):
            calls["n"] += 1
            return original(self, x, cu, max_seqlen)

        monkeypatch.setattr(VarlenTransformerLayer, "forward", counting)
        batch = masked_batch(seed=11)

        # When forward and backward run
        output = model(
            batch["tokens"],
            batch["segment_ids"],
            padding_mask=batch["padding_mask"],
            targets=batch["targets"],
            weights=batch["weights"],
        )
        output["loss"].backward()

        # Then every block ran twice: once in forward and once on recompute
        with then("the block forward count doubles"):
            assert calls["n"] == 2 * model.config.n_layers + 2 * (
                model.config.n_decoder_layers
            )

    def test_block_loop_autocast_is_bf16_on_cuda(self):
        # Given the block-loop autocast factory
        cuda_context = autocast_for("cuda")
        cpu_context = autocast_for("cpu")

        # Then CUDA recompute runs under bf16 autocast and CPU under nullcontext
        with then("the flash dtype is preserved"):
            assert cuda_context.fast_dtype == torch.bfloat16
            assert type(cpu_context).__name__ == "nullcontext"


class TestAccumulation:
    def test_accumulated_gradients_match_a_single_large_batch(self):
        # Given a model and one batch made of two identical windows
        torch.manual_seed(0)
        model = build_attention_autoencoder(tiny_config()).train()
        batch = masked_batch(seed=13)
        doubled = {
            key: torch.cat([value, value], dim=0) for key, value in batch.items()
        }

        # When the loss is accumulated over the two halves
        model.zero_grad(set_to_none=True)
        for _ in range(2):
            output = model(
                batch["tokens"],
                batch["segment_ids"],
                padding_mask=batch["padding_mask"],
                targets=batch["targets"],
                weights=batch["weights"],
            )
            (output["loss"] / 2).backward()
        accumulated = model.encoder.blocks[0].qkv.weight.grad.clone()

        # Then it matches one backward over the doubled batch
        model.zero_grad(set_to_none=True)
        output = model(
            doubled["tokens"],
            doubled["segment_ids"],
            padding_mask=doubled["padding_mask"],
            targets=doubled["targets"],
            weights=doubled["weights"],
        )
        output["loss"].backward()
        whole = model.encoder.blocks[0].qkv.weight.grad

        with then("the gradients agree within float tolerance"):
            assert torch.allclose(accumulated, whole, atol=1e-5)


class TestResumeMath:
    def test_stream_skip_uses_the_effective_batch(self):
        # Given a dataset config that accumulates three micro-batches
        import src.training.window_dataset as window_dataset

        config = window_dataset.WindowDatasetConfig(
            batch_size=2, accum_steps=3, resume_step=5, num_workers=1
        )

        # When the resume skip is computed from the effective batch
        skip = window_dataset.worker_skip_windows(
            config.resume_step, config.effective_batch_size, config.num_workers, 0
        )

        # Then it covers three times the micro-batch per optimizer step
        with then("the stream advances by batch x accumulation"):
            assert config.effective_batch_size == 6
            assert skip == 5 * 6


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA for compile")
def test_compiled_dca_step_is_finite():
    # Given a compiled DCA model on CUDA
    model = build_attention_autoencoder(dca_config()).cuda()
    compiled = torch.compile(model, dynamic=True)
    batch = {key: value.cuda() for key, value in masked_batch(seed=17).items()}

    # When one step runs
    output = compiled(
        batch["tokens"],
        batch["segment_ids"],
        padding_mask=batch["padding_mask"],
        targets=batch["targets"],
        weights=batch["weights"],
    )
    output["loss"].backward()

    # Then the loss is finite and gradients exist
    with then("compile keeps the DCA path healthy"):
        assert torch.isfinite(output["loss"])
        assert model.encoder.grn[0].proj.weight.grad is not None
