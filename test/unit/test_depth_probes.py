from __future__ import annotations

import chess
import numpy as np
import torch

from bdd import then
from src.dataset.token_stream import (
    PAD,
    SEP,
    VOCAB_SIZE,
    TokenStreamEncoder,
    pack_stream,
)
from src.evaluation.probes import (
    accuracy_counts,
    counts_to_metrics,
    depth_accuracy,
    encoder_states,
    latent_from_state,
    pca_basis,
    residual_drift,
    truncate,
)
from src.modeling import autoencoder as ae
from src.modeling.autoencoder import AttentionAutoencoderConfig
from src.modeling.masking import MaskingConfig, mask_stream


def tiny_model(n_layers: int = 2) -> ae.AttentionAutoencoderModel:
    config = AttentionAutoencoderConfig(
        vocab_size=VOCAB_SIZE,
        d_model=16,
        n_heads=2,
        n_layers=n_layers,
        n_decoder_layers=1,
        d_ff=32,
        dropout=0.0,
        max_seq_len=1024,
        max_segments=8,
        latent_dim=8,
    )
    torch.manual_seed(0)
    return ae.build_attention_autoencoder(config).eval()


def probe_batches(count: int = 2) -> list[dict[str, torch.Tensor]]:
    encoder = TokenStreamEncoder()
    board = chess.Board()
    batches = []
    for game in range(count):
        segments = []
        for _ in range(3):
            segments.append(encoder.encode(board))
            board.push(list(board.legal_moves)[game])
        tokens_np = pack_stream(segments)
        masking = MaskingConfig(
            p_rect=1.0, p_random=0.0, p_board=0.0, p_span=0.0, p_skip_gram=0.0
        )
        tokens, targets, _ = mask_stream(
            tokens_np, masking, np.random.default_rng(game)
        )
        segment_ids = np.cumsum(tokens == SEP)
        as_tensor = torch.from_numpy
        batches.append(
            {
                "tokens": as_tensor(tokens.astype(np.int64))[None],
                "segment_ids": as_tensor(segment_ids.astype(np.int64))[None],
                "targets": as_tensor(targets.astype(np.int64))[None],
            }
        )
    for batch in batches:
        batch["padding_mask"] = batch["tokens"].ne(PAD)
    return batches


class Passthrough(torch.nn.Module):
    def forward(self, x, cu_seqlens, max_seqlen):
        return x


def test_encoder_states_pooling_matches_model_encode_latent():
    # Given a tiny autoencoder and a packed window batch
    model = tiny_model()
    batch = probe_batches(1)[0]

    # When the last captured state is pooled and the model encodes directly
    with then("both latents agree"):
        states = encoder_states(
            model, batch["tokens"], batch["segment_ids"], batch["padding_mask"]
        )
        pooled = latent_from_state(model, states[-1], batch["padding_mask"])
        direct = model.encode_latent(
            batch["tokens"], batch["segment_ids"], batch["padding_mask"]
        )
        assert len(states) == model.config.n_layers + 1
        assert torch.allclose(pooled, direct, atol=1e-6)


def test_skipping_no_block_matches_the_full_stack():
    # Given a tiny autoencoder and a batch
    model = tiny_model()
    batch = probe_batches(1)[0]

    # When one block is skipped
    with then("the skipped state differs from the full stack"):
        states = encoder_states(
            model, batch["tokens"], batch["segment_ids"], batch["padding_mask"]
        )
        skipped = encoder_states(
            model,
            batch["tokens"],
            batch["segment_ids"],
            batch["padding_mask"],
            skip=0,
        )
        assert len(skipped) == len(states) - 1
        assert not torch.allclose(skipped[-1], states[-1])


def test_residual_drift_is_zero_for_an_identity_block():
    # Given a model whose first block is replaced by a passthrough
    model = tiny_model()
    model.encoder.blocks[0] = Passthrough()
    batch = probe_batches(1)[0]

    # When the drift is measured
    rows = residual_drift(model, [batch])

    # Then the passthrough block looks like the identity
    with then("block 1 neither moves nor rotates the state"):
        assert rows[0]["rel_delta_mean"] == 0.0
        assert abs(rows[0]["cos_mean"] - 1.0) < 1e-5
        assert len(rows) == model.config.n_layers


def test_depth_accuracy_at_full_depth_matches_the_untruncated_path():
    # Given a tiny autoencoder and paired batches
    model = tiny_model()
    batches = probe_batches(2)

    # When the accuracy is measured at full depth and without truncation
    full = depth_accuracy(model, batches)
    truncated = depth_accuracy(model, batches, depth=model.config.n_layers)

    # Then the two paths agree exactly
    with then("full depth is equivalent to no truncation"):
        assert full["masked_acc"] == truncated["masked_acc"]
        assert 0.0 <= full["masked_acc"] <= 1.0


def test_truncate_at_full_rank_is_the_identity():
    # Given a random embedding matrix and its PCA basis
    torch.manual_seed(0)
    matrix = torch.randn(64, 8)
    mean, components, _ = pca_basis(matrix)

    # When truncated at the full rank
    restored = truncate(matrix, mean, components, 8)

    # Then the matrix is returned unchanged
    with then("truncation is lossless"):
        assert torch.allclose(restored, matrix, atol=1e-5)


def test_accuracy_counts_split_piece_and_scalar_tokens():
    # Given predictions whose targets include content and scalar classes
    targets = torch.tensor([5, 6, 17, 20])
    predictions = torch.tensor([5, 7, 17, 21])

    # When the counts are accumulated
    counts = accuracy_counts(predictions, targets)
    metrics = counts_to_metrics(counts)

    # Then content and scalar accuracies are tracked separately
    with then("each class family has its own accuracy"):
        assert counts["total"] == 4
        assert counts["correct"] == 2
        assert metrics["piece_acc"] == 0.5
        assert metrics["scalar_acc"] == 0.5
        assert metrics["turn_acc"] == 1.0
        assert metrics["masked_acc"] == 0.5
