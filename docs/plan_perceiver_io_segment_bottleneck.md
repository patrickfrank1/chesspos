# Plan: Segment-Aware Perceiver IO

- **Status:** Implemented (core model + training/evaluation integration)
- **Date:** 2026-11-09
- **Branch:** `feature/perceiver-io-segment-bottleneck`
- **Scope:** Add a Perceiver IO model with a segment-aware latent bottleneck
  alongside the existing transformer model, including training and evaluation
  integration.
- **Out of scope:** Replacing or deleting the existing transformer, changing the
  token-stream format, changing the dataset pipeline, vector-store work.

Implemented in `src/modeling/shared/`, `src/modeling/encoder/`,
`src/modeling/perceiver/`, and `src/modeling/factory.py`, with BDD coverage in
`test/unit/test_perceiver.py`. The legacy transformer remains the default
architecture; Perceiver IO is selected with `model_arch: "perceiver_io"`.

## 1. Goal

Add a foundation model that:

- produces **context-aware** position embeddings over a sequence of board
  positions;
- handles both a single-position sequence and a multi-position window with the
  **same network and weights**;
- keeps content and state information separate so content embeddings cannot
  read turn/castle/en-passant tokens;
- exposes a decoder that can reconstruct per-square piece information from the
  bottleneck;
- remains usable by the existing MLM, InfoNCE, invariance, VICReg, hard-negative,
  retrieval, and observability code paths.

## 2. Approved design decisions

1. **New packages, legacy preserved.** Add `src/modeling/shared/`,
   `src/modeling/encoder/`, and `src/modeling/perceiver/`. Keep
   `src/modeling/transformer.py` working and make it the legacy path.
2. **Separate content and state latent banks.** No cross-bank attention.
   Content latents never read state tokens or state latents; state latents may
   attend the whole segment.
3. **Cross-segment self-attention within each bank.** Content latents interact
   across segments; state latents interact across segments. They never interact
   with each other.
4. **Contextual retrieval embeddings.** `encode_content` and the decoder read
   the content latents **after** cross-segment self-attention.
5. **Same network for `N=1` and `N>1`.** Cross-segment attention is grouped and
   masked by valid segment ids. With one segment it attends only to that
   segment (no foreign context); with multiple segments it attends across them.
6. **Latent counts:** `n_content_latents=16`, `n_state_latents=1`.
7. **Readout:** learned attention pooling / dedicated readout query, not
   unweighted mean pooling.
8. **Reconstruction loss:** implement the decoder reconstruction path now,
   controlled by `decoder_reconstruction_weight` and defaulted to `0.0`.
9. **Full integration:** model factory, checkpoint save/load/resume, training
   CLI/YAML, objectives, evaluation loading, and tests.
10. **Shared extraction/refactor:** move reusable layers, embeddings, masks, and
    grouping helpers into `src/modeling/shared/`, guarded by behavior-preserving
    equivalence tests against the legacy implementation.

## 3. Architecture

### 3.1 Input contract

The existing packed-window contract is unchanged:

- `tokens`: `[batch, seq]`
- `segment_ids`: `[batch, seq]` — one id per board position
- `padding_mask`: `[batch, seq]`
- each segment contains 64 square tokens followed by 3 state tokens:
  turn, castling, en-passant.

A sequence may contain one segment (single position) or many segments
(game/window). The same model handles both.

### 3.2 Encoder (`src/modeling/encoder/`)

- Token embedding = token embedding + segment embedding + slot embedding,
  reusing the shared `slot_indices` helper.
- `n_encoder_layers` transformer blocks with **segment-grouped varlen attention**.
- Preserve the isolation invariant at every layer:
  - content queries attend only to content keys in the same segment;
  - state/CLS/SEP queries may attend the whole segment.
- Output: per-token encoded states `H`, without cross-segment mixing and
  without content reading state tokens.

### 3.3 Segment-aware latent bottleneck (`src/modeling/perceiver/latent.py`)

For each segment:

1. **Content bank**
   - `n_content_latents` learned latent queries.
   - Cross-attend to that segment's **content-token states only**.
   - `n_local_latent_layers` latent self-attention blocks inside the segment.
2. **State bank**
   - `n_state_latents` learned latent queries.
   - Cross-attend to that segment's **full token states** (content + state).
   - `n_local_latent_layers` latent self-attention blocks inside the segment.

Then, for each bank independently:

3. **Cross-segment self-attention**
   - Content latents from all valid segments in the row attend to each other.
   - State latents from all valid segments in the row attend to each other.
   - Grouped/masked by segment validity; no cross-bank attention.
   - Uses segment-order information so the model knows position order.
   - `n_cross_segment_layers` layers.

No content latent ever reads a state token or state latent. No state latent
reads another segment's state unless it is part of the state bank's
cross-segment attention (which is allowed).

### 3.4 Decoder (`src/modeling/perceiver/decoder.py`)

Per segment, `67` output queries:

- 64 square-slot queries, using the shared slot embeddings;
- 3 state-slot queries.

Each query cross-attends to its segment's latents:

- content slot queries → contextual content latents only;
- state slot queries → contextual state latents only.

Then `n_decoder_layers` feed-forward/self-attention blocks produce per-token
hidden states.

The decoder output is used for:

- MLM logits at masked token positions;
- optional visible-content reconstruction (weight default `0.0`);
- per-token piece extraction, e.g. a linear probe on decoded hidden states.

### 3.5 Pooled embeddings and readout

- `encode_content` → contextual content readout per segment.
- `encode_state` → contextual state readout per segment.
- Readout is a learned attention pool (a readout query over the segment's
  content latents), not a mean.
- `encode_views` → `(content, state, segment_mask)` with the same shapes and
  semantics as the legacy model: `[batch, max_segments, d_model]`.
- `encode` → content embedding.
- `embed` → content embedding, left unnormalized here; the existing evaluation
  path L2-normalizes.

Because content latents are contextual, a position's retrieval embedding may
depend on the other positions in its window. This is intentional. Evaluation
must record whether queries are embedded alone or in context.

### 3.6 Model interface

`PerceiverIOModel` implements the same duck-typed interface as
`MaskedStreamTransformer`:

```python
forward(tokens, segment_ids, padding_mask, targets=None, weights=None)
encode_views(tokens, segment_ids, padding_mask)
encode_content(tokens, segment_ids, padding_mask)
encode_state(tokens, segment_ids, padding_mask)
encode(tokens, segment_ids, padding_mask)
embed(tokens, segment_ids, padding_mask)
content_segment_mask(tokens, segment_ids, padding_mask)
head
aux_heads
model_config
```

`forward` returns `hidden_states` and, when `targets` is provided, `loss`.
`hidden_states` are the decoded per-token states so the existing shared MLM head
and loss work unchanged.

### 3.7 Configuration

```python
@dataclass
class PerceiverIOConfig:
    arch: str = "perceiver_io"
    vocab_size: int = 52
    d_model: int = 512
    n_heads: int = 16
    d_ff: int = 2048
    dropout: float = 0.1
    max_seq_len: int = 2048
    max_segments: int = 128

    n_encoder_layers: int = 8
    n_content_latents: int = 16
    n_state_latents: int = 1
    n_local_latent_layers: int = 2
    n_cross_segment_layers: int = 2
    n_decoder_layers: int = 2

    content_readout: str = "attention"  # "attention" | "first_latent"
    decoder_reconstruction_weight: float = 0.0

    content_pooling: bool = True  # compatibility flag
    aux_heads: bool = False
    phase_classes: int = 3
```

Validation:

- `d_model % n_heads == 0`
- `n_content_latents > 0`, `n_state_latents > 0`
- `n_*_layers >= 0`
- `max_seq_len > 0`, `max_segments > 0`
- `content_readout` in the allowed set

## 4. Directory layout

```text
src/modeling/
  __init__.py                  # export legacy + new models/configs
  factory.py                   # build_model(config) dispatch
  transformer.py               # existing legacy model (kept)
  varlen.py                    # existing varlen helpers (kept)
  masking.py                   # existing masking (kept)

  shared/
    __init__.py
    attention.py               # varlen + segment-grouped cross-attention
    embeddings.py              # token/segment/slot embeddings, slot_indices
    layers.py                  # MLP, FeedForward, transformer/cross-attn blocks
    masks.py                   # content/state masks, segment offsets
    pooling.py                 # segment pooling helpers

  encoder/
    __init__.py
    config.py                  # EncoderConfig
    token_encoder.py           # embeddings + isolated per-segment encoder
    isolation.py               # content/state attention grouping

  perceiver/
    __init__.py
    config.py                  # PerceiverIOConfig
    cross_attention.py         # latent Q -> token K/V, grouped by segment
    latent.py                  # content/state latent banks + cross-segment attn
    decoder.py                 # slot queries -> latents -> per-token states
    model.py                   # PerceiverIOModel (legacy-compatible interface)
```

The shared extraction is behavior-preserving. Existing tests must continue to
pass, and new equivalence tests must compare shared helpers against the legacy
implementation.

## 5. Integration plan

### 5.1 Model factory and checkpoints

- Add `src/modeling/factory.py`:
  - `build_model(config)` returns `PerceiverIOModel` when
    `arch == "perceiver_io"`;
  - otherwise returns `MaskedStreamTransformer`.
- Old checkpoints have no `arch` field → default to the legacy transformer.
- New checkpoints store `arch` in `model_config`.
- Update model construction and resume paths:
  - `src/run/train_masked_transformer.py`
  - `src/training/ray_train.py`
  - `src/run/train_ray.py`
  - `src/evaluation/observability/loading.py`
  - `src/run/eval_model.py`
  - `src/run/classify_position.py`

### 5.2 CLI and YAML

Add:

- `--model-arch {transformer,perceiver_io}` to the foundation trainer;
- `model_arch` in `configs/training.yaml`;
- architecture-specific overrides for latent counts/layers/readout/reconstruction
  weight.

Resume must rebuild the same architecture from the checkpoint config.

### 5.3 Objectives and evaluation

The existing `compute_training_loss` path should work through the compatible
interface:

- MLM uses decoded hidden states and the shared head;
- InfoNCE/invariance/VICReg use `encode_views`;
- alone views already produce single-segment sequences (`N=1`);
- window views produce multi-segment sequences (`N>1`);
- hard negatives and retrieval use `embed` / `encode_content`.

Add optional reconstruction loss:

- `decoder_reconstruction_weight=0.0` by default;
- when positive, the decoder reconstructs original visible tokens (or all
  slots) in addition to MLM;
- no behavior change when the weight is zero.

### 5.4 Puzzle classifier

The puzzle classifier currently instantiates `MaskedStreamTransformer`. Keep
that default. Supporting a frozen Perceiver encoder is optional follow-up work.

## 6. BDD test plan (write first)

All tests use the Given/When/Then structure. Phase 1 stops after these tests are
written, for user review.

### 6.1 Shared extraction

- `test_slot_indices_match_legacy`
- `test_segment_grouped_attention_never_crosses_segments`
- `test_segment_offsets_handle_empty_segments`
- `test_shared_pooling_matches_legacy`
- `test_shared_content_isolation_matches_legacy`

### 6.2 Encoder

- `test_isolated_encoder_matches_legacy_content_isolation`
- `test_encoder_content_states_ignore_state_tokens`
- `test_encoder_state_states_see_state_tokens`
- `test_encoder_handles_single_and_multiple_segments`

### 6.3 Perceiver model

- `test_forward_shapes_for_packed_multiple_segments`
- `test_content_embedding_is_invariant_to_turn_castle_ep`
- `test_state_embedding_reacts_to_turn_castle_ep`
- `test_content_latents_do_not_read_state_latents`
- `test_cross_segment_content_attention_changes_contextual_latents`
- `test_cross_segment_state_attention_changes_contextual_state_latents`
- `test_single_segment_has_no_foreign_context`
- `test_decoder_content_logits_ignore_state_tokens`
- `test_decoder_state_logits_react_to_state_tokens`
- `test_decoder_logits_depend_on_piece_at_square`
- `test_mlm_loss_ignores_unmasked_tokens`
- `test_reconstruction_loss_is_zero_when_weight_zero`
- `test_reconstruction_loss_is_positive_when_enabled`
- `test_encode_views_shapes_and_segment_mask`
- `test_config_validation`

### 6.4 Factory and integration

- `test_model_factory_selects_arch`
- `test_legacy_checkpoint_without_arch_loads_as_transformer`
- `test_perceiver_checkpoint_roundtrip`
- `test_compute_training_loss_runs_with_perceiver` (finite loss, gradients flow)
- `test_train_cli_accepts_model_arch`
- `test_evaluate_loading_dispatches_arch`
- `test_resume_rebuilds_perceiver_architecture`
- short smoke test: one or two optimization steps on a tiny batch, finite loss,
  finite normalized `embed()` output.

### 6.5 Leakage check

Add an evaluation-level side-to-move probe comparison for the content embedding.
Ordered cross-segment content attention can indirectly reveal turn through board
evolution even though no state token is read. The model-level tests assert
direct state isolation; the probe monitors indirect leakage during training.

## 7. Phases

1. **Phase 0 — Baseline**
   - Run `uv run pytest` and `uv run ruff check` on `main`/branch before changes.
2. **Phase 1 — Failing BDD tests**
   - Write the tests in Section 6.
   - Stop for user review.
3. **Phase 2 — Shared extraction and encoder**
   - Implement `shared/` and `encoder/`.
   - Keep legacy tests green; add equivalence tests.
4. **Phase 3 — Perceiver**
   - Implement config, cross-attention, latent banks, decoder, model.
   - Make Phase 1 tests pass.
5. **Phase 4 — Full integration**
   - Factory, checkpoint save/load/resume, CLI/YAML, objectives, evaluation.
   - Make integration tests pass.
6. **Phase 5 — Reconstruction loss**
   - Implement the disabled-by-default reconstruction objective and tests.
7. **Phase 6 — Verification**
   - `uv run ruff format && uv run ruff check`
   - `uv run pytest`
   - Short CPU smoke training run and retrieval embedding smoke test.

## 8. Acceptance criteria

- Existing transformer behavior and checkpoints remain valid.
- `arch="perceiver_io"` checkpoints can be trained, saved, resumed, and loaded.
- The same model runs with one segment and with multiple segments.
- Content embeddings and content decoder logits are exactly invariant to
  turn/castle/en-passant token changes.
- State embeddings and state logits react to state tokens.
- Content and state latent banks never attend to each other.
- Cross-segment attention changes contextual latents when multiple segments are
  present and is a no-op across segments when only one segment is present.
- The decoder produces per-square logits from which piece information can be
  probed.
- Reconstruction loss is present but disabled by default and does not change
  behavior at weight zero.
- The full test suite and ruff pass.

## 9. Risks and open considerations

- **Indirect turn leakage:** cross-segment content attention can infer turn from
  ordered board evolution. Monitor with the side-to-move probe; if it leaks,
  options are to remove segment ordering from content cross-attention or
  restrict context to the decoder only.
- **Context-dependent retrieval:** the same board can embed differently alone
  vs inside a game window. This is intended; evaluation must state which context
  was used.
- **Single-segment masking:** cross-segment attention must correctly reduce to
  within-segment behavior for `N=1`; this is explicitly tested.
- **Shared refactor risk:** extracting legacy helpers can change numerics if not
  done carefully. Equivalence tests guard against this.
- **Memory/compute:** cross-segment attention operates on a small number of
  latents per segment, so it should remain cheap relative to token attention.
- **Checkpoint compatibility:** old checkpoints lack `arch`; the factory must
  default to the legacy transformer.

## 10. Deliverables

- `src/modeling/shared/`, `src/modeling/encoder/`, `src/modeling/perceiver/`
- `src/modeling/factory.py` and updated exports
- Training/evaluation integration for `arch="perceiver_io"`
- BDD tests described in Section 6
- Updated plan/status notes after test review and implementation
