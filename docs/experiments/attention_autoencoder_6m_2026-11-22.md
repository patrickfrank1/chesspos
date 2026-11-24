# Attention autoencoder: one 512-d bottleneck per packed window

**Runs:** `run_ae_6m_100k` (stopped at step 42,000 on 2026-11-22);
`run_ae_20m_100k` (stopped at step 81,600 on 2026-11-23)
**Architecture:** `attention_autoencoder`
**Parameters:** 6,068,276 (6M) / 20.39M (20M)
**Artifacts:** `hf://buckets/patrickfrank1/chesspos-checkpoints/run_ae_6m_100k/`
and `…/run_ae_20m_100k/`

## Status

The run was **stopped by decision at step 42,000** (of 100,000). Final
checkpoint: `last.pt` at step 42,000; best `val_loss` **0.7415** at step
42,000, masked-token accuracy **0.6499**. Training loss (weighted) was 0.2317.
The pod was left running and idle for the next experiment.

Stopping was not driven by a failure. Validation had plateaued: from step
28,000 onward `val_acc` stayed in a 0.636–0.651 band and `val_loss` in a
0.74–0.78 band, while the weighted training loss kept creeping down. A
post-mortem (below) located the plateau in the combination of the loss
weighting and the 512-d bottleneck, not in the data or the optimisation. It
also changes the plan for the next run: a 20M-parameter model with a wider
latent and a rebalanced loss, proposed at the end of this report.

The **20M follow-up** then ran to step 81,600 and was **stopped by decision on
2026-11-23**. Reconstruction continued to improve slowly (`val_loss` 0.5173 at
60k → **0.5022** at 80k, `val_acc` 0.9055 → 0.9093), while the embedding
geometry plateaued: on a fixed 4,096-position pool `test_r2_material` was
**0.745** at 56k and **0.750** at 80k (overlapping CIs), against **0.463**
for the 6M/512-d model at the same protocol. The 1,024-position training
probes that suggested a decline at 80k were sampling noise. More steps
reconstruct empty squares and state tokens better; they do not move the
material axis further. The last checkpoint is step 81,000; best `val_loss`
0.5022 at step 80,000.

This report records the design, the implementation surface, the training
configuration, the measurements taken during the run, the plateau post-mortem,
the token-weighting analysis, the Chinchilla sizing, and the follow-up work —
including the InfoNCE ablation still queued.

## TL;DR

- One 512-d vector summarises an entire packed window (~19 positions); a
  latent-only decoder reconstructs every non-pad token against the unmasked
  gold sequence. There is no token-embedding path into the decoder, so the
  bottleneck is the only channel from input to output.
- Full-reconstruction validation loss fell from **1.5685 (2k)** to **0.7415
  (42k)**; masked-token accuracy rose from **0.325** to **0.650** and then
  flattened from ~28k onward.
- **87.8% of all remaining token errors are empty squares** (64.1% of masked
  tokens, recall 0.526; pieces 0.877, state 0.978). A constant "always empty"
  predictor scores 0.641, so the model was only 1.3 points above trivial on the
  headline metric.
- The model *knows* which squares are empty (AUC 0.932 separating empty from
  piece) but the frequency-weighted loss makes it conservative. Adding `+3.5`
  to the EMPTY logit at eval lifts accuracy **0.648 → 0.861** with no
  retraining: this is a loss/metric mismatch, not lost information.
- Empty squares are 4.9% of the full loss weight under `1/p` cap 100; pieces
  are 59.7%, turn/castle/ep 23.0%, and CLS/SEP 12.4% (the last is an artifact:
  the unigram cache excludes them, so their smoothed weight hits the cap).
- The 512-d bottleneck is a real secondary limit: reconstructing the *same*
  position with clean input falls from **0.929 (alone)** to **0.753 (30
  positions)**, and the loss is almost entirely empty-square recall
  (0.880 → 0.595). Pieces barely move (0.992 → 0.950).
- Chinchilla sizing: the train corpus is ~177B tokens (2.60B positions × 68.05
  tokens), so compute-optimal is ~8.9B parameters, not 6M. At 6.07M the
  Chinchilla token budget is only ~121M tokens (1.8M positions, 0.07% of the
  corpus); the run used ~30B tokens, ~250× past the ratio.
- The apparent "train loss 0.23 vs val loss 0.74" gap is still a **metric
  definition** difference, not overfitting: the trainer logs the
  frequency-weighted CE, evaluation logs the unweighted mean (and over all
  tokens, not only masked ones). Train-split and test-split windows score
  within 0.0002 nats of each other.
- The bottleneck is load-bearing: decoding with another window's latent
  roughly doubles the loss (1.26 → 2.44 nats at step 4k), and the same
  bottleneck produces the window and alone views.
- The window latent already sits close to its own positions and far from other
  games (cos 0.52 vs 0.04) without any contrastive term. That emergent geometry
  motivates the planned NCE ablation.
- The **20M/1024-d follow-up** (stopped at 81,600) lifted `val_acc` to 0.9093
  and, on a 4,096-position pool, `test_r2_material` from **0.463 (6M/512-d)** to
  **0.750**, plateauing from ~56k. Capacity fixed the material axis; the
  apparent late-run decline in the 1,024-position probes was sampling noise.
  See *Full evaluation of the 20M bottleneck*.
- **Frozen downstream probes** place the bottleneck ahead of every earlier
  checkpoint: held-out score R² **0.051 / AUC 0.645** (previous frozen best
  0.023 / 0.587; material baseline 0.068 / 0.655) and sacrifice PR-AUC
  **0.288 / AUROC 0.745** (previous frozen best 0.203 / 0.697). One epoch is
  optimal for the frozen score head.

## Architecture

Packed windows are `CLS + segment + SEP + ...` streams (67 tokens per position:
64 squares, turn, castling, en-passant) padded to 2048.

| Component | Detail |
|---|---|
| Encoder embedding | token + segment + within-segment slot embeddings, `d_model=256` |
| Encoder | 4 pre-norm transformer blocks, 8 heads, `d_ff=1024`, dropout 0.1 |
| Attention | flattened valid-token stream via `cu_seqlens`, so CUDA runs flash-attn `varlen` with no S×S mask; padded tail scattered back afterwards |
| Bottleneck | single learned query scores every valid token (`key·q/√d`), sigmoid-gated values, softmax over valid tokens, weighted sum → `Linear(256→512)` → LayerNorm |
| Decoder | broadcast latent + segment/slot embeddings (no token embeddings), 3 blocks, 8 heads, `d_ff=1024` |
| Head | `Linear(256→52)` |
| Latent | one 512-d vector per packed window — this is also the retrieval embedding |

The decoder deliberately never sees input token identities. Without that
restriction a decoder could copy visible tokens around the bottleneck and the
reconstruction objective would say nothing about the latent.

### Loss

`L = Σ_i w_i · CE(logits_i, gold_i) / Σ_i w_i` over **every non-pad position**,
where `gold = where(targets != IGNORE, targets, tokens)` reconstructs the
unmasked sequence. Weights are capped inverse unigram frequencies
(`--freq-cap 100`, most frequent token = 1.0). Nothing else is active: VICReg,
InfoNCE, auxiliary heads and hard-negative mining are all disabled for this
architecture (it has no per-segment views to contrast).

## Implementation surface

| File | Change |
|---|---|
| `src/modeling/autoencoder/config.py` | `AttentionAutoencoderConfig`; defaults are now the deep-narrow recipe (the 6M run used explicit sizes) |
| `src/modeling/autoencoder/encoder.py` | packed-stream encoder, varlen blocks |
| `src/modeling/autoencoder/bottleneck.py` | gated attention pooling → 512-d |
| `src/modeling/autoencoder/decoder.py` | latent-conditioned decoder, position-only inputs |
| `src/modeling/autoencoder/model.py` | loss on all non-pad tokens, `embed() -> [B, 1, 512]` |
| `src/modeling/shared/layers.py` | `VarlenTransformerLayer` (flash-compatible block) |
| `src/modeling/factory.py` | `attention_autoencoder` dispatch + checkpoint config rebuild |
| `src/run/train_masked_transformer.py` | `--model-arch attention_autoencoder`, `--latent-dim`, arch presets, reconstruction-only objective, `view_helpers()`, resume adopts checkpoint arch |
| `src/training/window_dataset.py` | `full_reconstruction_weights` (supervise visible positions too) |
| `test/unit/test_attention_autoencoder.py` | 23 model/dataset/objective tests |
| `test/unit/test_train_masked_cli.py` | 9 CLI/wiring tests |

`embed()` returns `[B, 1, 512]` so the existing retrieval harness
(`embed_tokens`) works unchanged: at evaluation each row is a single position,
and the bottleneck simply pools that one position.

## Training configuration

| Setting | Value |
|---|---|
| Hardware | RTX 6000 Ada 48 GB (vast.ai), ~$0.54–0.60/h |
| Software | torch 2.8.0+cu128, flash-attn 2.8.3.post2, `torch.compile(dynamic=True)`, bf16 autocast |
| Batch | 240 windows × 2048 tokens |
| Optimizer | AdamW, lr 2e-4, warmup 2,000, cosine to 0 over 100,000 steps, weight decay 0.01, grad clip 1.0 |
| Steps | 100,000 (≈ 24 M windows ≈ 446 M positions ≈ 17% of the corpus) |
| Masking | 80% rectangle / 20% skip-gram, balanced modes |
| Data | `patrickfrank1/chess-positions`, lazy shard streaming (`--stream-shards`), 8 workers, `HF_HUB_DISABLE_XET=1` |
| Weighting | inverse unigram counts built from shard `.meta.json` sidecars (443 shards, 174.3 B content tokens) |
| Eval | every 2,000 steps, 20 batches (4,800 windows) from an 8,192-game pool, deterministic |
| Checkpoints | every 1,000 steps, uploaded to the HF bucket, watchdog auto-resumes from `last.pt` |

Batch size was chosen from a memory probe on the same hardware: dense
worst-case (every token valid) peak allocation was 39.4 GB at batch 256 and
37.0 GB at batch 240, against 47.4 GB usable. Real windows average ~1,245 valid
tokens, so steady-state usage is **27–28 GB** — batch 240 keeps ~20% headroom
in the worst case and ~40% in the typical case. Compile added ~26% throughput
(646k → 816k padded slots/s in the probe).

## Results (stopped at 42,000)

Validation (unweighted full-reconstruction loss and masked-token accuracy):

| Step | val_loss | val_acc | | Step | val_loss | val_acc |
|---:|---:|---:|---|---:|---:|---:|
| 2,000 | 1.5685 | 0.325 | | 20,000 | 0.8911 | 0.587 |
| 4,000 | 1.2498 | 0.445 | | 22,000 | 0.8614 | 0.602 |
| 6,000 | 1.1143 | 0.501 | | 24,000 | 0.8266 | 0.617 |
| 8,000 | 1.0250 | 0.536 | | 26,000 | 0.8185 | 0.620 |
| 10,000 | 1.0021 | 0.547 | | 28,000 | 0.7788 | 0.640 |
| 12,000 | 0.9795 | 0.556 | | 30,000 | 0.7801 | 0.638 |
| 14,000 | 0.9277 | 0.577 | | 32,000 | 0.7936 | 0.630 |
| 16,000 | 0.9094 | 0.582 | | 34,000 | **0.7534** | **0.647** |
| 18,000 | 0.8852 | 0.588 | | | | |

The last four evaluations show the plateau:

| Step | val_loss | val_acc | | Step | val_loss | val_acc |
|---:|---:|---:|---|---:|---:|---:|
| 36,000 | 0.7726 | 0.640 | | 40,000 | 0.7501 | 0.651 |
| 38,000 | 0.7758 | 0.636 | | 42,000 | **0.7415** | **0.650** |

Training loss (weighted) at step 42,000: **0.2317**, lr 1.28e-4, GPU ~100%.

## Measurements taken during the run

### 1. Why train and validation loss look far apart

Both are the same objective — mean cross entropy over all non-pad tokens — but
they differ in weighting and sample:

| Quantity (checkpoint step 4,000–5,000) | Value |
|---|---|
| Unweighted mean CE (= `val_loss`) | **1.193** |
| Unigram-weighted mean CE (= logged `loss`) | **0.469** |
| Held-out **test**-split windows (mixed modes) | 1.2483 |
| **Train**-split windows (mixed modes) | 1.2481 |
| Masked positions / visible positions (unweighted) | 1.42 / 1.07 |

Train and test splits differ by 0.0002 nats, so there is no held-out penalty.
The whole gap is the weighting. Per-token breakdown:

| Gold token | Share | Weight | Mean CE |
|---|---:|---:|---:|
| `EMPTY` | 61.1% | 1.0 | 1.65 |
| pieces (12 ids) | ~30% | 7–84 | 0.46–0.75 |
| `SEP` / `CLS` | 1.6% | 100 | 0.0001 / 0.0007 |
| turn / castling / en-passant | 2.6% | 60–84 | 0.06–0.68 |

Empty squares are the hard part and carry weight 1; the frequent-but-easy
structural tokens carry up to 100×. Weighted training loss is therefore a
*token-class-weighted* metric, while `val_loss` is a plain token mean — they
are not comparable in level, only in trend. The trainer's rolling 50-step
average (12k windows, after the update) and validation's 4,800 fixed windows
add a little noise on top, and dropout (on in training, off in evaluation)
works in the opposite direction.

### 2. The bottleneck is used, not ignored

Latent-swap control at step 4,000 (decode a window with its own latent vs. a
latent rolled from another window in the batch):

| Decode condition | Unweighted CE |
|---|---:|
| own latent | 1.26 |
| swapped latent | 2.44 |

A model that ignored the latent would show no difference. This is the cheapest
collapse detector for this architecture and should be logged periodically.

### 3. Emergent view geometry (step 34,000)

Cosine similarity between the 512-d window latent and **alone-view** latents
(each position encoded in isolation — exactly what `embed()` does at eval):

| Pair | mean | p10 | p50 | p90 |
|---|---:|---:|---:|---:|
| window ↔ its **own** positions | **0.521** | 0.156 | 0.566 | 0.793 |
| window ↔ **other-game** positions | **0.042** | −0.147 | 0.034 | 0.235 |
| position ↔ position, same window | 0.600 | 0.327 | 0.641 | 0.823 |

Structurally this is already what a contrastive objective would ask for, and it
comes for free: the same bottleneck module produces both views. Caution for
future work — the same measurement using **in-context mean-pooled** hidden
states (no extra forward pass) gives 0.035, i.e. an unrelated direction. Any
"alone view" objective must use a real single-position pass.

## Post-mortem: why validation accuracy plateaued

Analysed `best.pt` at step 40,000 on 1,024 held-out test windows (32 batches of
32), with the training masking config. Diagnostics: `diag_plateau.py`,
`diag_plateau2.py`, `diag_plateau3.py`, `diag_plateau4.py`,
`diag_weighting.py` on the pod.

### 1. Empty squares are 88% of the remaining errors

| Gold class | Masked tokens (share) | Accuracy | Wrong | Share of errors |
|---|---:|---:|---:|---:|
| **empty** | 215,814 (64.1%) | **0.526** | 102,252 | **87.8%** |
| pieces | 113,996 (33.9%) | 0.877 | 13,995 | 12.0% |
| turn/castle/ep | 6,853 (2.0%) | 0.978 | 153 | 0.1% |

The model predicts `EMPTY` on only **34.8%** of masked squares while 64.1% of
them are empty. Precision is 0.970, recall 0.526 — it is conservative, not
confused. A constant "always empty" predictor scores **0.641**; the model
scores 0.654, i.e. 1.3 points above trivial on the headline metric. Piece
identity is essentially solved at the token level (queen is the weakest at
0.804; pawns 0.895, king 0.934).

### 2. The information is in the latent; the decision rule is not

Occupancy is highly recoverable from the checkpoint:

| Quantity | Value |
|---|---:|
| AUC of p(EMPTY), empty vs piece | **0.932** |
| mean p(EMPTY) on gold-empty | 0.449 |
| mean p(EMPTY) on gold-piece | 0.047 |
| AUC, rect-masked / skip-gram | 0.909 / 0.956 |

Adding a constant to the `EMPTY` logit at eval (masked squares, step 40k):

| EMPTY bias | accuracy | empty recall | empty precision | unweighted CE | weighted CE |
|---:|---:|---:|---:|---:|---:|
| 0.0 | 0.648 | 0.526 | 0.970 | 0.948 | **0.496** |
| 1.0 | 0.764 | 0.718 | 0.944 | 0.660 | 0.527 |
| 2.0 | 0.835 | 0.853 | 0.913 | 0.487 | 0.625 |
| 3.0 | 0.861 | 0.929 | 0.878 | **0.422** | 0.809 |
| 3.5 | **0.861** | 0.952 | 0.859 | 0.424 | 0.938 |
| 4.0 | 0.855 | 0.968 | 0.840 | 0.447 | 1.092 |

The weighted CE is minimised at bias 0 and rises monotonically; unweighted
accuracy and CE are minimised near +3.2. The two objectives genuinely conflict,
and the model was optimising the first one.

### 3. Where the loss weight goes

Shares of the supervised loss (all non-pad tokens; masked-only shares are more
extreme):

| Group | Token share | Weight share | Masked-token weight share |
|---|---:|---:|---:|
| empty | 62.2% | **4.9%** | 6.8% |
| pieces | 34.6% | **59.7%** | 79.0% |
| turn/castle/ep | 4.4% | 23.0% | 14.2% |
| CLS/SEP | 1.6% | 12.4% | — |

`CLS`/`SEP` are absent from the unigram cache, so their smoothed inverse
frequency hits the cap (weight 100) even though the model already predicts them
essentially perfectly. Together with the state tokens, **35% of the loss weight
sits on tokens at 0.978–1.000 accuracy.** That is the budget that should move
to empty squares.

### 4. Single positions vs sequences

Clean-input capacity sweep: reconstruct **the same position 0** while varying
how many consecutive positions share the latent (192 windows, gold sequences,
no `MASK` tokens):

| Context positions | accuracy | empty recall | piece accuracy | state |
|---:|---:|---:|---:|---:|
| 1 | **0.929** | **0.880** | 0.992 | 1.000 |
| 2 | 0.934 | 0.890 | 0.990 | 1.000 |
| 4 | 0.909 | 0.847 | 0.987 | 1.000 |
| 8 | 0.870 | 0.783 | 0.979 | 0.998 |
| 12 | 0.836 | 0.726 | 0.974 | 0.998 |
| 16 | 0.811 | 0.687 | 0.967 | 0.997 |
| 20 | 0.789 | 0.651 | 0.962 | 0.997 |
| 24 | 0.773 | 0.629 | 0.954 | 0.995 |
| 30 | **0.753** | **0.595** | 0.950 | 0.997 |

A single position is compressed almost perfectly by 512 floats. As the pack
grows, the degradation is **almost entirely occupancy recall**; piece identity
survives. This is the second, smaller cause of the plateau: the latent is at
its information limit for ~30 positions, and 1024 dimensions is the direct
remedy.

### 5. Masking mode, geometry, phase

- `rect` masking (same 30–50% square rectangle in every position, no copy path)
  scores 0.603 and accounts for **62.3% of all errors**; skip-gram scores 0.711
  and 37.7% of errors.
- Empty-square recall by file: queenside 0.657, kingside 0.649, **centre files
  c–f 0.430**. Worst squares: f3 0.261, f6 0.272, f7 0.277, g7 0.270, d5 0.294,
  e5 0.302, e4 0.317. Best: h3 0.814, a3 0.803, h1 0.795, h6 0.772.
- By phase: ply 0–19 0.840, 20–39 0.743, 40–59 0.654, **60–79 0.592**,
  80–99 0.600, 100–139 0.61–0.63.
- Per-piece accuracy: P 0.895, N 0.843, B 0.837, R 0.867, **Q 0.804**,
  K 0.934 (white; black within 0.02).

### 6. Interpretation

Three factors, in order of importance:

1. **Loss/metric mismatch (dominant).** The weighted objective spends 35% of its
   mass on solved structural tokens and 5% on empty squares. The model is
   Bayes-optimal for that objective; unweighted accuracy is a different
   objective.
2. **Latent capacity (secondary).** Even with perfect calibration, a 512-d
   vector loses occupancy detail as the pack grows (0.88 → 0.60 empty recall),
   and the reconstruction ceiling at 30 positions is 0.75 on clean input.
3. **Rectangle masking (tertiary).** It is the harder mode and the majority of
   the error mass, but it is also the point of the objective (no copying), so
   it should be kept and made easier only by capacity.

## Token weighting: options and recommendation

Candidate schemes, evaluated on the real token mix (corpus counts). "Empty
share" is the fraction of the supervised loss weight on empty squares.

| Scheme | empty | pieces | state | CLS/SEP |
|---|---:|---:|---:|---:|
| current `1/p`, cap 100 | 4.9% | 59.7% | 23.0% | 12.4% |
| `1/p`, cap 5 | 24.8% | 66.3% | 8.9% | 0%* |
| `1/p`, cap 4 | 29.2% | 62.4% | 8.4% | 0%* |
| `1/p`, cap 3 | 35.4% | 56.9% | 7.7% | 0%* |
| `sqrt(1/p)`, cap 4 | 31.1% | 56.7% | 9.0% | 3.2% |
| **proposed table** | **39.3%** | **55.0%** | **5.7%** | **0%** |

`*` with CLS/SEP explicitly zeroed, which is recommended in every scheme.

A fixed-knowledge simulation (undo the training weights to recover the implicit
posterior, then re-apply each candidate weight vector) gives a lower bound:

| Scheme | accuracy | empty recall | piece accuracy |
|---|---:|---:|---:|
| unweighted (w=1) | 0.872 | 0.937 | 0.748 |
| current | 0.646 | 0.524 | 0.877 |
| `1/p` cap 5 | 0.821 | 0.810 | 0.843 |
| `1/p` cap 3 | 0.849 | 0.865 | 0.819 |
| `sqrt(1/p)` cap 4 | 0.842 | 0.850 | 0.825 |

The simulation holds the representation fixed, so it understates schemes that
send more gradient to occupancy. It is only meant to show the direction and
that no scheme gets both high empty recall and the current piece accuracy
without retraining.

### Proposed weights (per token id)

| Group | Token share | Weight | Loss share |
|---|---:|---:|---:|
| empty | 62.2% | 1.0 | 39.3% |
| pawn | 17.1% | 2.0 | 21.6% |
| knight | 3.2% | 3.0 | 6.0% |
| bishop | 3.4% | 3.0 | 6.5% |
| rook | 4.6% | 3.5 | 10.1% |
| queen | 2.0% | 4.0 | 5.2% |
| king | 3.0% | 3.0 | 5.7% |
| turn / castling / en-passant | 4.4% | 2.0 | 5.7% |
| CLS / SEP | 1.6% | 0.0 | 0% |

Rationale: empty's share of the loss rises from **4.9% to 39.3%**, while a rare
queen still carries 4× an empty square and 2× a pawn, so piece identity keeps
signal. The solved structural tokens release ~30 points of loss mass. With the
new weights the Bayes boundary moves to roughly `p(empty) ≈ 0.5–0.6` for the
occupancy decision instead of ~0.9, which is where unweighted accuracy wants
it.

**Principled alternative (recommended if the weight table still leaves empty
recall below ~0.85):** split the objective into two heads/terms over the same
52-way logits:

1. **occupancy** — binary CE over all 64 squares, `empty` vs
   `logsumexp(piece logits)`. The classes are naturally ~55/45, so no frequency
   weighting is needed.
2. **identity** — 12-way CE over *occupied* squares only, with mild
   inverse-frequency weights (pawn 1, N/B/R 2, Q/K 2.5).
3. **state** — turn/castling/ep with weight 0.1–0.2; **CLS/SEP** weight 0.

This makes the objective match the metric: every square contributes occupancy,
and only occupied squares contribute identity. It costs one loss function and
one set of tests, not an architecture change.

Also add the eval-time correction `argmax(logits − log w)` and report both raw
and calibrated accuracy. With the proposed weights the two should be close;
with the current weights they differ by 21 points.

**Status:** the table is implemented as
`--token-weight-preset balanced-v1` (default `invfreq` preserves the old
behaviour). `balanced-v1` returns the table above, zeroes
`PAD`/`MASK`/`CLS`/`SEP`, and **skips the unigram count pass entirely**.
BDD coverage:
`test/unit/test_window_dataset.py::TestTokenWeightPresets` and
`test/unit/test_train_masked_cli.py::TestTokenWeightPreset`.

## Chinchilla sizing

The corpus is 2,602,026,504 positions; each contributes 68.05 tokens (64
squares + turn + castling + ep + SEP) plus one CLS per window:

| Quantity | Value |
|---|---:|
| Train tokens D | ≈ 177 B |
| Chinchilla-optimal N = D/20 | ≈ **8.9 B parameters** |
| Implied compute C = 6ND | ≈ 9.4 × 10²¹ FLOPs |

At 6.07M parameters the Chinchilla budget is only ~121M tokens (≈1.8M
positions, 0.07% of the corpus). The run consumed ~30B tokens, i.e. ~4,900
tokens/parameter, **~250× past the ratio**: it was capacity-limited, not
data-limited. A 20M model is still ~440× below the corpus-optimal 8.9B, but it
is a reasonable next step for a single 48 GB card. Caveats: Chinchilla is fit
on web text, chess tokens are highly redundant (consecutive positions differ by
one move), and the 512-d bottleneck caps reconstruction independently of
parameter count.

## Proposed next run: 20M parameters

Architecture (candidate A; exact counts from the model class):

| Component | 6M baseline | Proposed 20M |
|---|---:|---:|
| `d_model` | 256 | **512** |
| heads / head dim | 8 / 32 | **8 / 64** |
| `d_ff` | 1,024 | **1,536** (3× d) |
| encoder layers | 4 | **4** |
| decoder layers | 3 | **3** |
| latent | 512 | **1,024** (2× d) |
| total params | 6.07M | **20.39M** |
| encoder / decoder / neck / head | 3.20 / 2.53 / 0.33 / 0.01 | **10.59 / 8.46 / 1.32 / 0.03** |
| encoder / decoder share | 52.7% / 41.6% | 51.9% / 41.5% |

Why this shape:

- **Wider, not deeper.** The measured limit is the information channel, so
  `d_model` and the latent double while the proven 4+3 depth is kept. Step time
  grows with width but not sequentially; a 6+4 alternative (below) is 43%
  deeper for the same parameters.
- **Bottleneck 1024.** The capacity sweep says 512 floats for 30 positions is at
  the limit (empty recall 0.88 → 0.60). Doubling is the direct lever; a 1536
  variant costs only ~0.5M more if occupancy still saturates.
- **Balance preserved.** Encoder 52%, decoder 41%, bottleneck 6% — the same
  split as the 6M recipe, so the decoder remains a training scaffold rather
  than the parameter sink.
- **`d_ff` 3× rather than 4×.** With 4× the 20M budget only fits 3+3 layers and
  shifts params into the decoder (candidate B: 20.91M, decoder 48%). The
  encoder is the part that compresses the window, so it should keep the depth.

Alternatives considered:

| Candidate | Shape | Total | Note |
|---|---|---:|---|
| A (proposed) | d512, 8 heads, ff1536, 4+3, latent 1024 | 20.39M | balanced, same ratios as 6M |
| C (depth) | d384, 6 heads, ff1536, 6+4, latent 1024 | 19.10M | +43% depth, encoder 56% |
| B (decoder-heavy) | d512, 8 heads, ff2048, 3+3, latent 1024 | 20.91M | not recommended |

Training changes to pair with it:

- **Memory probe first** (batch 96 / 128 / 160). Width doubles the activation
  footprint; the 6M run's 27 GB typical / 37 GB worst at batch 240 suggests
  ~half the batch for the same headroom. Pick the largest batch that leaves
  ~20% worst-case headroom.
- **Steps:** ~100k at batch ~120 is ~15B tokens (≈2× the wall time per step of
  the 6M run, similar total time). 150k gives ~22B. Keep `max_seq_len 2048` and
  `max_segments 30` for comparability; the pack-size sweep should be logged
  every evaluation.
- **Loss:** run with `--token-weight-preset balanced-v1` (implemented; see
  above); log raw **and** calibrated accuracy, per-class accuracy, occupancy
  AUC, latent-swap delta.
- **Masking:** 50% rectangle / 50% skip-gram (`--p-rect 0.5 --p-skip-gram 0.5`)
  instead of the 6M run's 80/20, so masked-token prediction is trained equally
  on within-board and cross-board context.
- **Metadata fix:** store the resolved `AuxiliaryLossConfig`/weight vector in
  the checkpoint metadata, not the raw CLI args.
- **Eval performance:** set `OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1` in the
  launch environment; after the CUDA-SVD fix `full_eval` is ~1–2 min, so it can
  stay enabled at every evaluation.
- **Then** the InfoNCE ablation, now on a model whose reconstruction channel is
  not the binding constraint.

## Results: 20M run (stopped at 81,600)

Run `run_ae_20m_100k`, launched 2026-11-22 and stopped by decision on
2026-11-23 at step 81,600 (last checkpoint 81,000). Configuration: 20.39M
params, `d_model 512`, 8 heads, `d_ff 1536`, 4 encoder + 3 decoder layers,
1024-d latent, batch 128, lr 2e-4 with 2,000-step warmup and cosine decay over
100,000 steps, `balanced-v1` token weighting, and a **50/50 rectangle/skip-gram**
masking split (changed from the 6M run's 80/20 before launch). The probe
columns are 1,024-position estimates; the 4,096-position re-evaluation is in
the next section.

| step | val_loss | val_acc | piece acc | piece prior | `test_r2_material` | `mean_top10_cos` |
|---:|---:|---:|---:|---:|---:|---:|
| 2,000 | 0.7443 | 0.8087 | 0.8396 | 0.8807 | 0.5042 | 0.7292 |
| 20,000 | 0.5488 | 0.8862 | 0.9002 | 0.9436 | 0.5482 | 0.5408 |
| 40,000 | 0.5401 | 0.8980 | 0.9082 | 0.9497 | 0.6877 | 0.5452 |
| 56,000 | 0.5193 | 0.9059 | 0.9168 | 0.9528 | **0.7033** | 0.5461 |
| 68,000 | 0.5116 | 0.9069 | 0.9170 | 0.9533 | 0.6775 | 0.5620 |
| 80,000 | **0.5022** | **0.9093** | 0.9168 | 0.9532 | 0.6872 | 0.5699 |

What the numbers say (all probe values below use the training-time 1,024
positions; the 4,096-position re-evaluation is in the next section):

- **Reconstruction keeps improving, slowly.** `val_loss` fell 0.017 between
  60k and 80k (≈0.00085 per 1k steps) and had not fully flattened, but
  `val_acc` gained only 0.004 over the same span.
- **Embedding geometry plateaus from ~56k.** On a fixed 4,096-position pool
  `test_r2_material` moves 0.745 (56k) → 0.750 (80k) with overlapping
  confidence intervals; the 1,024-pool dip to 0.687 at 80k is sampling noise.
  `pc1_piece_corr` is flat at ~0.87 and `mean_top10_cosine` at 0.59–0.61.
- **The plateau is representational, not optimisation.** The 1024-d latent and
  the rebalanced weights fixed the 6M run's headline-accuracy artefact (empty
  squares are no longer 88% of the loss) and lifted masking accuracy, but the
  remaining reconstruction gains are in token classes that do not change the
  pooled vector's material axis.
- **Scaling the bottleneck did add material information.** At the same
  4,096-position protocol the 6M/512-d model scores R² 0.463 and the
  20M/1024-d model 0.750; the earlier "capacity did not help" reading came
  from the noisy 1,024 pool.
- **Raw `val_loss` is not comparable to the 6M run:** `balanced-v1` reweights
  token classes and the masking mixture changed to 50/50, so only in-run trends
  and the probe metrics are comparable.

Artifacts: `run_ae_20m_100k/` (checkpoints, `last.pt` step 81,000,
`best_step80000.pt`, `train.log`) and `run_ae_20m_100k_logs/` (train log, eval
history/metrics, PCA plot, neighbour reports).

## Full evaluation of the 20M bottleneck (2026-11-23)

Protocol: `best_step80000.pt` (best `val_loss`), 8,192-game validation pool
from the `test` split, CPU (8 threads). The training-time `full_eval` suite was
run at its usual 1,024-position pool (`windows=20`, `batch_size=8`,
`counterfactual_n=256`, `seed=17`), then the embedding probes were repeated on
a fixed 4,096-position pool with 10 material-probe splits and 50 phase
bootstraps. Total wall time ≈2 min. The suite is reproducible with
`scripts/eval_bottleneck_embeddings.py`; raw outputs are under
`hf://buckets/patrickfrank1/chesspos-checkpoints/run_ae_20m_100k_logs/evals/`.

```bash
python scripts/eval_bottleneck_embeddings.py \
  --checkpoint data/models/ae_20m_100k/best_step80000.pt \
  --data-dir data/hf-cache/datasets--patrickfrank1--chess-positions/snapshots/<hash>/test \
  --out-dir data/eval/ae_20m_80000
```

### Reconstruction by masking mode (1,024 pool)

| mode | accuracy | piece acc | prior baseline | turn | castle | scalar |
|---|---:|---:|---:|---:|---:|---:|
| board (rect) | 0.9194 | 0.9160 | 0.9529 | 1.000 | 0.975 | 0.992 |
| random | 0.9312 | 0.9282 | 0.9621 | 1.000 | 1.000 | 1.000 |
| span | 0.9306 | 0.9273 | 0.9613 | 1.000 | 1.000 | 1.000 |

Piece accuracy sits *below* the unigram prior baseline. That is expected under
`balanced-v1`: the model is trained on a reweighted loss in which empty squares
are no longer nearly free, so the unweighted argmax metric is no longer the
objective. Turn, castling, and scalar attributes are effectively solved.

### Embedding probes (same checkpoint, two pool sizes)

| metric | 1,024 positions | 4,096 positions |
|---|---:|---:|
| `test_r2_material` | 0.687 [0.655, 0.720] | **0.742 [0.726, 0.757]** |
| `test_acc_side_to_move` | 1.000 | 1.000 |
| `mean_top10_cosine` | 0.570 | 0.614 |
| `mean_random_pair_cosine` | 0.149 | 0.149 |
| `top5_material_delta` | 2.00 | 1.81 |
| `random_pair_material_delta` | 2.35 | 2.24 |
| `pc1_piece_corr` | 0.874 [0.858, 0.888] | 0.875 [0.869, 0.882] |
| `pc2_piece_corr` | 0.372 [0.340, 0.406] | 0.372 [0.357, 0.385] |

Counterfactual probes (1,024 pool): cosine to the move-child **0.933**, to a
second child **0.914**, to a corrupted board **0.938**, to a random position
**0.160**. The bottleneck therefore carries a strong position identity while
still separating unrelated positions.

### Checkpoint sweep (fixed 4,096-position pool)

| model | step | `test_r2_material` | 95% CI | `mean_top10_cos` | random-pair cos | top-5 Δmaterial | PC1 corr | PC2 corr |
|---|---:|---:|---|---:|---:|---:|---:|---:|
| 20M | 2,000 | 0.5774 | [0.553, 0.602] | 0.776 | 0.282 | 1.88 | 0.862 | 0.352 |
| 20M | 10,000 | 0.5056 | [0.477, 0.534] | 0.594 | 0.143 | 1.85 | 0.814 | 0.435 |
| 20M | 34,000 | 0.7101 | [0.688, 0.732] | 0.586 | 0.143 | 1.81 | 0.866 | 0.395 |
| 20M | 56,000 | 0.7452 | [0.727, 0.764] | 0.593 | 0.130 | 1.79 | 0.862 | 0.399 |
| 20M | 66,000 | 0.7306 | [0.713, 0.748] | 0.605 | 0.149 | 1.81 | 0.875 | 0.368 |
| 20M | 76,000 | 0.7489 | [0.735, 0.763] | 0.605 | 0.141 | 1.81 | 0.873 | 0.374 |
| 20M | 80,000 | **0.7503** | [0.736, 0.765] | 0.614 | 0.149 | 1.81 | 0.875 | 0.372 |
| 6M | 42,000 | **0.4626** | [0.447, 0.479] | 0.624 | 0.159 | 1.75 | 0.815 | 0.138 |

### Findings

1. **The apparent late-run decline was pool noise.** At 1,024 positions
   `test_r2_material` reads 0.703 at 56k and 0.687 at 80k; on the same 4,096
   positions it reads 0.745 and 0.750 with overlapping CIs. Material geometry
   **plateaus at ≈0.75** from ~56k rather than degrading.
2. **The wider bottleneck substantially improved the material axis.** At the
   same 4,096-position protocol the 6M/512-d checkpoint scores **0.463** and
   the 20M/1024-d checkpoint **0.750** — +0.29 R². The 6M latent width was a
   real binding constraint, so the 20M run fixed more than the metric artefact.
3. **Pool size biases the ridge probe.** With 1,024 positions only ~819 rows
   remain for training a 1,024-d ridge, which under-estimates R² (0.69 vs 0.74
   at 80k; 0.26 vs 0.46 at 6M). Only compare probe values measured on the same
   pool size.
4. **What saturates at ~0.75.** Reconstruction still improves in empty and
   state tokens between 60k and 80k, but those do not move the material axis.
   The remaining material error is consistent with within-position crowding in
   a single 1,024-d vector. The next levers are supervision (InfoNCE or an
   explicit material head) or a wider latent — not more steps at this size.
5. **Position identity dominates similarity; capacity did not tighten
   neighbourhoods.** Child similarity is 0.93 against 0.16 for random
   positions, and `mean_top10_cosine` is essentially unchanged from the 6M run
   (0.614 vs 0.624). The 20M gain is specifically along the linear material
   axis (R² 0.46 → 0.75) and PC2 (0.14 → 0.37), not more isotropic clustering.
6. **Side-to-move is linearly trivial** (accuracy 1.000) in both models: the
   turn token is recoverable from a clean direction, so it is not a useful
   discriminator between checkpoints.

## Downstream probes on the bottleneck (2026-11-23)

Protocol: `best_step80000.pt`, frozen encoder, only the head trains. The
bottleneck vector is the `embed()` output. Score regression feeds one position
per input (`CLS + segment + SEP`), 40k train games from the `train` split (the
60 GB instance could not hold the full 56k-game split), 3 epochs, 3 seeds, MSE,
batch 256, lr 1e-3, and evaluates on the held-out `test` split (20k games: 10k
standard + 10k elite). Sacrifice classification packs a puzzle into a sequence,
embeds each position as its own window (`--feature native_positions`) and
pools the per-position bottlenecks with the existing sequence head (mean
pooling); 1M puzzles (16% subset, 7.57% prevalence), 1 epoch, seed 17, BCE.

### Position score regression (held-out test, mean ± std over 3 seeds)

| model | R² | decisive AUC | MAE |
|---|---:|---:|---:|
| **20M AE bottleneck (frozen)** | **0.051 ± 0.004** | **0.645 ± 0.001** | **0.820** |
| `run_20m_lazy` transformer (frozen) | 0.023 ± 0.001 | 0.587 ± 0.003 | 0.859 |
| `run_6m_isolated` transformer (frozen) | 0.010 ± 0.002 | 0.571 ± 0.001 | 0.867 |
| material baseline (affine, same data) | 0.0675 | 0.6549 | 0.829 |

Validation R² on the best epoch was 0.0617 ± 0.0011, consistent with the
held-out 0.051 after the known elite/standard distribution shift. **Epoch 1 is
the best epoch for every seed**; epochs 2–3 overfit (val R² 0.061 → 0.036 /
0.014), which is why the best checkpoint is always the first.

### Sacrifice classification (validation, natural prevalence, 1M subset)

| model | PR-AUC | AUROC | F1 |
|---|---:|---:|---:|
| **20M AE bottleneck (frozen)** | **0.288** | **0.745** | **0.322** |
| `run_20m_lazy` transformer (frozen, 1M) | 0.203 | 0.697 | 0.262 |
| `run_6m_isolated` transformer (frozen, 1M) | 0.200 | 0.685 | 0.254 |
| `run_20m_lazy` transformer (frozen, full corpus) | 0.224 | 0.712 | 0.274 |

Precision 0.269, recall 0.399, accuracy 0.840 (below the 0.925 always-negative
baseline, so read PR-AUC/AUROC). Prevalence is 7.57%, so PR-AUC 0.288 is ~3.8×
chance.

### Findings

1. **The bottleneck is the strongest frozen representation tested so far.**
   Against the previous best frozen transformer it adds **+0.028 R² and +0.058
   decisive AUC** on score regression, and **+0.085 PR-AUC / +0.048 AUROC** on
   sacrifice. The reconstruction-only objective produces a more usable
   position embedding than the masked-transformer trunk.
2. **It nearly matches material-only scoring.** Score R² 0.051 vs 0.068 and
   AUC 0.645 vs 0.655 for an affine function of material; its MAE is already
   better (0.820 vs 0.829), because the bottleneck head predicts conservatively
   and pays for it in R², not in absolute error. Given the same embedding
   reaches R² 0.75 on a material probe, the remaining gap to material is the
   realised-result target, not missing material information.
3. **One epoch is enough for the frozen head.** The probe overfits quickly, so
   longer schedules are wasted compute and would pick a worse checkpoint.
4. **Sacrifice is where the bottleneck gains most.** PR-AUC 0.288 is the best
   frozen number across all models/protocols in this report, and it is close to
   the earlier *fine-tuned* 6M transformer (0.387) without adapting any encoder
   parameters.
5. **Caveats.** The score probe uses 40k rather than 56k games (disk-bound on
   the 44 GB run) and the sacrifice cell is one seed; the earlier frozen
   transformer rows used the same frozen-head protocol, so the ranking is
   comparable, but absolute numbers may shift slightly with the larger split.

Artifacts: `hf://buckets/patrickfrank1/chesspos-checkpoints/downstream/ae20m/`
(`models/`, `evals/`; metrics JSON, predictions `.npz`, and probe
checkpoints), plus the local `data/models/downstream/ae20m/` and
`data/eval/downstream/ae20m/` trees.

## Fine-tuned downstream arms (2026-11-24)

The frozen probes above adapt only the head. These arms unfreeze the whole
autoencoder (encoder + bottleneck) for one epoch, keeping the same heads:
score on 40k games (batch 128, encoder LR 1e-5 / head LR 1e-3, seed 17) with a
20k-game held-out test, and sacrifice on the 1M-puzzle subset (batch 128, same
LRs, per-position bottleneck pooling, seed 17).

### Position score regression

| recipe | R² | decisive AUC | MAE |
|---|---:|---:|---:|
| frozen bottleneck (3 seeds) | 0.0510 ± 0.0043 | 0.6448 ± 0.0006 | 0.820 |
| **fine-tuned bottleneck** | **0.0681** | **0.6662** | **0.7982** |
| material baseline (affine) | 0.0675 | 0.6549 | 0.829 |
| 20M transformer, full fine-tune | 0.131 | 0.713 | 0.773 |

Validation for the fine-tuned arm: R² 0.0731, AUC 0.6702, MAE 0.8045. The arm
now clears the material-only baseline on all three metrics, but the gain over
the frozen bottleneck is modest (**+0.017 R², +0.021 AUC, −0.022 MAE**).

### Sacrifice classification (1M subset, natural prevalence)

| recipe | PR-AUC | AUROC | F1 |
|---|---:|---:|---:|
| frozen bottleneck | 0.288 | 0.745 | 0.322 |
| **fine-tuned bottleneck** | **0.341** | **0.775** | **0.361** |
| 20M transformer, full fine-tune | 0.455 | 0.834 | 0.449 |

Precision 0.323, recall 0.410, accuracy 0.862. Fine-tuning adds **+0.053
PR-AUC / +0.030 AUROC / +0.039 F1** over the frozen bottleneck.

### Findings

1. **Fine-tuning helps, but the bottleneck has less headroom than the
   transformer trunk.** The transformer's full fine-tune moved score R²
   0.023 → 0.131 and sacrifice PR-AUC 0.203 → 0.455; the bottleneck moves
   0.051 → 0.068 and 0.288 → 0.341. It starts much stronger frozen, and the
   remaining gap is not recoverable by adapting the same one-epoch recipe.
2. **The fine-tuned bottleneck beats material-only scoring.** This is the
   first checkpoint in the project to clear the affine material baseline on
   score R², decisive AUC, and MAE simultaneously.
3. **The transformer remains the better fine-tuning base for sacrifice**
   (PR-AUC 0.455 vs 0.341), even though the bottleneck is the better frozen
   representation. The two recipes measure different ceilings: retrieval-style
   reconstruction vs task-adapted supervision.
4. **Operational:** flash-attn is required on the GPU box. Without it the
   varlen fallback is a per-segment SDPA loop — a 128×5-position batch took
   **3.4 s/step** instead of **62 ms** after installing the matching wheel.
   Batch 256 OOMs on the longest (22-position) puzzles; 128 fits at ~15 GB.

Artifacts: `hf://buckets/patrickfrank1/chesspos-checkpoints/downstream/ae20m_finetune/`
(`models/`, `evals/`), plus the box-local `data/models/downstream_finetune/`
and `data/eval/downstream_finetune/` trees.

## Clean pack-size sweep on the 20M bottleneck (2026-11-23)

The 6M post-mortem measured the latent's information limit by reconstructing
clean gold windows while varying how many consecutive positions share the
latent (192 windows, no `MASK` tokens). Repeating it on `best_step80000` with
`scripts/eval/pack_size_sweep.py`:

| context positions | accuracy | empty recall | piece accuracy |
|---:|---:|---:|---:|
| 1 | 1.000 | 1.000 | 1.000 |
| 2 | 1.000 | 1.000 | 1.000 |
| 3 | 1.000 | 1.000 | 1.000 |
| 5 | 1.000 | 1.000 | 1.000 |
| 8 | 0.998 | 0.996 | 0.999 |
| 12 | 0.992 | 0.989 | 0.996 |
| 16 | 0.988 | 0.982 | 0.993 |
| 19 | 0.984 | 0.975 | 0.991 |
| 24 | 0.978 | 0.968 | 0.988 |
| 30 | **0.969** | **0.955** | 0.982 |

The 6M curve over the same sizes was 0.929 / 0.870 / 0.790 / **0.753** (1 / 8 /
19 / 30). The 20M model's 1,024-d bottleneck is therefore **not close to its
information limit at the training pack size**: it reconstructs 30 positions at
0.969 with 0.955 empty recall, where the 6M model managed 0.753 / 0.595.

This reframes the earlier plateau: what remains at 30 positions is no longer a
capacity story. The downstream probes already showed that a linear material
probe reaches R² 0.75 while the realised-result head reaches only 0.051, so the
next lever is supervision/task adaptation (or a larger latent only if longer
windows become a goal), not more reconstruction capacity at 1,024-d.

## Operational notes (issues hit and fix)

| Symptom | Cause | Fix |
|---|---|---|
| `full_eval` stalls for 10–37 min after the training steps | `phase_stats_ci` refits PCA for every bootstrap; a 1,024×1,024 numpy SVD takes **107 s** with 96 OpenBLAS threads (thread oversubscription) vs **4.6 s** with one thread | `phase_stats`/`phase_stats_ci` now run the SVD on CUDA via `torch.linalg.svd` (~17 s for 21 SVDs); launch env also sets `OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1` |
| Concurrent HF shard downloads stall at 0 bytes | `hf_xet` multi-threaded/multi-process downloads hang on these hosts | `HF_HUB_DISABLE_XET=1`; verified 8 parallel downloads at 134 MB/s |
| Startup stuck on "remote unigram pass: shard 1/443" | Threaded remote unigram pass downloads every shard and stalls | Built `data/hf-shards/unigram_counts.npz` from the 443 `.meta.json` sidecars (exact totals, 110 s) |
| No training step for many minutes with `--num-workers 16` | DataLoader/compile contention; first batch never delivered | `--num-workers 8` (GPU 98–100%) |
| `UnboundLocalError: content_hidden_fn` at startup | `view_helpers` resolution only assigned the variable in the visible-identity branch (also affected the default transformer path) | Extracted `view_helpers()`; commit `67a9531`, 3 BDD tests |
| Container stopped by the host at step 29,000 | Host-side failure (`cur_state=stopped`), trainer logged no error | Resumed on a new box from the bucket checkpoint; watchdog replays/skips the data stream deterministically |
| Replacement offer had a ~2 h lifetime | Short-duration host | Moved to a 252-day offer |
| `tokens/s` looks like ~1.3 M | Counter uses padded slots (`batch × 2048`) | Real throughput ≈ 750 k content tokens/s ≈ 11 k positions/s |

### Resume semantics

On `--resume`, `WindowDatasetConfig.resume_step` is set to the checkpoint step
and each worker skips the windows it already produced
(`floor((resume_step − worker_id − 1)/num_workers) + 1` batches). The stream is
deterministic (`seed + worker_id`, sorted shard inventory, replayed
`rng.permutation`), so this resumes rather than repeats. With step 29,000 and 8
workers that is exactly 870,000 windows per worker (29,000 is divisible by 8, so
the round-robin alignment is preserved). Two bounded losses: prefetched but
unconsumed windows (≤ 16 steps) are skipped unseen, and steps after the last
checkpoint before a crash (≤ one log interval, ~50 steps) are re-seen.
`--no-resume-data-skip` restarts the stream instead.

## Reproducing

```bash
# fresh box: torch 2.8.0+cu128 image, then flash-attn wheel
pip install https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3.post2/\
flash_attn-2.8.3.post2+cu12torch2.8cxx11abiTRUE-cp311-cp311-linux_x86_64.whl

export HF_TOKEN=$(cat /root/.hf_token) HF_HUB_DISABLE_XET=1
export PYTHONPATH=/workspace/chesspos PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

python3 -m src.run.train_masked_transformer \
  --model-arch attention_autoencoder \
  --d-model 256 --n-layers 4 --n-heads 8 --d-ff 1024 \
  --n-decoder-layers 3 --latent-dim 512 \
  --data-repo patrickfrank1/chess-positions --data-cache data/hf-cache \
  --stream-shards --streaming --stream-shard-cache data/hf-shards \
  --num-workers 8 --steps 100000 --batch-size 240 --lr 2e-4 --warmup 2000 \
  --max-seq-len 2048 --max-segments 30 \
  --log-every 50 --eval-every 2000 --eval-batches 20 --save-every 1000 \
  --bf16 --compile \
  --checkpoint-dir data/models/ae_6m_100k \
  --log-file data/models/ae_6m_100k/train.log \
  --upload-bucket patrickfrank1/chesspos-checkpoints \
  --upload-prefix run_ae_6m_100k
```

`--model-arch attention_autoencoder` alone selects the current deep-narrow
preset (d 512, 6 encoder / 3 decoder layers, 8 heads, `d_ff` 1024, latent
512, 20.4M params). The explicit size flags above reproduce the 6M run.

Pull artifacts:

```bash
hf buckets sync hf://buckets/patrickfrank1/chesspos-checkpoints/run_ae_6m_100k/ ./models/run_ae_6m_100k/
```

## Known gaps

- **`val_loss` changed meaning after this report.** The autoencoder evaluation
  now logs an unweighted content-only loss (PAD/`CLS`/`SEP` excluded). The
  historical 0.5022 at step 80k includes the untrained structural tokens,
  which contribute roughly 0.29 nats; the content-only value is about 0.22.
  Do not compare numbers across the change.
- **Checkpoint metadata is misleading for this arch.** `build_metadata()`
  serialises raw CLI args, so `last.meta.json` reports `nce_weight=0.1`,
  `inv_weight=0.5`, `var_weight=1.0` even though the resolved objective has all
  of them at zero. The `loss weights:` log line is authoritative.
- **No retrieval benchmark for this run.** `--retrieval-benchmark` expects the
  puzzle-repo benchmark; without it there are no near-duplicate or strategic
  metrics, so improvements are judged on validation reconstruction only.
- **`token_rank_effective`** is computed on decoder hidden states for this
  architecture, which is less meaningful than for the transformer.
- **Validation sample is narrow**: a fixed, unshuffled 4,800-window slice of an
  8,192-game pool.
- **The reported `val_acc` is uncalibrated.** It is the argmax of logits
  trained under the frequency-weighted loss, so it measures a different
  decision boundary. On the 40k checkpoint it understates masked-square
  accuracy by **21 points** (0.648 raw vs 0.861 after a `+3.5` EMPTY bias).
- **`CLS`/`SEP` carry weight 100 by accident.** They are absent from the
  unigram cache, so the smoothed `1/p` weight hits the cap. They consume 12.4%
  of the supervised loss mass while being predicted at ~0.999 accuracy.
- **The plateau analysis used one checkpoint (40k) and one fixed 1,024-window
  test slice.** The bias sweep and AUC are stable across rect/skip-gram and
  window lengths, but not across checkpoints.
- **The latent-swap diagnostic is not logged** by the training loop; it was run
  ad hoc.

## Future work

### 1. InfoNCE ablation (planned)

The run is reconstruction-only by design: the architecture emits one latent per
window, so there are no per-segment views for VICReg/InfoNCE, and hard-negative
mining is skipped. The measurements above show the geometry we would want
already exists (0.52 same-window vs 0.04 other-game), so an objective with
random in-batch negatives would be nearly satisfied at initialisation (~0.22
nats). The informative version needs hard negatives.

Variants, in increasing cost:

| Variant | Objective | Extra cost |
|---|---|---|
| **A** | Window latent ↔ `k` sampled positions encoded in isolation (multi-positive InfoNCE), negatives = other windows' alone views | ~13% encoder FLOPs (`k=4` × 67 vs 1 × 2000 tokens) |
| **B** | A + negatives restricted to the hardest other-game views; same-game pairs farther than ±`temporal_radius` ignored (reuse `build_contrastive_masks`) | negligible over A |
| **C** | Window ↔ window with same-game positives (two windows per game, ±3 plies) | paired sampler or a second pass |

Proposed weights: `nce 0.05–0.1`, temperature 0.05–0.07, plus **VICReg variance
(0.5) and covariance (0.05)** on the window latent to guard against contrastive
collapse, all ramped over ~2,000 steps. `nce_weight=0` must reproduce current
behaviour exactly.

Guardrails / acceptance criteria:

- same-window cosine rises (target ~0.7) while other-game cosine stays low;
- `latent-swap` delta stays ≥ 1.0 nat (if it shrinks, the contrastive term is
  eating the reconstruction channel);
- validation reconstruction degrades by < 5%;
- retrieval metrics improve on the puzzle benchmark (near-duplicate
  recall@1/5/10, strategic MAP) — the decisive criterion, since the risk is
  blurring within-game position detail (same-window positions are already at
  0.60).

A cheap alternative worth trying first is **masking invariance**: cosine between
`latent(masked window)` and `latent(unmasked window)` with a stop-gradient. No
negatives, no collapse risk, and it targets the training/evaluation input
mismatch directly (training sees ~30% masked windows, evaluation sees clean
positions).

### 2. Metric and tooling fixes

- **Done (2026-11-23):** the periodic autoencoder evaluation now logs the
  **unweighted** reconstruction loss next to the weighted one, per-class
  accuracy (empty / pieces / state), the **calibrated** accuracy
  `argmax(logits − log w)`, the **occupancy AUC**, and the **latent-swap**
  delta. The resolved token weights and preset are stored in checkpoint
  metadata (`token_weights`, `token_weight_preset`).
- Still open: wire the retrieval benchmark for this architecture.

### 3. Capacity and data questions

- The clean pack-size sweep is the capacity measurement to repeat on the 20M
  model: same-target reconstruction at 1/8/19/30 positions. The 6M curve
  (0.929 / 0.870 / 0.790 / 0.753) is the baseline.
- A 1024-d latent is the proposed next step; 1536 is the fallback if occupancy
  recall still saturates. Beyond that, capping segments per window
  (`p_single` / `p_few`) would sharpen the compression ratio, at the cost of
  changing what a "window" means.
- The 20M run at ~15B tokens is still ~440× below the corpus-optimal 8.9B
  model, but it is the largest balanced step that fits a single 48 GB card.
  After it, the question is 50–100M params on ~1–2B tokens (Chinchilla
  balanced, ~1% of the corpus), not more steps at 20M.
