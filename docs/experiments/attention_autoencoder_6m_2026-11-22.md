# Attention autoencoder: one 512-d bottleneck per packed window

**Run:** `run_ae_6m_100k` (in progress, started 2026-11-22)
**Architecture:** `attention_autoencoder`
**Parameters:** 6,068,276
**Artifacts:** `hf://buckets/patrickfrank1/chesspos-checkpoints/run_ae_6m_100k/`

## Status

Training is **running** at the time of writing (step ~34,850 of 100,000,
17:24 UTC on 2026-11-22). Best validation loss so far **0.7534 at step
34,000**, still improving. ETA for step 100,000 is roughly 01:30 UTC on
2026-11-23.

This report records the design, the implementation surface, the training
configuration, the measurements taken during the run, the metric definitions
that make train and validation numbers look far apart, and the follow-up work —
including the InfoNCE ablation proposed at the end.

## TL;DR

- One 512-d vector summarises an entire packed window (~19 positions); a
  latent-only decoder reconstructs every non-pad token against the unmasked
  gold sequence. There is no token-embedding path into the decoder, so the
  bottleneck is the only channel from input to output.
- Full-reconstruction validation loss fell from **1.5685 (2k)** to **0.7534
  (34k)**; masked-token accuracy rose from **0.325** to **0.647**.
- The apparent "train loss 0.24 vs val loss 0.75" gap is a **metric
  definition** difference, not overfitting: the trainer logs the
  inverse-frequency-weighted CE (cap 100), evaluation logs the unweighted mean.
  On identical windows and identical weights the two agree exactly, and
  train-split and test-split windows score within 0.0002 nats of each other.
- The bottleneck is load-bearing: decoding with another window's latent
  roughly doubles the loss (1.26 → 2.44 nats at step 4k).
- The window latent already sits close to its own positions and far from other
  games (cos 0.52 vs 0.04) without any contrastive term. That emergent geometry
  motivates the planned NCE ablation.

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
| `src/modeling/autoencoder/config.py` | `AttentionAutoencoderConfig`; defaults are the 6M recipe |
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

## Results so far

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

Training loss (weighted) at step 34,850: **0.2418**, lr 1.49e-4, GPU ~98%.

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

## Operational notes (issues hit and fix)

| Symptom | Cause | Fix |
|---|---|---|
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

`--model-arch attention_autoencoder` alone selects the 6M preset
(d 256, 4 encoder / 3 decoder layers, 8 heads, `d_ff` 1024, latent 512).

Pull artifacts:

```bash
hf buckets sync hf://buckets/patrickfrank1/chesspos-checkpoints/run_ae_6m_100k/ ./models/run_ae_6m_100k/
```

## Known gaps

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

- Log the **unweighted** reconstruction loss alongside the weighted one, and
  pass the training weights into `evaluate()`, so train/val are comparable.
- Store the **resolved** `AuxiliaryLossConfig` in checkpoint metadata.
- Add the latent-swap delta to the periodic evaluation output.
- Wire the retrieval benchmark for this architecture.

### 3. Capacity and data questions

- One 512-d vector spans ~19 positions (~1,245 tokens). Capping segments per
  window (`p_single` / `p_few`) would sharpen the compression ratio and make
  reconstruction a fairer test of the bottleneck.
- The run sees ~17% of the corpus in 100k steps; more steps versus a larger
  batch is an open question (throughput is flat from batch 192 to 256, so
  larger batches buy tokens, not speed).
