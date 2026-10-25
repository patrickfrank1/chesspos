# Tiny objective sweep: VICReg, invariance, InfoNCE, masking

Status: completed 2026-10-25. Informs the new base-training loss defaults.

## Motivation

The shipped foundation checkpoint has a very low effective representation rank
(embedding collapse), and the new base-training objectives were added to counter
it: harder rectangle/skip-gram MLM, context-removal invariance, VICReg
variance/covariance, and optional in-batch InfoNCE. Before spending a full GPU
run, a tiny model was trained on a small data subset to answer two questions:

1. Which loss/masking knobs actually affect collapse?
2. What defaults should the next full-scale run use?

## Setup

- Instance: vast.ai `57689566`, RTX 3060 12 GB, $0.07/h.
- Code: `feature/improve-semantic-similarity` @ `742a011` (the Ray
  `train_loss` reporting fix is a separate, later change).
- Data: `patrickfrank1/chess-positions`; two train shards (242 MB, 200,000
  games) and one test shard (47 MB, 38,819 games) copied to
  `data/smoke/{train,test}` on the box. The train/test shard layout and the
  separate `benchmarks/retrieval/` folder were verified end-to-end.
- Retrieval benchmark: the shipped 25,000-position benchmark (14 near queries,
  1,381 strategic queries) pulled from
  `patrickfrank1/chess-positions@main` (`benchmarks/retrieval/**`).
- Model: `d_model=128`, `n_layers=2`, `n_heads=4`, `d_ff=512`, dropout 0.1
  (0.43 M parameters), `max_seq_len=2048`, `max_segments=64`.
- Training: 1,000 steps, batch 32, lr 3e-4, warmup 500, bf16, no compile,
  `num_workers=0`, eval every 200 steps, checkpoint at 1,000, seed 17.
- Masking (unless noted): 80% rectangle (30–50% of squares, 2x2–4x4,
  synchronized across segments) + 20% skip-gram (`skip_gram_unmask_rate=0.10`).
- Metrics:
  - `token_rank_effective`: effective rank of token hidden states over 4,096
    tokens.
  - `retrieval_rank_effective`, `retrieval_uniformity`, near/strategic MAP and
    recall@10 from the frozen benchmark.

The tiny model/subset is deliberately cheap; use the **relative ranking** of
configurations, not the absolute values, to guide the full-scale run. The near
query set has only 14 queries, so near-MAP differences are noisy.

## Runs and results

All runs share the setup above; only the loss/masking flags differ.

| run | loss weights (mlm/inv/var/cov/nce) | masking | val | acc | token rank | ret rank | strat MAP | strat R@10 | near MAP |
|---|---|---|---|---|---|---|---|---|---|
| MLM-only | 1 / 0 / 0 / 0 / 0 | mixture | 1.797 | .352 | 12.68 | 1.15 | .1183 | .611 | .3408 |
| old defaults | 1 / .1 / .1 / .01 / 0 | mixture | 1.799 | .350 | 15.82 | 1.17 | .1250 | .620 | .3341 |
| NCE-only | 1 / 0 / 0 / 0 / .2 | mixture | 2.021 | .309 | 13.93 | 4.96 | .0871 | .506 | .4264 |
| **VICReg-only** | 1 / 0 / 1.0 / .1 / 0 | mixture | **1.852** | **.345** | **46.10** | 1.60 | .1010 | .580 | .4537 |
| VICReg + NCE + inv .02 | 1 / .02 / 1.0 / .1 / .2 | mixture | 1.970 | .311 | 43.44 | 7.50 | .0752 | .426 | .3423 |
| VICReg + NCE, inv 0 | 1 / 0 / 1.0 / .1 / .2 | mixture | 1.957 | .317 | 44.20 | 8.04 | .0767 | .426 | .3698 |
| rect-only + VICReg + NCE | 1 / 0 / 1.0 / .1 / .2 | rect-only | 1.976 | .316 | 43.41 | 6.81 | .0780 | .424 | .5929 |

`old defaults` are the defaults before this sweep
(`inv=.1, var=.1, cov=.01, nce=0`), not the earlier 2,000-step mixture run
(which used `nce=.2` and a 2,000-step LR schedule and is therefore not directly
comparable at the same step).

### Token rank by evaluation step

| run | 200 | 400 | 600 | 800 | 1000 |
|---|---|---|---|---|---|
| MLM-only | 31.58 | 14.75 | 13.40 | 12.84 | 12.68 |
| old defaults | 31.22 | 16.35 | 15.26 | 15.87 | 15.82 |
| NCE-only | 23.24 | 15.07 | 13.97 | 13.93 | 13.93 |
| VICReg-only | 41.19 | 42.02 | 46.52 | 46.21 | 46.10 |
| VICReg + NCE + inv .02 | 35.43 | 28.96 | 38.57 | 42.45 | 43.44 |
| VICReg + NCE, inv 0 | 36.39 | 29.66 | 37.74 | 42.99 | 44.20 |
| rect-only + VICReg + NCE | 32.70 | 27.87 | 37.32 | 42.25 | 43.41 |

## Findings

1. **VICReg variance is the anti-collapse lever.** `var=1.0, cov=0.1` raises
   token rank from 12.68 (MLM-only) to 46.10, with only a small MLM cost
   (val loss 1.797 -> 1.852, accuracy .352 -> .345). The old defaults
   (`var=.1, cov=.01`) only reach 15.82, so they are too weak to matter.

2. **InfoNCE alone does not fix token collapse.** NCE-only stays at rank 13.93,
   but it does spread the pooled retrieval embeddings (retrieval rank
   1.15 -> 4.96, uniformity -0.095 -> -1.371). It costs strategic MAP
   (.118 -> .087) and val loss (1.797 -> 2.021).

3. **Adding NCE to strong VICReg is worse than VICReg-only here.**
   VICReg-only vs VICReg+NCE (inv 0): val 1.852 vs 1.957, token rank 46.10 vs
   44.20, strategic MAP .101 vs .077. NCE raises pooled retrieval rank
   (1.60 -> 8.04) but loses semantic structure at this scale.

4. **Invariance weight is near-neutral at 0.02.** VICReg+NCE with inv .02 vs
   inv 0: val 1.970 vs 1.957, token rank 43.44 vs 44.20, strategic MAP .0752
   vs .0767. A large invariance weight (0.5) has not yet been ablated; it is a
   deliberate next hypothesis (see below).

5. **Masking is not the lever.** Rect-only vs the 80/20 rectangle/skip-gram
   mixture at the same strong loss gives nearly identical token rank
   (43.41 vs 44.20), strategic MAP (.0780 vs .0767) and recall@10
   (.424 vs .426). The near-MAP difference (.593 vs .370) is on 14 queries and
   should not be trusted. Keep the mixture.

6. **Pooled retrieval collapse remains under VICReg-only.** Token rank is 46,
   but `retrieval_rank_effective` is only 1.60 (random-init is ~1.3). The
   reason is structural: VICReg currently regularizes **window-pooled**
   segment embeddings, while retrieval encodes each position **alone**. Only
   inv/NCE touch the alone embeddings. NCE raises the pooled rank but hurts
   strategic MAP at the tested weight.

## New defaults

Following the sweep (strong VICReg) and the explicit request to keep MLM,
InfoNCE and a large context-removal invariance weight:

| parameter | value |
|---|---|
| `mlm_weight` | 1.0 |
| `inv_weight` | **0.5** |
| `var_weight` | **1.0** |
| `cov_weight` | **0.1** |
| `nce_weight` | **0.1** |
| `nce_temperature` | 0.1 |
| `vicreg_gamma` | 1.0 |
| `vicreg_eps` | 1e-4 |
| `aux_warmup` | **200** |
| `p_rect` / `p_skip_gram` | 0.8 / 0.2 (unchanged) |
| `skip_gram_unmask_rate` | 0.10 (unchanged) |

Rationale and caveats:

- `var=1.0, cov=0.1` is the clear sweep winner for token rank at a small MLM
  cost.
- `inv=0.5` is large on purpose: it should force the context-free/retrieval
  embedding to agree with the contextual one, which the sweep did not solve.
  The sweep only tested inv 0 and 0.02, so 0.5 is a hypothesis, not a measured
  optimum.
- `nce=0.1` includes the requested InfoNCE term at half the tested 0.2 weight
  (0.2 hurt strategic MAP in the tiny run). Raise it only if pooled retrieval
  rank stays too low and strategic MAP holds up.
- `aux_warmup=200` avoids the large invariance term dominating before MLM has
  established useful representations. It was not separately ablated; the
  sweep used 0.
- Keep the rectangle/skip-gram mixture; masking was neutral in the sweep.

### Validation run

A 1,000-step run with the new defaults was launched on the same tiny setup to
confirm stability and to provide a first comparison point. Results are appended
below when it finishes.

| run | val | acc | token rank | ret rank | strat MAP | strat R@10 | near MAP |
|---|---|---|---|---|---|---|---|
| new defaults (pending) | | | | | | | |

## Recommended next steps

1. Run the full base model (d512/l8) with the new defaults and monitor
   `token_rank_effective`, `retrieval_rank_effective`, `retrieval_uniformity`,
   strategic MAP/recall, and per-theme MAP against the shipped baseline.
2. If pooled retrieval rank stays low while token rank is healthy, apply the
   VICReg variance/covariance term to the **alone (context-free) embeddings**
   as well, or add a projection head. This is the missing regularizer for the
   representation the benchmark actually uses.
3. If strategic MAP drops, ablate `nce_weight` (0.05 vs 0.1) and
   `inv_weight` (0.1/0.25/0.5) at fixed strong VICReg; masking can stay fixed.
4. Keep using the shipped 25k benchmark for every checkpoint.

## Artifacts and reproduction

- Local logs: `logs/vastai_tiny_smoke/sweep/*.log` (sweep) and
  `logs/vastai_tiny_smoke/` (earlier 8-step/300-step/1,000-step/2,000-step
  runs).
- Local checkpoints: `data/models/sweep/<run>/` and
  `data/models/tiny_run_mix/`.
- Box: `/workspace/sweep/`, `/workspace/chesspos/data/smoke/`,
  `/workspace/chesspos/data/models/sweep/`.
- The sweep was run with a local `os.cpu_count = lambda: 1` wrapper to avoid a
  host-RAM spike from concurrent Parquet decoding; this only affects data
  loading concurrency, not the training math.

Representative command (same for every run, with flags changed):

```bash
python3 -m src.run.train_masked_transformer \
  --train-dir data/smoke/train --val-dir data/smoke/test \
  --retrieval-benchmark patrickfrank1/chess-positions \
  --retrieval-benchmark-revision main \
  --steps 1000 --batch-size 32 --max-seq-len 2048 --max-segments 64 \
  --d-model 128 --n-layers 2 --n-heads 4 --d-ff 512 --num-workers 0 \
  --log-every 50 --eval-every 200 --eval-batches 10 --save-every 1000 \
  --token-rank-tokens 4096 --bf16 --no-full-eval --seed 17 \
  --inv-weight 0.0 --var-weight 0.0 --cov-weight 0.0 --nce-weight 0.0 \
  --checkpoint-dir data/models/sweep/mlm_only
```

The new-default run uses:

```bash
  --inv-weight 0.5 --var-weight 1.0 --cov-weight 0.1 \
  --nce-weight 0.1 --aux-warmup 200
```
