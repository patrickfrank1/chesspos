# Experiment documentation

This directory collects training and analysis experiments. Plans, designs, and
runbooks stay in `docs/`.

## Current status

The Perceiver 6M line is **paused**, not because the forward pass is broken but
because the content embedding does not learn material. The next useful work is
content-preservation supervision, not simply more training steps. See
[perceiver_6m_lr2e4_2026-11-10.md](perceiver_6m_lr2e4_2026-11-10.md).

## Experiment reports

| Date | Report | Topic | Outcome |
|---|---|---|---|
| 2026-11-10 | [Perceiver 6M: LR schedule collapse and validation drift](perceiver_6m_lr2e4_2026-11-10.md) | Segment-aware Perceiver IO, LR-schedule extension, NCE guard, resume data replay | First run exploded at ~33.4k; resume run stayed stable but val drifted and material R² stayed negative. Stopped at 41.5k. |
| 2026-11-09 | [Perceiver content-embedding collapse](analysis_perceiver_content_collapse_2026-11-09.md) | Shared latent readout and large segment embedding | Root-caused a constant-direction collapse; fixed with small segment-embedding init and decoded-slot attention pooling. |
| 2026-11-09 | [run_6m_isolated plateau](analysis_run_6m_isolated_plateau_2026-11-09.md) | Transformer baseline, content-isolated InfoNCE | Lost material in deeper layers because visible content positions lack identity supervision. |
| 2026-10-25 | [Tiny objective sweep](experiment_objective_sweep_2026-10-25.md) | VICReg, invariance, InfoNCE, masking | Selected the current auxiliary objective mix. |
| — | [Previous experiments](previous_experiments.md) | Earlier embedding approaches | Historical notes on representation and pooling experiments. |
| — | [UMAP embedding structure](umap_embedding_structure.md) | Embedding geometry analysis | UMAP analysis of chess-position embeddings and failure modes. |

## Artifacts

Checkpoints, logs, eval histories, and run scripts are synced to:

```text
hf://buckets/patrickfrank1/chesspos-checkpoints/
```

Relevant Perceiver 6M prefixes:

- `run_perceiver6m/`
- `run_perceiver6m_lr2e4/`
- `run_perceiver6m_collapsed_20261109T232217Z/`
- `run_perceiver6m_latentcollapse_20261109T232839Z/`
- `run_perceiver6m_maxseg128_failed_20261109T230652Z/`

## Recurring failure modes

1. **Embedding collapse to a common direction.**
   - Monitor `mean_top10_cosine`, `mean_random_pair_cosine`, and NCE against
     `ln(batch_size)`.
   - VICReg centered variance and invariance alone do not prevent a large
     common additive component.
2. **Train-loss explosion after LR-schedule changes.**
   - Changing `--steps` changes the cosine schedule.
   - A 200k schedule kept LR ~1.6x higher at 30k than the original 72k schedule.
3. **Validation drift and repeated data.**
   - Large streaming resume steps are expensive to fast-forward.
   - Disabling resume skip avoids the wait but replays earlier data.
4. **Material is missing from the pooled content vector.**
   - Per-token information can exist while the pooled retrieval embedding lacks
     a usable material axis.
   - Add explicit content-preservation supervision before scaling steps further.

## Next experiment checklist

- [ ] Add visible-content identity loss or a non-zero content reconstruction
      weight.
- [ ] Add a material auxiliary head on the content embedding.
- [ ] Run short 10k–20k comparisons with eval every 1k–2k.
- [ ] Add a validation-loss / material-R² early-stop guard.
- [ ] Implement fast resume-data skipping for streaming shards.
