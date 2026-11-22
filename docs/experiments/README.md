# Experiment documentation

This directory collects training and analysis experiments. Plans, designs, and
runbooks stay in `docs/`.

## Current status

The Perceiver 6M line is **paused**, not because the forward pass is broken but
because the content embedding does not learn material. A visible-content identity
loss is now implemented on the transformer path, but the recommended next step is
a simpler single-tower contrastive baseline with validated hard negatives. See
[text_embedding_survey_and_simplification_2026-11-10.md](text_embedding_survey_and_simplification_2026-11-10.md)
for the full analysis and plan.

A new line ran the attention autoencoder that compresses each packed window into
a single 512-d bottleneck and reconstructs every token from it. The 6M run was
**stopped at step 42,000** after validation plateaued; the post-mortem traces
the plateau to the frequency weighting (empty squares were 4.9% of the loss
weight but 88% of the errors) plus the 512-d latent capacity, not to the data.
A 20M-parameter follow-up with a 1024-d latent and rebalanced weights is
proposed. See
[attention_autoencoder_6m_2026-11-22.md](attention_autoencoder_6m_2026-11-22.md).

## Experiment reports

| Date | Report | Topic | Outcome |
|---|---|---|---|
| 2026-11-22 | [Attention autoencoder, one 512-d bottleneck per window](attention_autoencoder_6m_2026-11-22.md) | Single latent per packed window, latent-only decoder, full-sequence reconstruction, 6M params | Stopped at 42k/100k (val loss 1.57 → 0.7415, masked acc 0.33 → 0.650). Plateau = loss/metric mismatch: 88% of errors are empty squares, only 4.9% of loss weight; +3.5 EMPTY bias lifts accuracy to 0.861. 512-d latent caps 30-position reconstruction at 0.753. 20M/1024-d follow-up proposed. |
| 2026-11-15 | [Downstream probes: sacrifice classification and position score regression](downstream_probes_2026-11-15.md) | Frozen, last-block, and full-model fine-tuning of `run_20m_lazy` and `run_6m_isolated` on two downstream tasks | Score: frozen AUC 0.587 vs 0.656 for a material baseline; full fine-tuning lifts them to 0.713 / 0.700. Sacrifice (1M subset): frozen PR-AUC ~0.20; full fine-tuning lifts 20m to 0.455 / 6m to 0.387 (AUROC 0.834 / 0.802). Pooling is irrelevant; adaptation is the lever. |
| 2026-11-10 | [Perceiver 6M: LR schedule collapse and validation drift](perceiver_6m_lr2e4_2026-11-10.md) | Segment-aware Perceiver IO, LR-schedule extension, NCE guard, resume data replay | First run exploded at ~33.4k; resume run stayed stable but val drifted and material R² stayed negative. Stopped at 41.5k. |
| 2026-11-10 | [Text embedding survey and simplified chess embedding plan](text_embedding_survey_and_simplification_2026-11-10.md) | MTEB/embedding-model patterns, chesspos mapping, hard-negative validation | Recommends a single-tower contrastive baseline, validated hard negatives, large batches, and staged training. |
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

Attention autoencoder runs:

- `run_ae_6m_100k/` — `last.pt` (step 42k), `best_stepNNNNN.pt`, `.meta.json`
  sidecars
- `run_ae_6m_100k_logs/` — `train.log`, eval plots, neighbour reports

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
   - This is a lossy-embedding property, not a requirement for full
     reconstruction. Store FEN/metadata and use filters/heads for material-aware
     tasks; add reconstruction only for generation/editing.
5. **A weighted loss can optimise a different metric than the one you report.**
   - Inverse-frequency weighting made empty squares 4.9% of the loss but 88% of
     the errors; `val_acc` was 21 points below the calibrated accuracy.
   - Log per-class accuracy, calibrated accuracy, and occupancy AUC, not just
     the argmax of the weighted logits.

## Next experiment checklist

- [x] Post-mortem the 6M plateau and stop the run (42k/100k).
- [x] Add the `balanced-v1` per-token weight preset (`--token-weight-preset balanced-v1`).
- [ ] 20M run: d512 / 4+3 / latent 1024 / rebalanced token weights.
- [ ] Memory probe for the 20M batch size before launching.
- [x] Add visible-content identity loss (implemented in `e0941d8`; not trained yet).
- [ ] Build a one-board pair dataset with positives and candidate hard negatives.
- [ ] Add a hard-negative validation/filtering layer.
- [ ] Train a single-tower CLS-pooled contrastive baseline.
- [ ] Add material auxiliary head on the pooled content embedding.
- [ ] Run short 10k–20k comparisons with eval every 1k–2k.
- [ ] Add a validation-loss / material-R² early-stop guard.
- [ ] Implement fast resume-data skipping for streaming shards.
