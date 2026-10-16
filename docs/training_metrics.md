# Training metrics: masked token loss and accuracy

Applies to `src/run/train_masked_transformer.py` logging (`val_loss`,
`val_acc`). Both are computed **only over masked tokens** — the positions the
model must reconstruct. Unmasked positions carry target `-100`
(`IGNORE_INDEX`) and are excluded from loss and accuracy.

## val_loss — mean per-token cross-entropy (natural log)

`val_loss = -ln p(correct token)`, averaged over all masked tokens in the
eval batches.

- **Perplexity** is the intuitive reading: `perplexity = exp(val_loss)`. It
  is the effective number of equally likely candidates the model hesitates
  between per masked token. Example: `val_loss=3.46` → perplexity ≈ 31.8.
- Probability assigned to the true token: `p = exp(-val_loss)`. Example:
  `val_loss=3.46` → ≈ 3.2%.
- Baselines: uniform guess over the 806-token vocabulary →
  `ln(806) ≈ 6.69` (matches the loss seen in the first steps); a perfect
  model → 0.
- Monotone and smooth — use it as the primary training/selection signal
  (best checkpoint = lowest `val_loss`).

## val_acc — exact top-1 hit rate

Fraction of masked tokens predicted exactly right (argmax equals target).

- Random baseline: 1/806 ≈ 0.12%.
- Skewed upward by frequent tokens: common piece-square states and special
  tokens recur constantly, so `val_acc` jumps in steps rather than moving
  smoothly. It systematically overstates "position knowledge" — trivial
  special-token predictions (CLS/SEP/PAD context) count the same as a
  piece-square prediction.
- Interpret loss and acc together: `val_loss=3.46, val_acc=0.16` means "of
  100 masked tokens, 16 are exactly right; on the rest the model spreads
  probability as if over ~32 candidates."

## What a misprediction means here

One mispredicted token = one wrong symbol in a position stream — typically a
wrong piece-square state (piece, square, or special token) for a single
board position. Loss treats near-misses and far misses equally (it only sees
the probability of the true token), so use the embedding evaluations
(`docs/plan_gpu_training_vastai.md`, section 6) to judge semantic quality;
loss/acc only measure token-level reconstruction.
