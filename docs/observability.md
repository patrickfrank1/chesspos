# Position Similarity Ranking Runbook

The observability tool now focuses on one question: **does the embedding rank
semantically related positions above unrelated ones?**

Everything is a 5-way ranking task (1 positive + 4 negatives). The report has
three tabs:

1. **Test** — in-game temporal neighbour vs out-of-game negatives.
2. **Motifs** — same tactical theme, different puzzle, vs different themes.
3. **Surrogates** — same structural class vs different structural classes.

Each tab shows top-1/MRR/mean-rank against chance, a definition of the
condition, and three example buckets per condition: **3 best**, **3 random**,
and **3 worst** queries. Each example shows the query and all five candidates;
the positive candidate is outlined in green.

## Run

```bash
uv run python -m src.run.evaluate_similarity \
  --checkpoint data/models/run_d512l8_p1p3_v1_20261026/last.pt \
  --test-dir /tmp/obs_report_data/test \
  --puzzles /tmp/obs_report_data/puzzles \
  --themes fork,pin,skewer,sacrifice,discoveredAttack,backRankMate \
  --n-test 200 --n-puzzles 500 --per-game-cap 6 \
  --rank-k 5 --rank-queries 200 \
  --rank-examples 3 --rank-example-mode matched \
  --device cpu --out-dir data/eval/similarity/run
```

Relevant options:

| Option | Meaning |
|---|---|
| `--rank-k` | Candidates per query (default 5). |
| `--rank-queries` | Queries per theme/class (default 200). |
| `--rank-examples` | Examples per best/random/worst bucket (default 3). |
| `--rank-example-mode` | `random`, `matched`, or `both` examples (default `matched`). |
| `--per-game-cap` | Contiguous test positions sampled per game. Use ≥2 for temporal ranking; for a max ply distance of `N`, use at least `N + 1`. |
| `--max-ply-distance` | Temporal positive is sampled uniformly from same-game positions within this many plies (default 5). |

## Data

- **Test**: a 67-token packed test split. Positive = a random same-game
  position within `--max-ply-distance` plies (default 5); negatives = positions
  from other games. Positions are sampled as contiguous runs per game so the
  candidate pool contains genuinely nearby positions.
- **Motifs**: `data/processed/puzzles` built from the Lichess puzzle CSV.
  Positive = same theme, different puzzle. Themes are sequence-level labels;
  `--puzzle-position first|last|random` controls which position is sampled.
- **Surrogates**: derived from the board. Evaluated on the test positions when
  available, otherwise on all loaded positions.

### Surrogate classes

| Task | Classes |
|---|---|
| Material balance | white advantage (> +2), equal (−2…+2), black advantage (< −2) |
| Game phase | opening (≥24 pieces), middlegame (14–23), endgame (<14) |
| Open files | none, some |
| Isolated / passed / doubled pawns | none, some |
| Pawn islands | low (≤2), mid (3–4), high (≥5) |
| King safety | low (≤1), mid (2–3), high (≥4) |
| Mobility | low (≤20), mid (21–35), high (≥36) |
| Centre control | low (≤1), mid (2–3), high (4) |
| Bishop pair | none, yes |
| In check | no, yes |
| King attack | low (0 attacked squares around a king), mid (1–2), high (≥3) |
| Knight on a/h file | none, some |
| Inactive bishop | none, some (≤2 pseudo-legal moves) |
| Smothered position | no, yes (in check with no legal king move) |

## Random vs matched negatives

- **Random**: negatives are drawn from different classes/games/puzzles.
- **Matched**: negatives are chosen to match the query on the *other*
  surrogates. For a surrogate task, the target surrogate is excluded from
  matching (leave-one-surrogate-out), so the target signal is not controlled
  away.

Chance baselines for `k=5`: top-1 = 0.2, MRR ≈ 0.4567, mean rank = 3.0.

"Best" examples have the positive ranked 1st; "worst" have it ranked last;
"random" examples are sampled from the middle after removing best and worst.
The report shows the definition and class thresholds for every condition.

## Outputs

```
out/
  config.json
  metrics.json          # flat ranking metrics for all tabs
  samples.parquet       # provenance, FENs, themes, surrogates
  embeddings.npy        # L2-normalized float32 embeddings
  ranking_examples.json # query + candidates for every recorded example
  report.html           # focused, self-contained tabbed report
```

## Tests

```bash
uv run pytest test/unit/test_observability_*.py \
  test/integration/test_evaluate_similarity_cli.py
```
