# Sprint 18 — Bucket-Level Error Analysis (Canonical Model)

---

# Goal

Aggregate MAE hides where a model actually struggles. Every real insight in this
project so far came from breaking error down by true RUL range (Sprint 13 found the
RUL-cap problem this way; Sprint 14 confirmed the fix this way). This sprint applies
the same method to the new canonical model (regime-aware CatBoost, Sprint 17) to find
out whether its improvement is spread evenly or concentrated somewhere specific.

---

# A Verification Note, Stated Plainly

The numbers in this document come from running `bucket_error_analysis.py` against
**a lightly-tuned (2-trial) verification model**, not the user's real, fully-tuned
regime-aware CatBoost (test MAE 19.133, from the real 20+ trial run in Sprint 17).
This sandbox does not have access to that real model file — only the user's own
machine does.

**What this run confirms**: the script itself is correct — it applies regime
normalization to test data properly (`.transform()`, not refit), produces sensible
metrics, and the bucket-level pattern is directionally consistent with prior sprints.
**What it does not yet confirm**: the exact magnitude of the current canonical model's
bucket-level performance. Whoever runs `bucket_error_analysis.py` against the real
promoted `best_model.pkl` should replace the numbers below with the real result.

---

# Verification Run — What the Script Produces

Overall (this sandbox's under-tuned model): MAE 19.307, RMSE 25.658, R² 0.779,
MAPE 28.23%.

| RUL Bucket | n | MAE | Mean Error (bias) |
|---|---:|---:|---:|
| 0-25 (critical) | 50 | 5.97 | +2.79 |
| 25-50 | 30 | 11.66 | +5.19 |
| 50-75 | 28 | 24.03 | +18.36 |
| 75-125 | 73 | 21.52 | +8.45 |
| 125+ (beyond old cap) | 67 | 28.30 | -25.08 |

Overall bias: mean error -1.02, median +0.69. 51.6% of engines over-predicted, 48.4%
under-predicted — roughly balanced overall, consistent with the mean/median being
close to zero, unlike the earlier cap=125 model which was heavily bias-skewed.

## Comparison to the pre-regime-aware cap=150 model (Sprint 14/15 real numbers)

| RUL Bucket | Pre-regime-aware MAE | This run's MAE |
|---|---:|---:|
| 0-25 (critical) | 6.37 | **5.97** (better) |
| 25-50 | 11.22 | 11.66 (slightly worse) |
| 50-75 | 22.79 | 24.03 (worse) |
| 75-125 | 21.31 | 21.52 (about the same) |
| 125+ | 29.31 | **28.30** (better) |

**Not a uniform win across every bucket** — regime-aware normalization (at least in
this under-tuned verification run) helps the two extremes (near-failure and
beyond-old-cap engines) but is roughly flat or slightly worse in the middle of the RUL
range. This mirrors a pattern already seen once before in this project (Sprint 14's
cap=150 vs. cap=125 comparison also redistributed error rather than improving
everywhere) — worth checking whether the same is true at the real, fully-tuned
model's numbers.

---

# Top-10 Largest Errors (Verification Run)

Every one of the ten largest errors is on an engine with true RUL above 150 — the
model still cannot predict much beyond its training range, the same fundamental
limitation identified in Sprint 13, just shifted from ~125 to ~150.

---

# What This Means for the Real Model

The question this sprint set out to answer — is the Sprint 17 improvement spread
evenly or concentrated? — has a **provisional answer from this verification run**:
concentrated at the extremes (critical near-failure engines, and engines beyond the
old cap), not evenly spread, and not obviously an improvement in the 50-125 middle
range. This should be re-checked against the real 19.133 model before treating it as
confirmed.

---

# Next Steps

1. **Run `bucket_error_analysis.py` against the real promoted `best_model.pkl`** (the
   19.133 model) and replace the numbers in this document with the real result — this
   is the one concrete action item from this sprint.
2. If the real numbers confirm the same pattern (better at extremes, flat/worse in
   the middle), that's a specific, actionable target for a future improvement — e.g.
   investigating why regime normalization doesn't help the 50-125 range as much, or
   whether a different regime count would help there specifically.
3. If the real numbers look meaningfully different from this verification run, that's
   itself worth noting — it would mean the exact hyperparameters (not just regime
   normalization as a concept) materially change *where* the model's errors land, not
   just their average size.
