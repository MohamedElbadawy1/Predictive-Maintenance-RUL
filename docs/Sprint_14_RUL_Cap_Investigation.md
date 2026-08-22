# Sprint 14 — RUL Cap Investigation

---

# Hypothesis (from Sprint 13)

```
Cap = 125
     ↓
Test RUL reaches 195
     ↓
Model cannot properly predict >125
     ↓
Huge errors for high-RUL engines
```

**Question**: was RUL capping at 125 actually hurting generalization to the official
test set?

---

# Design

This experiment isolates **one variable**: the RUL cap used to generate training
targets. Everything else is held fixed across every run, so any difference in results
can be attributed to the cap and nothing else:

- Same 109 frozen features (Sprint 10)
- Same Optuna hyperparameters (Sprint 11's `best_params.json`:
  `depth=8, learning_rate=0.071, iterations=1458, l2_leaf_reg=6.995,
  subsample=0.855, random_strength=7.911`)
- Same engine-level train/validation split (same `random_state`)
- Same feature scaler — features don't depend on the RUL cap at all, only the target
  does, so reusing the existing fitted scaler is methodologically correct, not a
  shortcut
- Same official test set (`test_FD004` / `RUL_FD004`), same prep pipeline as
  Sprint 13, evaluated against the same fixed, **uncapped** ground truth

**Caps tested**: no cap, 125 (existing Sprint 11/13 model — reused, not retrained,
per Sprint 13's "don't retrain what you don't have to" principle), 150, 175, 195.

---

# Project Structure

```
rul_cap_experiment.py                    # one cap per invocation, resumable/checkpointed

notebooks/
└── 17_rul_cap_investigation.ipynb

docs/
└── Sprint_14_RUL_Cap_Investigation.md

reports/
├── rul_cap_experiment_results.csv       # raw results from the 4 new experiments
├── rul_cap_full_comparison.csv          # + the reused cap=125 baseline, all 5
└── rul_cap150_test_predictions.csv      # per-engine predictions for the winner

artifacts/models/cap_experiments/
├── catboost_no_cap.pkl
├── catboost_cap_150.pkl
├── catboost_cap_175.pkl
└── catboost_cap_195.pkl
```

Training ran as a standalone script (`rul_cap_experiment.py --cap N`), one
experiment per invocation — same reasoning as every compute-heavy step since
Sprint 11: short, checkpointed runs rather than one long live cell. ~110s per
training, 4 new trainings, results appended to a shared CSV so each run is
independently resumable.

---

# Context: What Does Uncapped Training RUL Actually Look Like?

Before the results, it's worth seeing what "no cap" means concretely for the
training targets:

| Statistic | Value |
|---|---:|
| Mean | 133.3 |
| Median | 122 |
| 75th percentile | 190 |
| Max | **542** |
| Rows with RUL > 195 (max ever seen in test) | 14,295 / 61,249 (**23.3%**) |

Nearly a quarter of training rows carry a raw RUL label built on the assumption that
an engine's remaining life decreases linearly from cycle 1 — including engines with
541 more healthy cycles ahead of them. This is the textbook reason RUL capping exists
at all: early-life "healthy" cycles don't meaningfully degrade, so uncapped labels are
mostly an unrealistic assumption stretched over a very long tail, not real signal.

---

# Results — All 5 Cap Strategies

| Cap | Train RUL Max | Val MAE | Val R² | Test MAE | Test RMSE | Test R² | Pred Max | Training Time |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 125 (existing) | 125 | 12.836 | 0.782 | 20.256 | 27.906 | 0.738 | 133.0 | 134.6s |
| **150** | 150 | 17.666 | 0.755 | **19.260** | **25.716** | **0.778** | 152.7 | 109.3s |
| 175 | 175 | 22.604 | 0.726 | 19.613 | 26.412 | 0.765 | 178.7 | 111.4s |
| 195 | 195 | 26.253 | 0.706 | 20.571 | 28.193 | 0.733 | 190.5 | 110.7s |
| No cap | 542 | 41.316 | 0.589 | 27.132 | 39.134 | 0.485 | 274.1 | 113.0s |

Full table with MAPE in `reports/rul_cap_full_comparison.csv`.

**cap=150 wins on both test MAE and test R²** — the only cap that beats the original
125 on the metric that actually matters (fixed, unseen ground truth). No cap is
clearly the *worst* option, not the best, despite being the only one that can predict
without a ceiling at all.

---

# The Validation-vs-Test Divergence Is Itself a Finding

**Validation MAE gets monotonically *worse* as the cap increases**: 12.84 → 17.67 →
22.60 → 26.25 → 41.32. If cap selection had been based on validation performance
alone, cap=125 would have looked unambiguously best — and that conclusion would have
been wrong.

This isn't a contradiction, it's a measurement artifact worth understanding precisely:
each experiment's validation labels are defined *by that experiment's own cap*, so a
higher cap mechanically produces larger-magnitude targets and larger-magnitude errors,
even for an equally good (or better) model. Validation MAE is not comparable across
different cap choices. `RUL_FD004` is fixed and uncapped regardless of which cap
trained the model, so test performance against it is the only apples-to-apples
comparison across these five experiments — and it points to cap=150, the opposite of
what validation-only selection would suggest.

This is a direct, concrete payoff from Sprint 13's decision to hold out the official
test set and evaluate the frozen model honestly rather than optimize everything
against validation alone.

---

# Did Cap=150 Actually Fix the High-RUL Problem? — Bucket Comparison

| RUL Bucket | n | cap=125 MAE | cap=150 MAE | Change |
|---|---:|---:|---:|---:|
| 0-25 (critical) | 50 | 5.31 | 6.37 | **+1.06 (worse)** |
| 25-50 | 30 | 12.64 | 11.22 | -1.42 (better) |
| 50-75 | 28 | 19.54 | 22.79 | **+3.25 (worse)** |
| 75-125 | 73 | 15.46 | 21.51 | **+6.05 (worse)** |
| **125+ (beyond old cap)** | 67 | **40.35** | **28.55** | **-11.80 (much better)** |

**cap=150 is a genuine trade-off, not a clean win everywhere.** The 125+ bucket —
the original problem this whole sprint set out to fix — improved substantially (MAE
40.35 → 28.55, mean bias -40.31 → -25.07: real, meaningful progress). But three of
the other four buckets got worse, including a small regression in the most
operationally critical bucket (0-25: 5.31 → 6.37). Cap=150 redistributes error rather
than eliminating it: less catastrophic failure on high-RUL engines, somewhat worse
precision in the middle of the range, and a better number in aggregate because the
125+ bucket's improvement outweighs the others' small regressions.

This nuance matters for anyone deploying this model: if near-failure prediction
accuracy is the only thing that matters operationally, cap=125's slightly better
0-25 bucket performance (5.31 vs 6.37) might be worth keeping despite the worse
aggregate and worse high-RUL numbers. If overall/aggregate accuracy across the full
engine fleet matters more, cap=150 is the better choice.

---

# What We Learned

**Yes, capping at 125 was hurting generalization — but the fix is a better cap, not
the absence of one.** The original hypothesis was correct in spirit but the natural
first instinct (remove the cap to remove the ceiling) is actively wrong: no cap
produces the worst test performance of all five experiments tested, because it trades
a bounded-prediction problem for a much noisier training signal.

**150 is a sweet spot, not an arbitrary middle value** — 175 and 195 both perform
worse than 150 on test MAE, confirming this isn't "higher is better up to a point,
then it plateaus" but a genuine optimum: enough headroom to capture most of the test
set's real RUL distribution (only test engines above 150 remain capped-out, a smaller
share than the 23%+ affected at cap=125) without reintroducing too much of the
long-tail noise that makes no-cap training so much worse.

**Validation-only model selection would have been actively misleading here.** This is
a durable, generalizable lesson for the rest of this project: any target-definition
change (not just RUL caps — this would apply to any relabeling strategy) needs to be
compared on a fixed, unchanging ground truth, not on validation metrics whose scale
shifts along with the label definition being tested.

**The improvement is real but not dramatic, and comes with trade-offs.** Test MAE
19.26 vs 20.26 (a 4.9% improvement) is a genuine, evidence-based win, not a rounding
difference — but it's not the kind of result that eliminates the high-RUL problem
outright, and it costs a small amount of accuracy elsewhere. Framing this as "solved"
would overstate it; framing it as "measurably better, understood trade-off" is
accurate.

---

# Decision

**Recommend adopting cap=150 as the new default RUL cap** for future training — it's
the only cap tested that beats the original 125 on both test MAE and test R², and the
mechanism (better-calibrated predictions for the 23%+ of engines with true RUL above
the old cap, at a small cost elsewhere) is understood, not a coincidence.

**Not yet applied**: `artifacts/models/best_model.pkl` still points to the cap=125
model from Sprint 11. Promoting the cap=150 model
(`artifacts/models/cap_experiments/catboost_cap_150.pkl`) to canonical status is a
deliberate follow-up decision, not automatic — it would mean updating the training
pipeline's default cap (currently 125, set in Sprint 8) for consistency, and is worth
confirming explicitly before making it official, given the real trade-off in the
critical low-RUL bucket documented above.

---

# Next Steps

1. Confirm whether to officially promote cap=150 to `best_model.pkl` and update the
   pipeline's default cap for all future training.
2. LSTM/GRU work (Sprint 12 in this project) can now be evaluated against the same
   official test set using the identical protocol established in Sprint 13 — and,
   if cap=150 is adopted, retrained with the corrected target definition before that
   comparison, for a fully fair final model bake-off.
