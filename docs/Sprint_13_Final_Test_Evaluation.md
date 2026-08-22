# Sprint 13 — Final Test Evaluation & Error Analysis

*(Numbered Sprint 13 in this project's file sequence — "Sprint 12" in the original
brief was already used for the LSTM baseline work, so this continues the sequence
without a naming collision.)*

---

# Goal

Up through Sprint 11, every metric in this project came from an internal
train/validation split of `train_FD004`. `test_FD004` and `RUL_FD004` were never
touched. This sprint is the first and only time they're used: a clean, honest
measurement of how well the **frozen** Sprint 11 CatBoost model
(`artifacts/models/best_model.pkl`) generalizes to engines it has never seen in any
form — not in training, not in validation, not in feature selection, not in
hyperparameter tuning.

**No retraining happens in this sprint.** The model is loaded exactly as Sprint 11
saved it and used purely for inference.

```
NASA FD004
     ↓
Data Loading
     ↓
Validation
     ↓
RUL Generation (train only)
     ↓
Feature Engineering
     ↓
Feature Selection
     ↓
Train / Validation
     ↓
Hyperparameter Optimization
     ↓
FINAL MODEL
     ↓
UNSEEN TEST SET
     ↓
FINAL EVALUATION
```

---

# Project Structure

```
notebooks/
└── 16_final_test_evaluation.ipynb

docs/
└── Sprint_13_Final_Test_Evaluation.md

reports/
├── final_val_vs_test_comparison.csv
└── final_test_predictions.csv          # per-engine: actual, predicted, error
```

No new `src/` modules were needed — this sprint only chains together the classes
already built in Sprints 8-11 (`DataLoader`, `DataValidator`, `FeatureEngineer`,
`FeatureReducer`, `BaseTrainer`, `RegressionEvaluator`) in inference mode, using the
saved scaler and saved feature list rather than fitting anything new.

---

# 1. Loading the 3 Datasets

| Dataset | Shape |
|---|---|
| `train_FD004` | (61,249, 26) |
| `test_FD004` | (41,214, 26) |
| `RUL_FD004` | (248, 1) |

All three pass `DataValidator` (train and test fully valid; RUL file has one
warning — duplicate rows found, which is expected and harmless since many engines
legitimately share the same remaining-life value). 248 unique test engines, matching
248 RUL rows exactly, engine IDs 1-248 — confirming standard NASA ordering (`RUL_FD004`
row *i* corresponds to `unit_number` *i*) before relying on it for alignment.

---

# 2. Preparing the Test Data

```
test_FD004
     ↓
Feature Engineering (same rolling/lag/diff config as training)
     ↓
Take the LAST cycle per engine (the "current" snapshot to predict from)
     ↓
Scale using the scaler FITTED ON TRAINING (never refit here)
     ↓
Select the frozen 109 features
     ↓
X_test
```

**Nothing is fit on the test set** — not the scaler, not feature selection. `test_FD004`
engines are truncated mid-life (not run to failure), so there's no RUL to *generate*
here the way Sprint 8 did for training; `RUL_FD004` directly gives the true remaining
life at each engine's truncation point, which is what the prediction from that
engine's last available cycle gets compared against.

One detail that had to be gotten exactly right: `StandardScaler` (used internally by
`FeatureScaler`) was fit on 151 named columns during Sprint 8 and validates both count
and identity of columns on `transform()`. The saved scaler's own
`feature_names_in_` was used as the authoritative column list and order for
reconstructing the test feature matrix — not re-derived by hand — to guarantee an
exact match. After feature engineering, scaling, and reducing to the Sprint 10 frozen
list (`FeatureReducer.load_selected_features`), `X_test` is `(248, 109)` — one row per
test engine. Zero NaNs in the final feature matrix (every test engine has more than
the 3 cycles needed for the longest lag feature).

---

# 3. Generating Predictions

```python
trainer = BaseTrainer.load(MODELS_DIR / "best_model.pkl")
y_pred = trainer.predict(X_test)
```

Straight inference — no retraining, no refitting.

---

# 4. Test Set Metrics

| Metric | Value |
|---|---:|
| MAE | 20.256 |
| RMSE | 27.906 |
| R² | 0.7380 |
| MAPE | 27.48% |

`y_true` (actual RUL) ranges **[6, 195]**. `y_pred` ranges **[2.5, 133.0]** — the first
clue to what's driving the errors, expanded on below.

---

# 5. Validation vs. Test — The Real Generalization Number

| Dataset | MAE | RMSE | R² |
|---|---:|---:|---:|
| Validation (Sprint 11) | 12.836 | 19.092 | 0.7818 |
| **Test (unseen, official)** | **20.256** | **27.906** | **0.7380** |
| Change | +57.8% | +46.2% | -0.044 |

A real, sizeable generalization gap — not a rounding difference, and not something to
explain away. The rest of this sprint identifies *why*, with evidence, rather than
treating it as an unexplained black box.

---

# 6. Error Analysis

## Overall error distribution

- **Mean error (bias): -7.71** (predictions run low on average)
- **Median error: -1.96** (much smaller than the mean — a small number of very large
  under-predictions are pulling the mean down, not a broad systematic shift)
- Over-predicted: 42.7% of engines | Under-predicted: 57.3% of engines

## Engines with the largest errors

Every one of the 10 largest errors is a large **under**-prediction, and every one is
on a high-true-RUL engine:

| Unit | Actual RUL | Predicted RUL | Error |
|---:|---:|---:|---:|
| 153 | 173 | 90.5 | -82.5 |
| 54 | 176 | 98.3 | -77.7 |
| 95 | 171 | 97.0 | -74.0 |
| 127 | 195 | 121.8 | -73.2 |
| 122 | 192 | 121.8 | -70.2 |
| 19 | 194 | 125.4 | -68.6 |
| 228 | 189 | 121.4 | -67.6 |
| 246 | 194 | 127.3 | -66.7 |
| 83 | 172 | 108.4 | -63.6 |

This is not scattered noise — it's one identifiable pattern repeated nine times.

## Performance by true-RUL range

| RUL Range | n | MAE | Mean Error |
|---|---:|---:|---:|
| 0-25 (critical) | 50 | **5.31** | +2.14 |
| 25-50 | 30 | 12.64 | +8.60 |
| 50-75 | 28 | 19.54 | +13.51 |
| 75-125 (within training range) | 73 | 15.46 | +0.62 |
| **125+ (beyond training cap)** | 67 | **40.35** | **-40.31** |

This table is the core finding of the sprint.

---

# What We Learned

**The dominant cause of the generalization gap is identified, not mysterious: the
training RUL cap (125) doesn't match test-set reality.** True test RUL goes up to 195;
predictions never exceed 133. CatBoost, like most tree ensembles, essentially cannot
extrapolate past the range of targets it was trained on — every leaf value is an
average of training labels, which were capped at 125. Engines with true RUL above 125
are structurally unpredictable for this model as currently trained, and the 125+
bucket's MAE of 40.35 (vs. 5.31 for the most critical bucket) is nearly **8× worse**,
entirely explained by which side of the cap an engine falls on.

**A second, subtler pattern**: for true RUL in the 25-75 range, the model
systematically *over*-predicts (mean error +8.60 to +13.51). Combined with the
under-prediction above the cap, this is classic regression-to-the-mean behavior from a
capped-target gradient-boosted model — predictions get pulled toward the dense middle
of the training distribution from both directions, compressing the effective
prediction range relative to the true one.

**The good news is that this lines up well with operational priorities.** The
best-performing bucket is exactly the one that matters most for maintenance
decisions — engines with 0-25 cycles of true remaining life (MAE 5.31, smallest bias of
any bucket). The model is least reliable for engines that aren't close to needing
attention yet, which is a far less costly place to be wrong than the reverse. A small
optimistic bias in the critical bucket (+2.14 cycles) is worth noting for anyone using
this for real scheduling decisions, but is small relative to the granularity of
maintenance planning.

**The validation metrics from Sprint 11 (MAE 12.84, R² 0.782) overstated real-world
performance** — not because anything was done incorrectly in Sprint 11, but because
validation engines came from the same run-to-failure training distribution the model
was tuned against, while test engines are genuinely different truncated trajectories
including some with atypically high remaining life. This is exactly why holding out
`test_FD004`/`RUL_FD004` until now, and evaluating the frozen model rather than
retraining first, was the right call — a retrained model would have blurred this
measurement.

---

# Final Model Decision

**Can the current CatBoost model generalize to completely unseen engines? Yes, with a
real and now-quantified gap — not a failure, but not the validation numbers either.**

- Test MAE (20.26) is 58% higher than validation MAE (12.84). This is a genuine
  generalization gap with an identified, actionable cause (the RUL cap), not
  unexplained variance.
- The model is usable today for its most operationally important case: predicting
  near-failure engines, where it performs best (MAE 5.31).
- **The single most actionable fix identified this sprint: reconsider the RUL cap.**
  A cap of 125 was a reasonable choice given the training data's own distribution
  (Sprint 8), but the test set reveals real engines with remaining life up to 195 —
  a mismatch between the training assumption and test-set reality that directly causes
  the largest errors in the entire evaluation. This is a training-time decision to
  revisit, not a sign the model architecture or feature set is wrong.

---

# Decisions

- `artifacts/models/best_model.pkl` is unchanged by this sprint — evaluation only, as
  the brief required. No retraining on train+validation combined happened before this
  measurement, preserving a clean read on current-pipeline generalization.
- The RUL cap (currently 125, set in Sprint 8) is flagged as the top candidate for
  revisiting in a future tuning pass, backed by direct evidence from this sprint
  rather than a guess.
- Per-engine predictions and errors are saved in full
  (`reports/final_test_predictions.csv`) for any further analysis without needing to
  rerun inference.
- This sprint's numbers — not Sprint 11's validation numbers — are the ones that
  should be quoted as this model's real-world performance going forward.
- Next: LSTM/GRU work (already underway in this project's Sprint 12) can now be
  evaluated against this same official test set using the identical protocol
  established here, for a fully fair final model comparison.
