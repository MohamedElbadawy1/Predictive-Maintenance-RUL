# Sprint 15 — LSTM on Corrected Labels, Evaluated on the Official Test Set

---

# Goal

Close two gaps left open by prior sprints, in one pass:

1. The LSTM baseline (Sprint 12) was trained on the flawed `cap=125` labels
   Sprint 14 proved were suboptimal.
2. The LSTM baseline was **never evaluated on the official test set** — every
   LSTM-vs-CatBoost comparison so far mixed an internal-validation LSTM number
   against CatBoost numbers from entirely different (and more rigorous) evaluation
   protocols.

This sprint retrains the LSTM on `cap=150` and evaluates it with the exact same
test-set protocol CatBoost has used since Sprint 13 — producing the first genuinely
fair comparison between the two model families in this project.

---

# A Note on This Session's Environment

Partway through this project, the sandbox environment this work runs in reset
completely — every trained model, cached dataset, and fitted scaler from every prior
sprint was gone from disk (this has happened once before, during Sprint 12). What
survived: every piece of source code delivered to the user across every sprint, and
the notebooks' saved outputs (real numbers from when they originally ran).

Rather than treat this as a setback to work around quietly, it was used as a
correctness check: the repository was rebuilt from the delivered source files, the
frozen Sprint 10 feature list was reconstructed deterministically (same code, same
data — the "Remove Rolling Features" experiment's outcome doesn't depend on random
chance), and the cap=150 CatBoost model was retrained from Sprint 11's frozen
hyperparameters. **It reproduced bit-for-bit identical metrics to the original run**
(MAE 17.665746 / 19.260494, matching to the sixth decimal place) — confirming the
entire pipeline is genuinely deterministic and the rebuild is trustworthy, not an
approximation.

---

# Project Structure

```
notebooks/
└── 18_lstm_cap150_test_evaluation.ipynb

docs/
└── Sprint_15_LSTM_Cap150_Test_Evaluation.md

reports/
├── lstm_cap150_test_predictions.csv
└── sprint15_fair_comparison.csv
```

`prepare_lstm_sequences.py` and `train_lstm.py` both gained a `--cap` argument
(previously hardcoded to whatever cap the cached checkpoint happened to have). The
RUL cap is no longer trusted from a cached checkpoint — it's recomputed fresh from
raw data every time, matching `rul_cap_experiment.py`'s approach, so a stale cap can
never silently leak into a new experiment.

`SequenceGenerator` gained a new method: `get_last_window()`. Sprint 12's
`transform()` builds *every* sliding window for training; test-time inference needs
only the single most recent window per engine, with no target column at all (test
engines have no per-row RUL). Same "skip, never pad" principle as training: an engine
with fewer than `window_size` cycles is excluded, not padded with values the model
never trained on.

---

# 1. Retraining on Cap=150

Same 25 raw features (Sprint 12's ablation finding still holds — lag/diff features
remain the wrong input for a sequence model regardless of which cap labels it),
same architecture (`LSTM(64) → Dropout(0.2) → Dense(1)`), same
`max_epochs=40, patience=8, batch_size=256`. Only the RUL cap changed.

**Training stopped early at epoch 32** (vs. epoch 36 for the original cap=125 raw
run) — converged slightly faster with the corrected labels.

---

# 2. Evaluating on the Official Test Set — New Work, Not a Rerun

Sprint 13's test-evaluation protocol extracts the **last row** per test engine —
correct for CatBoost, which predicts from a single snapshot. An LSTM needs a
**30-cycle window** ending at each engine's last available cycle, which required
building `SequenceGenerator.get_last_window()` — genuinely new code, not a
repurposing of what existed.

**11 of 248 test engines (4.4%) have fewer than 30 cycles of history** (shortest:
19 cycles) and were excluded — consistent with the project's established
no-padding principle. **For a fair comparison, CatBoost's numbers below are computed
on this identical 237-engine subset**, not the full 248 — otherwise a difference in
which engines got evaluated could masquerade as a difference in model quality.

---

# 3. The Fair Comparison

Same cap (150), same 237 test engines, same official ground truth:

| Model | n | MAE | RMSE | R² | MAPE |
|---|---:|---:|---:|---:|---:|
| **CatBoost (cap=150)** | 237 | **19.015** | **25.630** | **0.768** | **30.05%** |
| LSTM raw (cap=150) | 237 | 22.527 | 30.350 | 0.675 | 37.45% |

CatBoost wins decisively — an 18.5% MAE gap, smaller than the misleading ~43% gap
implied by comparing Sprint 12's validation-only LSTM number against Sprint 13's
properly test-evaluated CatBoost number, but still a clear, real win for CatBoost.

---

# 4. Validation-to-Test Generalization Gap

| Model | Val MAE | Test MAE | Gap |
|---|---:|---:|---:|
| CatBoost (cap=150) | 17.666 | 19.015 | **+7.6%** |
| LSTM raw (cap=150) | 22.180 | 22.527 | **+1.6%** |

A genuinely interesting, unexpected result: **the LSTM's validation number was a much
more honest preview of its test performance than CatBoost's was.** CatBoost is more
accurate overall, but its validation metric understated real-world error by a wider
margin. This doesn't change which model is better today, but it's worth tracking as
tuning continues — a model whose validation score can be trusted is easier to make
decisions from.

---

# 5. Bucket Analysis — Both Models, Same 237 Engines

| RUL Bucket | CatBoost MAE | LSTM MAE |
|---|---:|---:|
| 0-25 (critical) | **6.37** | 9.05 |
| 25-50 | **11.22** | 12.03 |
| 50-75 | 22.79 | **22.50** |
| 75-125 | **21.31** | 27.65 |
| 125+ (beyond old cap) | **29.31** | 33.31 |

CatBoost wins 4 of 5 buckets outright; the 50-75 bucket is close to a tie (LSTM
marginally ahead, 22.50 vs 22.79). No bucket shows a meaningful LSTM advantage — this
is a clean win for CatBoost across the board, not just in aggregate.

---

# What We Learned

**This sprint closed a real methodological gap, independent of who won.** Every
LSTM-vs-CatBoost comparison before this one compared numbers that weren't produced
the same way. That's fixed now, and it matters for every future comparison
(Sprint 16's GRU inherits a validated protocol from day one).

**CatBoost's win is real and now properly measured, not an artifact of unfair
evaluation.** The gap shrank once the comparison became fair (43% → 18.5%), which is
itself informative — some of the LSTM's apparent weakness in Sprint 12 really was
measurement unfairness, not model weakness. But a real, sizeable gap remains even
after correcting for that.

**The generalization-gap finding is a genuine surprise worth carrying forward.** It
wasn't hypothesized going in — the LSTM's small validation-to-test gap only showed up
once both models were finally evaluated the same way. This kind of result is exactly
why building the fair-comparison infrastructure (rather than assuming Sprint 12's
numbers were good enough) was worth the sprint.

**The window-size limitation (11 excluded test engines) is a structural trade-off,
not a bug**, and it's a concrete design question for Sprint 16: a smaller window
would recover those 11 engines at the cost of less temporal context per prediction.

---

# Decisions

- `artifacts/models/best_model.pkl` (CatBoost, cap=150) remains the project's leading
  model. This sprint's fair comparison confirms that leadership, rather than assuming
  it.
- The LSTM (raw features, cap=150) is saved for reference:
  `artifacts/models/lstm_baseline_raw_cap150.keras`, alongside its scaler
  (`scalers/lstm_feature_scaler_raw_cap150.pkl`) and training state.
- `SequenceGenerator.get_last_window()` is now available for any future sequence
  model's test-set evaluation (GRU in Sprint 16, and beyond) — this was the missing
  piece, not something to rebuild per model.
- `prepare_lstm_sequences.py` and `train_lstm.py`'s `--cap` argument means any future
  cap change (if Sprint 14's investigation is ever revisited or extended) no longer
  requires editing the scripts by hand.
- Next: **Sprint 16 = GRU baseline**, starting from the raw 25-feature set and
  cap=150 labels from day one — no ablation or cap-correction retrofit needed, both
  questions are already answered. The fair test-set comparison protocol
  (`get_last_window`, same-subset CatBoost comparison) is ready to reuse directly.
