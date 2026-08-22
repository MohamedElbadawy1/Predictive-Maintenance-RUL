# Sprint 12 — LSTM Baseline

---

# Goal

Build a simple, deliberately unoptimized LSTM baseline on `train_FD004` and compare it
fairly against the Optuna-tuned CatBoost model from Sprint 11 — same engine split, same
metrics, same everything except the model itself.

```
train_FD004
     ↓
   RUL
     ↓
 Features
     ↓
Train / Validation
     ↓
Sequence Generation
     ↓
LSTM Baseline
     ↓
Evaluation
```

`test_FD004` / `RUL_FD004` are **not touched** in this sprint. They stay reserved for
the final comparison once every candidate model (CatBoost, LSTM, GRU, and whatever
tuned versions follow) has been through the same validation process.

---

# Project Structure

```
src/
├── preprocessing/
│   └── sequence_generator.py       # SequenceGenerator: rows -> sliding windows
└── deep_learning/
    ├── lstm_model.py                # build_lstm_baseline()
    └── dl_trainer.py                 # DLTrainer: Keras version of BaseTrainer

notebooks/
└── 15_lstm_baseline.ipynb

docs/
└── Sprint_12_LSTM_Baseline.md

prepare_lstm_sequences.py             # standalone data-prep + sequence caching script
train_lstm.py                         # standalone, resumable LSTM training script
```

`notebooks/13_feature_selection.ipynb` and `14_hyperparameter_optimization.ipynb` were
already taken by Sprints 10-11, so this sprint continues at `15_lstm_baseline.ipynb`.

---

# 1. Preparing the Data

The saved `train_prepared.csv` / `validation_prepared.csv` from Sprint 8 dropped
`unit_number` — reasonable for CatBoost (which treats every row independently), but
useless here: `SequenceGenerator` needs the engine ID to group cycles and the exact
same engine split CatBoost trained on to make the comparison fair. So Sprint 12 rebuilds
from `artifacts/data/features/train_features.csv` (the engineered-features checkpoint,
which still has `unit_number`), then re-applies:

- The same 109 frozen features from Sprint 10 (`selected_features.json`)
- The same engine-level train/validation split as every CatBoost sprint (identical
  `DataSplitter` config and `random_state` — same 199 train engines, same 50 validation
  engines)
- A fresh `FeatureScaler`, fit on the training split only

## A real bug: NaN loss on the first training attempt

Lag and diff features are undefined for the first 1-3 cycles of every engine — there's
nothing yet to lag or difference against. CatBoost handles this natively (its splits
just route NaN rows to whichever side improves the loss). Keras does not: **a single
NaN anywhere in the input silently poisons the entire loss to `nan`**, and the first
real training attempt hit exactly this — `loss: nan, val_loss: nan` from epoch 1.

Fixed by forward-filling then back-filling each engine's features before scaling
(`train_df.groupby(engine).transform(lambda g: g.ffill().bfill())`), so the earliest
cycles of every engine get a defined value instead of corrupting the whole run.
29,253 NaN values in the training split, 7,350 in validation — all filled, confirmed
zero remaining before scaling.

---

# 2. Sequence Generation

`SequenceGenerator` (`src/preprocessing/sequence_generator.py`) converts per-cycle rows
into fixed-size sliding windows per engine:

```python
generator = SequenceGenerator(window_size=30)
X_seq, y_seq, engine_ids = generator.transform(df, feature_columns=final_features)
# X_seq.shape == (n_samples, 30, 109)
```

- Groups by `unit_number`, sorts by `time_in_cycles` within each group (never assumes
  the input is already sorted)
- Each window's target is the RUL at the window's *last* cycle: cycles 1-30 predict RUL
  at cycle 30, cycles 2-31 predict RUL at cycle 31, and so on
- Windows never span two engines — building them per-group makes this structural, not
  just a rule that's followed
- An engine with fewer cycles than `window_size` is skipped (not padded) — kept as a
  simple, honest baseline behavior

With `window_size=30`: **43,523 training sequences**, **10,505 validation sequences**,
shape `(30, 109)` each. Zero engines were dropped — every one of the 199 train and 50
validation engines in FD004 has well over 30 cycles.

---

# 3. The LSTM Baseline

```
Input(30, 109) → LSTM(64) → Dropout(0.2) → Dense(1) → RUL
```

Deliberately minimal, per the sprint brief ("don't optimize it yet"): one LSTM layer,
one dropout layer, one dense output unit. Adam optimizer, MSE loss, MAE tracked as a
secondary metric. This is Sprint 13+'s starting point, not a tuned model — no
architecture search, no stacked layers, no bidirectional variant, no attention.

---

# 4. Training — Why a Standalone Script, Not the Notebook

Same reasoning as Sprint 11's Optuna search: training live inside a Jupyter kernel
process risks hitting execution-timeout walls on long-running compute in this sandbox
(single CPU core). Rather than repeat that friction, training ran as a standalone,
resumable script — `train_lstm.py` — checkpointing the model and a training-state JSON
(epoch count, best `val_loss`, early-stopping patience counter) after **every single
epoch**. Each invocation resumes exactly where the last one stopped, including
early-stopping state:

```bash
python train_lstm.py --n-epochs 5   # trains up to 5 more epochs, exits
python train_lstm.py --n-epochs 8   # resumes from the checkpoint
```

Settings: `max_epochs=40`, `patience=8` (on validation loss), `batch_size=256`. Per-epoch
time ranged roughly 8-14s on this hardware — much faster and more consistent than
CatBoost's `depth=9/10` trials in Sprint 11, so no search-space bounding was needed here.

**Training stopped early at epoch 24.** Full history:

| Epoch | Train MAE | Val MAE | Val Loss (MSE) |
|---:|---:|---:|---:|
| 1 | 75.59 | 67.59 | 5968.0 |
| 5 | 48.33 | 45.68 | 2584.6 |
| 10 | 29.64 | 29.22 | 1088.2 |
| 16 | 14.50 | 19.72 | **619.8 (best val_loss)** |
| 20 | 10.93 | 18.80 | 661.0 |
| 24 | 9.36 | **18.34 (best val_MAE)** | 696.0 |

Full history in `notebooks/15_lstm_baseline.ipynb`.

**Worth noting explicitly:** the checkpointing callback saves the model after every
epoch (overwriting), tracking early-stopping patience against `val_loss` (MSE) — the
model's compiled loss, which is standard practice. It does not separately preserve the
single best-val_loss epoch's weights (no `restore_best_weights`-style mechanism). In
this run that turned out not to matter: epoch 24, the final checkpoint, also happens to
have the *lowest validation MAE* of the entire run (18.34) — the metric this sprint
actually compares on — even though epoch 16 had marginally lower validation MSE. This
was fortunate, not guaranteed; adding a proper best-checkpoint save (by val MAE
specifically) is a clear improvement for Sprint 13.

**Train MAE falling every single epoch (75.6 → 9.4) while validation MAE plateaus and
drifts starting around epoch 17-20 is textbook overfitting** — exactly what early
stopping exists to catch, and did.

---

# 5. Results — CatBoost vs. LSTM (109 Features)

| Model | MAE | RMSE | R² | MAPE | Training Time |
|---|---:|---:|---:|---:|---:|
| **CatBoost (Optuna-optimized, Sprint 11)** | **12.836** | **19.092** | **0.782** | — | 134.6s |
| LSTM Baseline (109 features) | 18.345 | 26.382 | 0.599 | 40.53% | 269.6s |

CatBoost wins decisively on every metric, in roughly half the training time. Full
results saved to `reports/lstm_baseline_results.csv` and
`reports/catboost_vs_lstm_comparison.csv`.

---

# 6. Ablation — Do the Lag/Diff Features Even Help an LSTM?

A fair challenge came up after the first pass: the 109 features frozen in Sprint 10
(63 lag features, 21 diff features included) were selected **for CatBoost** — a
row-by-row model with no memory of its own, where lag features were the only way to
give it temporal context at all. An LSTM doesn't have that problem. It already
receives the actual sequence of raw values across the 30-cycle window and is designed
to learn temporal patterns from that sequence directly.

Worse, feeding lag features into an LSTM window creates real redundancy: at window
position `t`, the `lag_1` feature equals the raw value at `t-1` — which the LSTM also
sees directly at window position `t-1`, in the very same input tensor. The same piece
of information shows up twice, encoded two different ways, adding dimensionality
without adding information.

**This was tested, not just argued.** Rerunning the identical pipeline — same engine
split, same window size, same architecture (`Input → LSTM(64) → Dropout(0.2) →
Dense(1)`), same training settings (`max_epochs=40`, `patience=8`, `batch_size=256`) —
with only the 25 true raw features (21 sensors, 3 operational settings,
`time_in_cycles`, via `FeatureCategorySelector.only(columns, ["raw", "time"])`, no
lag/diff/rolling) instead of the 109 CatBoost-tuned features:

```bash
python prepare_lstm_sequences.py --feature-set raw
python train_lstm.py --feature-set raw --n-epochs 5   # resumable, same pattern as before
```

No NaN filling was actually needed for the raw feature set (raw sensor/setting values
are defined from cycle 1) — confirmed 0 NaNs before and after, unlike the 109-feature
run where 29,253 NaNs needed filling.

**Training stopped early at epoch 36** (vs. epoch 24 for the 109-feature run) — it took
longer to converge with less redundant signal to lean on, but reached a meaningfully
better optimum:

| Model | Features | MAE | RMSE | R² | MAPE | Training Time |
|---|---:|---:|---:|---:|---:|---:|
| **CatBoost (Optuna-optimized, Sprint 11)** | 109 | **12.836** | **19.092** | **0.782** | — | 134.6s |
| LSTM (109 features, incl. lag/diff) | 109 | 18.345 | 26.382 | 0.599 | 40.53% | 269.6s |
| **LSTM (25 raw features only)** | 25 | **15.455** | **23.890** | **0.671** | **32.30%** | 383.9s |

**Removing the lag/diff features improved LSTM MAE by 15.8%** (18.345 → 15.455) while
using less than a quarter of the input dimensionality. This confirms the hypothesis
directly, not just plausibly: those engineered features were actively hurting the
LSTM, not merely unnecessary. The gap to CatBoost narrows substantially (from +5.51 MAE
down to +2.62 MAE) — the first LSTM result understated what a sequence model can do on
this task, because it was handed a feature set optimized for a different kind of model
entirely.

Full three-way comparison saved to `reports/three_way_model_comparison.csv`.

---

# What We Learned

**This result is expected, not a failure.** An unoptimized, single-layer LSTM starting
out behind a hyperparameter-tuned gradient-boosted model is the normal starting point
for this kind of comparison — CatBoost had two full sprints of feature selection and
tuning behind it; the LSTM had none. The point of this sprint was to get a fair,
honest first data point, not to win outright.

**The feature-set ablation was the single highest-value thing done in this sprint.**
Handing the LSTM the CatBoost-optimized 109-feature set — rather than the raw
25-feature set a sequence model actually needs — made the LSTM look far worse than it
actually is. A 15.8% MAE improvement from removing redundant, model-mismatched features
is a bigger lever than most hyperparameter tuning would have provided at this stage.
This is a durable lesson for the rest of the project, not a one-off fix: **feature
selection is model-specific, and a feature set frozen for one model family should not
be assumed to transfer to a structurally different one.**

**The overfitting pattern is informative for what comes next.** In both LSTM runs,
training MAE kept falling every epoch (109-feature run: 75.6→9.4; raw-feature run
tracked similarly) while validation MAE plateaued and drifted — textbook overfitting,
caught cleanly by early stopping both times. That points toward regularization (more
dropout, recurrent dropout, L2) and possibly a smaller window size or fewer units,
rather than a bigger network, as the more promising tuning direction for Sprint 13+.

**The NaN bug is a real, generalizable lesson for any future sequence model work in this
project** — but scoped correctly now: it only applies when lag/diff-style engineered
features are in play. The raw feature set needed zero NaN handling. GRU in Sprint 13,
using the raw feature set, inherits this simplification for free.

**Training time now genuinely trades off against accuracy, not just against CatBoost's
speed.** 383.9s (raw-feature LSTM, 36 epochs) vs. 269.6s (109-feature LSTM, 24 epochs)
vs. 134.6s (CatBoost) — the better LSTM result cost more compute, not less. Whether
that 2.9× cost over CatBoost is worth a 2.62 MAE gap is a real question for
Sprint 13+ to help answer, not a foregone conclusion either way.

---

# Decisions

- CatBoost (Sprint 11's Optuna-optimized version) remains the project's leading model.
  `artifacts/models/best_model.pkl` is **not** overwritten by this sprint's LSTM — it
  lost the comparison, even in its stronger (raw-feature) form.
- **The 25-feature raw set — not the 109-feature CatBoost-tuned set — is the correct
  input for Sprint 13's GRU baseline** and any further sequence-model work. Carrying
  the CatBoost feature set forward would keep comparing sequence models on an unfair,
  self-sabotaging footing.
- Both LSTM variants are saved for reference:
  - 109-feature run: `artifacts/models/lstm_baseline_full.keras`,
    `scalers/lstm_feature_scaler_full.pkl`, `lstm_training_state_full.json`
  - 25-feature (raw) run: `artifacts/models/lstm_baseline_raw.keras`,
    `scalers/lstm_feature_scaler_raw.pkl`, `lstm_training_state_raw.json`
- `SequenceGenerator`, `build_lstm_baseline`, and `DLTrainer` are reusable as-is for
  Sprint 13's GRU baseline — only the model-building function changes.
  `prepare_lstm_sequences.py` and `train_lstm.py` both already support
  `--feature-set {full,raw}` as a first-class option, not a one-off patch.
- Per the sprint brief: `test_FD004` / `RUL_FD004` remain untouched. Every metric in
  this document comes from the internal validation split, exactly like Sprints 8-11.
- Next: **Sprint 13 = GRU baseline**, starting from the raw 25-feature set directly
  (no repeated ablation needed — this sprint already settled that question), same
  split, same comparison protocol — then a final tuning and model-selection pass
  before the test set is finally used.
