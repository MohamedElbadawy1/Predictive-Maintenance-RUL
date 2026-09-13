# Predictive Maintenance — Remaining Useful Life (RUL) Prediction

Predicting how many operating cycles remain before a jet engine fails, from sensor
readings, using NASA's C-MAPSS FD004 dataset — the hardest of the four C-MAPSS
subsets (6 operating conditions, 2 fault modes simultaneously).

The goal: enable maintenance to happen right before it's needed, instead of on a
fixed schedule or after something breaks.

---

## Current Best Model

| | |
|---|---|
| **Model** | CatBoost, regime-aware normalization, Optuna-tuned |
| **Test MAE** | **19.133** (official, unseen `test_FD004` engines) |
| **Test RMSE** | 25.367 |
| **Test R²** | 0.784 |
| **Features** | 109 (of 151 engineered), frozen via experiment (Sprint 10) |
| **RUL cap** | 150 (found via experiment, Sprint 14) |
| **Normalization** | Per-operating-regime (6 regimes), not global (Sprint 17) |

Full history of how this model was arrived at — including the dead ends and negative
results — is in [`docs/`](#documentation-index) below. Every number in this project
comes from a real experiment; nothing here is assumed.

**Important**: this model requires an extra preprocessing step most of the project's
history didn't need — see [Regime-Aware Normalization](#regime-aware-normalization)
below before using `best_model.pkl` directly.

---

## Dataset

NASA C-MAPSS, subset **FD004**. Three files:

| File | What it is |
|---|---|
| `train_FD004` | Engines run to actual failure — full trajectories |
| `test_FD004` | Engines with trajectories cut off before failure — truncated |
| `RUL_FD004` | True remaining life at each test engine's cutoff point |

`test_FD004` / `RUL_FD004` are held out from training and hyperparameter tuning
entirely — used only for final, honest evaluation.

---

## Project Structure

```
data/raw/                      Raw NASA files (never modified)

src/
├── data/                      Loading + validation
├── preprocessing/             RUL generation, feature engineering, splitting,
│                               scaling, regime normalization, sequence generation
├── explainability/             Feature importance, category-based selection,
│                               importance-based reduction
├── models/                     Model factory, trainer (with built-in MLflow
│                               tracking), weighted ensemble
├── deep_learning/              LSTM/GRU sequence models
├── optimization/               Optuna hyperparameter tuning (CatBoost/XGBoost/
│                               LightGBM)
├── evaluation/                 Regression metrics (MAE, RMSE, R², MAPE)
├── experiments/                MLflow tracking setup
├── inference/                   Consolidated production inference pipeline
└── config/config.py            Single source of truth for every path and constant

notebooks/                     One notebook per pipeline stage, real outputs saved
docs/                          One doc per sprint — goal, method, real results
artifacts/                     Generated: models, scalers, processed data, MLflow store
reports/                       Generated: experiment result CSVs
```

Root-level pipeline scripts (each also has a matching notebook where noted):

| Script | What it does |
|---|---|
| `tune_and_ensemble.py` | Full pipeline: raw data → tuned CatBoost/XGBoost/LightGBM → ensemble |
| `tune_and_ensemble_regime_aware.py` | Same, with regime-aware normalization |
| `regime_normalization_experiment.py` | Controlled A/B: regime-aware vs. global scaling |
| `rul_cap_experiment.py` | Compares RUL cap strategies (no cap, 125, 150, 175, 195) |
| `train_lstm.py` / `train_gru.py` | Resumable sequence-model training with checkpointing |
| `promote_regime_aware_model.py` | Promotes a tuning result to canonical `best_model.pkl` |
| `bucket_error_analysis.py` | Breaks the canonical model's test error down by true RUL range |

---

## Quickstart

```bash
pip install -r requirements.txt

# Full pipeline from raw data: tuning + ensemble, all three traditional ML models
python tune_and_ensemble.py --n-trials 20

# Same, with regime-aware normalization (current best)
python tune_and_ensemble_regime_aware.py --n-trials 20
```

Both scripts are self-contained — they rebuild everything from `data/raw/`, they
don't assume any cached intermediate file already exists.

---

## Regime-Aware Normalization

FD004's 6 operating conditions mean the same sensor reads differently depending on
the engine's current condition, independent of degradation. Global scaling (used
through Sprint 16) conflates that with real degradation signal. `RegimeNormalizer`
detects the 6 regimes via K-Means on the operational settings, then normalizes each
sensor within its own regime rather than globally.

```python
from src.preprocessing.regime_normalizer import RegimeNormalizer

normalizer = RegimeNormalizer(n_regimes=6, sensor_columns=SENSOR_COLUMNS)
train_normalized = normalizer.fit(train_df).transform(train_df)   # fit on train only
test_normalized = normalizer.transform(test_df)                    # predict, never refit
```

This must run **before** feature engineering — rolling/lag/diff features are computed
on the normalized values, not raw ones. See `docs/Sprint_17_Regime_Aware_Normalization.md`
for the full experiment (controlled A/B test, then a full re-tuned pipeline) and an
honest discussion of where it helped and where it didn't (the ensemble, notably,
underperformed its own best member under this normalization).

---

## Experiment Tracking

Every training run — every hyperparameter trial included — is logged to MLflow
automatically via `BaseTrainer`, no separate logging call needed:

```python
trainer = BaseTrainer(model, run_name="my_run", tags={"feature_set": "109"})
trainer.train(X_train, y_train, X_val, y_val)  # trains AND logs params/metrics/model
```

To browse:

```bash
mlflow ui --backend-store-uri sqlite:///artifacts/mlruns/mlflow.db
```

Then open `http://localhost:5000`.

---

## Production Inference

`InferencePipeline` is the single place raw-sensor-data-to-prediction logic lives —
every analysis script's ad-hoc version of this chain has been consolidated into one
class:

```python
from src.inference.pipeline import InferencePipeline

pipeline = InferencePipeline()  # loads model + scaler + feature list once
predictions = pipeline.predict(raw_engine_readings)  # one row per engine
```

It auto-detects whether the current canonical model needs regime-aware normalization
(reading the flag `promote_regime_aware_model.py` writes) — callers don't need to
track which preprocessing path is currently active. See
`docs/Sprint_19_Inference_Pipeline.md` for the full input/output contract and how it
was verified.

This is Phase 1 of the path to production — see that same doc for what's next
(model packaging, a serving API, testing, containerization, monitoring).

---

## Training Pipeline & Model Registry

Training is callable, not just runnable as a script — so it can eventually be
triggered from an API endpoint, not just a terminal:

```python
from src.training.pipeline import TrainingPipeline

pipeline = TrainingPipeline()
result = pipeline.run(n_trials=20, model_names=["catboost", "xgboost", "lightgbm"])

if result.promoted:
    print(f"New champion: {result.winning_model_name}, MAE {result.champion_metric_after:.3f}")
```

"Best model" is defined by MLflow's **Model Registry** (alias-based `champion`, not
the deprecated stages API) — `TrainingPipeline` compares any new candidate's **test**
metric against the registry's current champion and only promotes (registers a new
version, moves the alias, updates the local files `InferencePipeline` reads) if it's
genuinely better. Nothing is overwritten blindly.

**First-time setup**: if you already have a real tuned model logged in MLflow from
before this registry existed, see `docs/Sprint_20_Training_Pipeline_MLflow_Registry.md`
for the exact steps to register it as the initial champion, so future training runs
compare against your real result instead of starting from nothing.

---

## `pipeline/` — Predict, Retrain, and Tune as Callable Entry Points

Three files, all MLflow-native (predict) or MLflow-aware (retrain), and the layer a
future FastAPI service would call into directly rather than reimplementing:

```bash
# Predict using the champion model loaded ENTIRELY from MLflow (model +
# scaler + regime normalizer + feature list — nothing from local files)
python pipeline/predict.py --input raw_engine_data.csv

# Retrain from raw data using the champion's already-known best hyperparameters
# (fast — no search, good for periodic retraining on fresh data)
python pipeline/train_with_best_params.py

# Retrain from raw data with a full hyperparameter search
python pipeline/train_with_tuning.py --n-trials 20
```

All three only promote a new model if it genuinely beats the current champion on
the official test set — see `docs/Sprint_21_Pipeline_Folder_MLflow_Native.md` for
the full design and real verification results.

A fourth and fifth script, `Pipeline/train_lstm.py` and `Pipeline/train_gru.py`,
train the LSTM and GRU baselines (`src/deep_learning/`) from raw data through to a
test-set evaluation, each logged to MLflow as a comparison run. Neither is part of
the champion-promotion flow above — see
`docs/Sprint_15_LSTM_Cap150_Test_Evaluation.md`'s reconstruction note for why and for
the current three-way LSTM/GRU/CatBoost result:

```bash
python Pipeline/train_lstm.py                # train + evaluate, resumes automatically
python Pipeline/train_lstm.py --no-resume     # restart training from epoch 0
python Pipeline/train_gru.py                  # same protocol, GRU instead of LSTM
```

---

## FastAPI Service (`api/`)

Routes separated into their own folder — `api/routes/predict.py` and
`api/routes/train.py` — with the app itself, schemas, and shared cached state each
in their own file:

```bash
pip install fastapi uvicorn
uvicorn api.main:app --reload
```

Then open `http://localhost:8000/docs` for interactive Swagger docs.

| Endpoint | What it does |
|---|---|
| `POST /predict/` | Predict RUL for one or more engines |
| `POST /predict/reload` | Force-reload the cached model after a new promotion |
| `POST /train/tuning` | Kick off a full hyperparameter search (background job) |
| `POST /train/best-params` | Retrain with the champion's known params, no search |
| `GET /train/status/{job_id}` | Poll a training job |
| `GET /health` / `GET /health/ready` | Liveness / readiness checks |

Training runs as a background job (returns a `job_id` immediately, poll for the
result) since even a fixed-params retrain takes real time. See
`docs/Sprint_22_FastAPI_Service.md` for the full design and real end-to-end
verification (including a real prediction request matching the known reference
value exactly, and a full training job lifecycle watched start to finish).

---

## Documentation Index

Chronological, one file per sprint. Numbering note: Sprints 1–9 (data understanding
through baseline modeling) and this project's Sprint 10 ("Model Explainability")
predate the numbered-underscore docs below — both exist in `docs/`, distinguished by
filename style.

| Sprint | Topic |
|---|---|
| 01–09 | Project setup through traditional ML baselines *(see `docs/Sprint 0N — ...md`)* |
| 10 (space-separated) | Model Explainability |
| [10](docs/Sprint_10_Feature_Selection.md) | Feature Selection — 151 → 109 features, 8 real experiments |
| [11](docs/Sprint_11_Hyperparameter_Optimization.md) | Hyperparameter Optimization (Optuna) |
| [12](docs/Sprint_12_LSTM_Baseline.md) | LSTM Baseline + the raw-features discovery |
| [13](docs/Sprint_13_Final_Test_Evaluation.md) | First official test-set evaluation |
| [14](docs/Sprint_14_RUL_Cap_Investigation.md) | RUL cap investigation — 125 → 150 |
| [15](docs/Sprint_15_LSTM_Cap150_Test_Evaluation.md) | Fair CatBoost vs. LSTM comparison |
| [16](docs/Sprint_16_GRU_Baseline_MLflow.md) | GRU baseline + MLflow adoption |
| [17](docs/Sprint_17_Regime_Aware_Normalization.md) | Regime-aware normalization (current model) |
| [18](docs/Sprint_18_Bucket_Error_Analysis.md) | Bucket-level error analysis of the canonical model |
| [19](docs/Sprint_19_Inference_Pipeline.md) | Consolidated inference pipeline (Production Phase 1) |
| [20](docs/Sprint_20_Training_Pipeline_MLflow_Registry.md) | Training pipeline + MLflow Model Registry |
| [21](docs/Sprint_21_Pipeline_Folder_MLflow_Native.md) | `pipeline/` folder — MLflow-native predict, train, tune |
| [22](docs/Sprint_22_FastAPI_Service.md) | FastAPI service — separated routes, background training jobs |

For a runnable, narrated tour of the whole project, see
`notebooks/00_project_walkthrough.ipynb`.

---

## Requirements

```
pandas, numpy, scikit-learn, xgboost, lightgbm, catboost, optuna, tensorflow-cpu, mlflow, fastapi, uvicorn
```

See `requirements.txt` for the full pinned list.
