# Predictive Maintenance — Remaining Useful Life (RUL) Prediction

Predicts how many operating cycles a jet engine has left before it fails, from its
sensor readings — so maintenance can happen right before it's needed, instead of on
a fixed schedule or after something breaks. Built on NASA's C-MAPSS **FD004**, the
hardest of the four subsets (6 operating conditions, 2 fault modes at once).

The served model is an **LSTM + GRU ensemble** that reaches a test MAE of **16.39**
cycles on NASA's held-out test engines, running on ONNX Runtime.

## 🔗 Live Demo

**https://predictive-maintenance-rul-1.streamlit.app/**

Click **Use sample data** to run the model on 3 real NASA test engines, or upload
your own CSV ([format below](#try-the-app)).

---

## Results

Every model was evaluated on the same official, held-out `test_FD004` engines:

| Model | Test MAE | Test R² |
|---|---:|---:|
| CatBoost | 18.72–19.13 | 0.78 |
| LSTM (seed=42) | 17.69 | 0.787 |
| GRU (seed=42) | 16.74 | 0.819 |
| **LSTM + GRU average (served)** | **16.39** | **0.822** |

Single-seed results carry roughly ±1 MAE point of run-to-run variance — see
[`docs/Sprint_15_LSTM_Cap150_Test_Evaluation.md`](docs/Sprint_15_LSTM_Cap150_Test_Evaluation.md)
for the full comparison.

## What was built

- **Data pipeline** — loading and validation, RUL target generation, feature
  engineering, and regime-aware normalization.
- **Models** — Optuna-tuned CatBoost / XGBoost / LightGBM, plus LSTM and GRU
  sequence models. Every training run is logged to MLflow.
- **Serving** — the LSTM+GRU average is exported to ONNX and run with ONNX Runtime
  (no TensorFlow needed at serving time), behind a FastAPI service and a Streamlit
  app.
- **Hosting** — the live demo runs on Streamlit Community Cloud from
  `deploy/streamlit-cloud/app.py`, which calls the ensemble directly in-process
  (a single Streamlit process, no separate API). The exported files in
  `artifacts/models/ensemble/` are committed (force-added despite `.gitignore`) so
  the hosted app can load them.

## Try the App

**Use sample data** loads `deploy/streamlit-cloud/sample_data.csv` — 551 real rows
from three held-out test engines (units 1, 5 and 12; 230, 51 and 270 cycles of
history). Then **Predict RUL** returns one prediction per engine.

**Your own CSV** needs one row per (engine, cycle) with 26 columns:

```
unit_number, time_in_cycles, operational_setting_1, operational_setting_2,
operational_setting_3, sensor_1, sensor_2, ..., sensor_21
```

Send each engine's **full history so far**, with at least 30 rows — the model reads
a 30-cycle window. Engines with fewer come back with `short_history_warning: true`
and no prediction, rather than a guess.

---

## Run Locally

This project uses [`uv`](https://docs.astral.sh/uv/) (a `requirements.txt` is also
kept for plain pip):

```bash
uv sync

# Train + export the LSTM+GRU ensemble (writes artifacts/models/ensemble/)
uv run python Pipeline/train_ensemble.py

# The Streamlit app (calls the ensemble directly)
uv run streamlit run deploy/streamlit-cloud/app.py

# Or the API, with frontend/app.py as a client for it
uv run uvicorn api.api_main:app --reload
uv run streamlit run frontend/app.py
```

Browse MLflow runs with
`mlflow ui --backend-store-uri sqlite:///artifacts/mlruns/mlflow.db`.

Or run the API and frontend in containers:

```bash
docker compose up --build   # API: localhost:8000/docs · Frontend: localhost:8501
```

Both containers mount `./artifacts`, so train first — the API reads the exported
ensemble from there.

### `Pipeline/` scripts

Thin CLI wrappers — the actual logic lives in `src/`:

| Script | What it does |
|---|---|
| `train_ensemble.py` | Trains (or reuses, with `--skip-training`) the LSTM and GRU, checks their average beats each alone, exports both to ONNX, and writes everything the ensemble needs to `artifacts/models/ensemble/` |
| `train_lstm.py` / `train_gru.py` | Resumable sequence-model training with checkpointing, seed control, and an official test-set evaluation |
| `ensemble_evaluate.py` | Blends CatBoost + LSTM + GRU predictions on the same test engines and reports every combination |
| `train_with_tuning.py` | Raw data → Optuna-tuned CatBoost/XGBoost/LightGBM → promotes the best to MLflow's `champion` alias if it beats the current one |
| `train_with_best_params.py` | Retrains with the champion's known hyperparameters (no search) |
| `predict.py` | CLI: predict on a CSV of raw readings with the MLflow champion (CatBoost) |

## API

`POST /predict/` serves the LSTM+GRU ensemble by default. Set
`MODEL_BACKEND=catboost` before starting to serve the MLflow champion instead.

| Endpoint | What it does |
|---|---|
| `GET /health` | Liveness check |
| `GET /health/ready` | Whether the model has been loaded yet |
| `POST /predict/` | Predict RUL for one or more engines |
| `POST /predict/reload` | Force-reload the cached model |
| `POST /train/tuning` | Full hyperparameter search (background job) |
| `POST /train/best-params` | Retrain with the champion's known params (background job) |
| `GET /train/status/{job_id}` | Poll a training job |

Interactive docs are at `http://localhost:8000/docs`. Training endpoints return a
`job_id` immediately since a retrain takes real time.

## Regime-Aware Normalization

FD004's 6 operating conditions make the same sensor read differently depending on
the engine's current condition, independent of degradation. `RegimeNormalizer`
detects the regimes with K-Means on the operational settings, then normalizes each
sensor within its own regime instead of globally. It is fit on training data only,
and both serving paths apply it before scoring.

---

## Project Structure

```
data/raw/            Raw NASA files (never modified)
src/
├── data/            Loading + validation
├── preprocessing/   RUL generation, feature engineering, splitting, scaling,
│                    regime normalization, sequence generation
├── explainability/  Feature importance and selection
├── models/          Model factory, trainer (with MLflow tracking), weighted ensemble
├── deep_learning/   LSTM/GRU models, sequence data prep
├── optimization/    Optuna tuning (CatBoost/XGBoost/LightGBM)
├── evaluation/      Regression metrics (MAE, RMSE, R², MAPE)
├── experiments/     MLflow tracking setup
├── training/        TrainingPipeline — tuning, promotion, registry
├── pipelines/       Inference: EnsembleInferencePipeline (ONNX) and
│                    MLflowInferencePipeline (CatBoost champion)
└── config/          Every path and constant
Pipeline/            CLI entry points (above)
api/                 FastAPI service
frontend/            Streamlit client for the API
deploy/streamlit-cloud/   The hosted app + sample data
notebooks/           One notebook per pipeline stage, outputs saved
docs/                One doc per sprint — goal, method, real results
tests/               Unit tests (trainer, splitting, features, scaling, metrics,
                     deep-learning prep, ensemble inference)
```

CI (`.github/workflows/ci.yml`) runs the tests on every push and pull request to
`main`, then builds the Docker images.

## Documentation

One file per sprint in `docs/`. For a narrated tour, see
`notebooks/00_project_walkthrough.ipynb`.

| Sprint | Topic |
|---|---|
| 01–10 | Setup, data understanding, ingestion, EDA, targets, features, preparation, ML baselines, explainability (`docs/Sprint 0N — ...md`) |
| [10](docs/Sprint_10_Feature_Selection.md) | Feature selection — 151 → 109 features |
| [11](docs/Sprint_11_Hyperparameter_Optimization.md) | Hyperparameter optimization (Optuna) |
| [12](docs/Sprint_12_LSTM_Baseline.md) | LSTM baseline |
| [13](docs/Sprint_13_Final_Test_Evaluation.md) | First official test-set evaluation |
| [14](docs/Sprint_14_RUL_Cap_Investigation.md) | RUL cap investigation |
| [15](docs/Sprint_15_LSTM_Cap150_Test_Evaluation.md) | CatBoost vs. LSTM vs. GRU, and the LSTM+GRU ensemble |
| [18](docs/Sprint_18_Bucket_Error_Analysis.md) | Bucket-level error analysis |
| [19](docs/Sprint_19_Inference_Pipeline.md) | Inference pipeline |
| [20](docs/Sprint_20_Training_Pipeline_MLflow_Registry.md) | Training pipeline + MLflow Model Registry |
| [21](docs/Sprint_21_Pipeline_Folder_MLflow_Native.md) | `Pipeline/` folder |
| [22](docs/Sprint_22_FastAPI_Service.md) | FastAPI service |

## Requirements

`pandas, numpy, scikit-learn, xgboost, lightgbm, catboost, optuna, tensorflow-cpu,
tf2onnx, onnx, onnxruntime, mlflow, fastapi, uvicorn, streamlit, requests, joblib,
matplotlib, shap` — pinned in `pyproject.toml` + `uv.lock`.

The hosted app installs far less (`deploy/streamlit-cloud/requirements.txt`: streamlit,
pandas, numpy, scikit-learn, joblib, onnxruntime).