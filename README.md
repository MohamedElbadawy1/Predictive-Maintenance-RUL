# Predictive Maintenance — Remaining Useful Life (RUL) Prediction

Predicting how many operating cycles remain before a jet engine fails, from sensor
readings, using NASA's C-MAPSS FD004 dataset — the hardest of the four C-MAPSS
subsets (6 operating conditions, 2 fault modes simultaneously).

The goal: enable maintenance to happen right before it's needed, instead of on a
fixed schedule or after something breaks.

---

## Current Best Model

Three model families were trained and evaluated on the same official, held-out
`test_FD004` engines. A simple average of the two sequence models beats every
single model individually — not by a trivial margin:

| Model | Test MAE | Test R² |
|---|---:|---:|
| CatBoost (MLflow "champion" — the one actually served) | 18.72–19.13 | 0.78 |
| LSTM (seed=42) | 17.69 | 0.787 |
| GRU (seed=42) | 16.74 | 0.819 |
| **LSTM + GRU average (best result, not served)** | **16.39** | **0.822** |

The LSTM+GRU average is the strongest result in this project, but it is **not**
what the API actually serves — CatBoost remains the registered MLflow "champion"
because promoting a sequence-model average would need its own sequence-aware
serving path (`MLflowInferencePipeline` currently expects one tabular row per
engine, not a 30-cycle window), which hasn't been built yet. See
[`docs/Sprint_15_LSTM_Cap150_Test_Evaluation.md`](docs/Sprint_15_LSTM_Cap150_Test_Evaluation.md)
for the full comparison, the reconstruction history behind it, and why single-seed
results here should be read with real caution (documented seed-to-seed variance of
roughly ±1 MAE point).

Full history of how the CatBoost model was arrived at — including the dead ends and
negative results — is in [`docs/`](#documentation-index) below. Every number in this
project comes from a real experiment; nothing here is assumed.

**Important**: the served model requires an extra preprocessing step most of the
project's history didn't need — see [Regime-Aware Normalization](#regime-aware-normalization)
below.

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
├── deep_learning/              LSTM/GRU models, shared sequence data prep
├── optimization/               Optuna hyperparameter tuning (CatBoost/XGBoost/
│                               LightGBM)
├── evaluation/                 Regression metrics (MAE, RMSE, R², MAPE)
├── experiments/                MLflow tracking setup
├── training/                   TrainingPipeline — tuning, promotion, registry
├── pipelines/                  MLflowInferencePipeline — loads the champion +
│                               its preprocessing artifacts entirely from MLflow
└── config/config.py            Single source of truth for every path and constant

Pipeline/                       Thin CLI entry points — see the table below
api/                            FastAPI service (see FastAPI Service, below)
frontend/                       Streamlit UI that calls the API
notebooks/                     One notebook per pipeline stage, real outputs saved
docs/                          One doc per sprint — goal, method, real results
tests/                         unittest suite — models, preprocessing, deep
                               learning, API, frontend client
artifacts/                     Generated: models, scalers, processed data, MLflow store
reports/                       Generated: experiment result CSVs
Dockerfile, docker-compose.yml  Containerized api + frontend services
.github/workflows/ci.yml        Runs the test suite, then builds both images
```

`Pipeline/` scripts (thin wrappers — the actual logic lives in `src/`):

| Script | What it does |
|---|---|
| `train_with_tuning.py` | Full pipeline: raw data → Optuna-tuned CatBoost/XGBoost/LightGBM → promotes the best if it beats the current champion |
| `train_with_best_params.py` | Retrains with the champion's already-known best hyperparameters (no search) |
| `predict.py` | CLI: load the champion from MLflow, predict on a CSV of raw engine readings |
| `train_lstm.py` / `train_gru.py` | Resumable sequence-model training with checkpointing, seed control, and an official test-set evaluation |
| `ensemble_evaluate.py` | Blends CatBoost + LSTM + GRU predictions on the same test engines and reports every combination |

---

## Quickstart

This project uses [`uv`](https://docs.astral.sh/uv/) (a `requirements.txt` is also
kept in sync for a plain-pip workflow):

```bash
uv sync

# Full pipeline from raw data: Optuna tuning across all three traditional ML
# models, promotes the winner as champion if it beats the current one
uv run python Pipeline/train_with_tuning.py --n-trials 20

# LSTM / GRU baselines (same official test-set protocol as CatBoost)
uv run python Pipeline/train_lstm.py
uv run python Pipeline/train_gru.py

# Compare all three models and their blends on the same test engines
uv run python Pipeline/ensemble_evaluate.py
```

Or run everything containerized — see [Docker](#docker) below.

Every script above is self-contained — it rebuilds what it needs from `data/raw/`,
it doesn't assume any cached intermediate file already exists.

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
on the normalized values, not raw ones. This is a case where the doc referenced in
early sprint planning (`Sprint_17_Regime_Aware_Normalization.md`) was never actually
written — see the [Documentation Index](#documentation-index)'s note on Sprints 16–17
for what happened instead. The honest result: it helped the individual CatBoost,
LSTM, and GRU models (see [Current Best Model](#current-best-model)), but the
CatBoost+LSTM+GRU ensemble underperformed its own best member under it.

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

`/predict` is served by one of two independent backends, chosen once at process
start by the `MODEL_BACKEND` environment variable (default **`ensemble`**):

| `MODEL_BACKEND` | Class | Reads from | Test MAE |
|---|---|---|---|
| `ensemble` (default) | `EnsembleInferencePipeline` | local files under `artifacts/models/ensemble/` (written by `Pipeline/train_ensemble.py`) | **16.69** |
| `catboost` | `MLflowInferencePipeline` | MLflow's `champion`-aliased registry model | 18.72–19.13 |

```python
from src.pipelines.ensemble_inference_pipeline import EnsembleInferencePipeline

pipeline = EnsembleInferencePipeline()  # loads LSTM + GRU + scaler + regime normalizer once
predictions = pipeline.predict(raw_engine_readings)  # one row per engine, averaged
```

Both classes take the same input shape (raw per-cycle readings, full history per
engine — not just the latest row) and return the same output shape (one row per
engine, `predicted_RUL` + metadata), so `api/dependencies.py`'s `PipelineCache`
can load either one without the route code caring which. `MLflowInferencePipeline`
auto-detects whether the current champion needs regime-aware normalization by
reading the `requires_regime_normalizer` tag `TrainingPipeline` sets on every
promoted run; `EnsembleInferencePipeline` reads the same information from its own
`manifest.json` instead, since Keras models here aren't registered in MLflow's
Model Registry at all (`train_lstm.py` / `train_gru.py` log them as comparison
runs only) — see that class's module docstring for why local files are the right
choice for this specific path, and why its preprocessing deliberately mirrors
`src/deep_learning/data_prep.py` step for step.

This class is what both `Pipeline/predict.py` (CLI) and the FastAPI service
(`api/dependencies.py`) call — it used to live inside `Pipeline/predict.py` itself,
which caused a real bug: importing it cross-folder from `api/` broke depending on
the checked-out folder's exact case (fine on some machines, silently broken on
others, and reliably broken inside the Linux-based Docker image). Moving it to
`src/` fixed that for good — see
[`docs/Sprint_19_Inference_Pipeline.md`](docs/Sprint_19_Inference_Pipeline.md) for
the original input/output contract.

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
metric (logged under the `test_MAE` key specifically — a run's bare `MAE` metric is
its *validation* score, logged earlier by `BaseTrainer.train()`; conflating the two
caused a real, documented false alarm during this project's own development, see
[`docs/Sprint_15_LSTM_Cap150_Test_Evaluation.md`](docs/Sprint_15_LSTM_Cap150_Test_Evaluation.md))
against the registry's current champion and only promotes if it's genuinely better.
Nothing is overwritten blindly.

**First-time setup**: if you already have a real tuned model logged in MLflow from
before this registry existed, see `docs/Sprint_20_Training_Pipeline_MLflow_Registry.md`
for the exact steps to register it as the initial champion, so future training runs
compare against your real result instead of starting from nothing.

**Rolling back to a previous champion** — no retraining required — is documented in
[`docs/Rollback_Strategy.md`](docs/Rollback_Strategy.md).

---

## `Pipeline/` — Predict, Retrain, and Tune as Callable Entry Points

Thin CLI wrappers — the actual logic lives in `src/` (see
[Project Structure](#project-structure)) — and the layer the FastAPI service calls
into directly rather than reimplementing:

```bash
# Predict using the champion model loaded ENTIRELY from MLflow (model +
# scaler + regime normalizer + feature list — nothing from local files)
python Pipeline/predict.py --input raw_engine_data.csv

# Retrain from raw data using the champion's already-known best hyperparameters
# (fast — no search, good for periodic retraining on fresh data)
python Pipeline/train_with_best_params.py

# Retrain from raw data with a full hyperparameter search
python Pipeline/train_with_tuning.py --n-trials 20
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

A sixth script, `Pipeline/train_ensemble.py`, is what actually promotes an
LSTM+GRU pair into production: it trains (or reuses, with `--skip-training`) both
baselines for one seed, confirms their equal-weight average beats each of them
individually on the test set, and writes everything `EnsembleInferencePipeline`
needs to serve that average — manifest, feature scaler, regime normalizer — to
`artifacts/models/ensemble/`:

```bash
python Pipeline/train_ensemble.py                    # train both from scratch, then build
python Pipeline/train_ensemble.py --skip-training    # reuse already-trained *_final.keras checkpoints
python Pipeline/train_ensemble.py --seed 7            # compare a different seed before deploying
```

Unlike the CatBoost path, nothing here touches MLflow's `champion` alias — the two
serving backends are fully independent (see
[Production Inference](#production-inference)).

A sixth script, `Pipeline/ensemble_evaluate.py`, blends CatBoost + LSTM + GRU
predictions on the same test engines and reports every combination — this is where
the LSTM+GRU result in [Current Best Model](#current-best-model) comes from:

```bash
python Pipeline/ensemble_evaluate.py
```

---

## FastAPI Service (`api/`)

Routes separated into their own folder — `api/routes/predict.py` and
`api/routes/train.py` — with the app itself, schemas, and shared cached state each
in their own file:

```bash
uv run uvicorn api.api_main:app --reload
```

Then open `http://localhost:8000/docs` for interactive Swagger docs.

| Endpoint | What it does |
|---|---|
| `GET /health` | Liveness check — always `{"status": "ok"}` if the process is up |
| `GET /health/ready` | Readiness — whether the champion model has been loaded yet |
| `POST /predict/` | Predict RUL for one or more engines |
| `POST /predict/reload` | Force-reload the cached model after a new promotion |
| `POST /train/tuning` | Kick off a full hyperparameter search (background job) |
| `POST /train/best-params` | Retrain with the champion's known params, no search |
| `GET /train/status/{job_id}` | Poll a training job |

**Auth**: `/predict/*` and `/train/*` are open by default — fine for local use.
Setting an `API_KEY` environment variable before starting the service requires every
request to those routes to send a matching `X-API-Key` header (`/health` stays open
either way, for orchestration/monitoring tools that shouldn't need a key):

```bash
API_KEY=your-secret-here uv run uvicorn api.api_main:app
```

```bash
curl -H "X-API-Key: your-secret-here" -X POST http://localhost:8000/predict/ -d '...'
```

---

## Frontend (`frontend/`)

A minimal Streamlit UI that calls the API above — upload a CSV of raw engine
readings, get each engine's predicted RUL as a table and a chart. It never loads a
model itself; it's a thin HTTP client (`frontend/api_client.py`) in front of the API:

```bash
# in a separate terminal, with the API already running
uv run streamlit run frontend/app.py
```

Then open `http://localhost:8501`. The sidebar takes the API's base URL and, if
you've set one, its `API_KEY` — both also read from environment variables of the
same name, so they're filled in automatically under Docker Compose.

---

## Docker

Both services share one image (`Dockerfile`), built with `uv` from `pyproject.toml`
+ `uv.lock` (this project's real dependency source of truth — `requirements.txt` is
kept in sync for a plain-pip workflow, but Docker uses the lockfile for fully
reproducible installs):

```bash
docker compose up --build
```

- API: `http://localhost:8000/docs`
- Frontend: `http://localhost:8501`

First run has no champion yet (`artifacts/` starts empty) — train one from inside
the container before `/predict` will work:

```bash
docker compose exec api python Pipeline/train_with_tuning.py --n-trials 20
```

To require an API key, set `API_KEY` before starting (`export API_KEY=...` or a
`.env` file next to `docker-compose.yml`) — both services pick it up automatically.

`artifacts/` and `data/` are mounted as volumes, not baked into the image: a
champion trained inside the container is visible on the host and survives
`docker compose down`, and raw CMAPSS data never needs to be copied into the image
at all.

---

## Deploying to Hugging Face Spaces (free tier)

`deploy/huggingface/` holds a self-contained variant of the setup above for a free
[HF Space](https://huggingface.co/spaces) (`CPU basic`, Docker SDK): one container
running both services, since a Space exposes exactly one public port, with the
LSTM+GRU ensemble baked into the image at build time, since a free Space's storage
is **ephemeral** — nothing written after the container starts (a training run,
`docker compose`'s bind-mounted `artifacts/`) survives a restart or rebuild.

**1. Train and commit the artifacts this serves** (once — or again whenever you
want to redeploy on a fresher model):

```bash
python Pipeline/train_ensemble.py        # writes artifacts/models/*.keras and
                                          # artifacts/models/ensemble/
git add -f artifacts/models/*.keras artifacts/models/ensemble/
git commit -m "Add trained LSTM+GRU ensemble for deployment"
```

(`artifacts/` is in `.gitignore` for local dev — `-f` is deliberate here, since
this is the one case where the trained files *are* the deployment.)

**2. Create the Space**: [huggingface.co/new-space](https://huggingface.co/new-space) →
pick a name → SDK: **Docker** → create.

**3. Push this repo's code to the Space**, with the Space's own Dockerfile and
README at its root instead of this repo's (a Space's Docker SDK only looks for
`Dockerfile` / `README.md` at the repo root, not in a subfolder):

```bash
git clone https://huggingface.co/spaces/<your-username>/<your-space-name> hf-space
cd hf-space
cp -r ../Predictive-Maintenance-RUL/{api,src,frontend} .
cp ../Predictive-Maintenance-RUL/{pyproject.toml,uv.lock} .
cp -r ../Predictive-Maintenance-RUL/artifacts .
cp ../Predictive-Maintenance-RUL/deploy/huggingface/Dockerfile .
cp ../Predictive-Maintenance-RUL/deploy/huggingface/start.sh .
cp ../Predictive-Maintenance-RUL/deploy/huggingface/README.md .
git add -A && git commit -m "Deploy predictive maintenance RUL app" && git push
```

The Space rebuilds automatically on push — build logs are on the Space's page. Once
it's live, `MODEL_BACKEND=ensemble` (set in `deploy/huggingface/Dockerfile`) means
`/predict` is served entirely from the two committed `.keras` files and the
`artifacts/models/ensemble/` folder — no MLflow, no database, nothing else to
provision.

**Optional but recommended**: since anyone can open a public Space, set `API_KEY`
under the Space's **Settings → Repository secrets** — both services already read
it automatically (same mechanism as the Docker Compose setup above), and it's
worth doing before sharing the link since `/train/*` can trigger a real (if slow,
on free CPU hardware) training job.

---

## CI (`.github/workflows/ci.yml`)

Runs on every push and pull request to `main`: installs dependencies from
`uv.lock`, runs the full test suite (models, preprocessing, deep learning, the API,
and the frontend's API client), and — only if tests pass — builds both Docker
images to confirm they still build cleanly.

Training runs as a background job (`POST /train/*` returns a `job_id` immediately,
`GET /train/status/{job_id}` polls for the result) since even a fixed-params retrain
takes real time. See `docs/Sprint_22_FastAPI_Service.md` for the full design and
real end-to-end verification (including a real prediction request matching the
known reference value exactly, and a full training job lifecycle watched start to
finish).

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
| [15](docs/Sprint_15_LSTM_Cap150_Test_Evaluation.md) | Fair CatBoost vs. LSTM comparison, later reconstructed after a sandbox reset — see that doc's "Corrected note" and "Final result" sections for the full story, the GRU baseline, and the LSTM+GRU ensemble result |
| 16, 17 *(referenced in early sprint docs, never actually written)* | A GRU baseline and regime-aware normalization were both planned here — regime-aware normalization was built (`src/preprocessing/regime_normalizer.py`) and GRU was eventually added, but as part of the Sprint 15 reconstruction above, not as standalone docs. Nothing to link — noted here so the gap doesn't look accidental |
| [18](docs/Sprint_18_Bucket_Error_Analysis.md) | Bucket-level error analysis of the canonical model |
| [19](docs/Sprint_19_Inference_Pipeline.md) | Consolidated inference pipeline (Production Phase 1) |
| [20](docs/Sprint_20_Training_Pipeline_MLflow_Registry.md) | Training pipeline + MLflow Model Registry |
| [21](docs/Sprint_21_Pipeline_Folder_MLflow_Native.md) | `Pipeline/` folder — MLflow-native predict, train, tune |
| [22](docs/Sprint_22_FastAPI_Service.md) | FastAPI service — separated routes, background training jobs |
| [Rollback Strategy](docs/Rollback_Strategy.md) | How to roll back a bad model (MLflow alias, no retraining) or a bad code change (git + CI), independently of each other |

For a runnable, narrated tour of the whole project, see
`notebooks/00_project_walkthrough.ipynb`.

---

## Requirements

```
pandas, numpy, scikit-learn, xgboost, lightgbm, catboost, optuna, tensorflow-cpu,
mlflow, fastapi, uvicorn, streamlit, requests, joblib, matplotlib, shap
```

`pyproject.toml` + `uv.lock` are the source of truth (`uv sync` installs the exact
pinned versions in the lockfile); `requirements.txt` lists the same packages for a
plain `pip install -r requirements.txt` workflow, kept in sync by hand.
