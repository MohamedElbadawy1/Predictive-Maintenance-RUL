# Sprint 21 — `pipeline/` Folder: MLflow-Native Predict, Train, and Tune

---

# Goal

Three callable entry points, all building on Sprint 20's `TrainingPipeline` and
Model Registry work:

1. **Predict using the model as it exists in MLflow** — not local files.
2. **Retrain from raw data using the champion's already-known best hyperparameters**
   — no search, for fast periodic retraining on fresh data.
3. **Retrain from raw data with a full hyperparameter search** — for when the
   configuration itself might need to change, not just be refreshed on new data.

Also answers a real sequencing question: **build these before FastAPI, not after.**
An API should be a thin layer that parses a request and calls real logic — building
it first would mean cramming business logic into route handlers, the same mistake
Phase 1 (`InferencePipeline`) already fixed for prediction. These three files are
what a future FastAPI layer will call into.

---

# What Was Built

```
pipeline/
├── predict.py                    # File 1
├── train_with_best_params.py     # File 2
└── train_with_tuning.py          # File 3

src/training/
├── data_prep.py                  # shared raw-data-to-model-ready-features logic
└── pipeline.py                   # TrainingPipeline — extended this sprint
```

**`src/training/data_prep.py`** — extracted the raw-data-to-model-ready-features
chain (ingest -> validate -> RUL -> split -> regime-normalize -> feature engineer ->
select -> scale) into one function, `prepare_training_data()`. It was about to be
duplicated a third time (once in `TrainingPipeline`, and now again in two new
training scripts) — exactly the scattered-logic problem Phase 1 fixed for
prediction. `TrainingPipeline` itself was refactored to use this shared function
too, removing its own duplicate copy.

**`src/training/pipeline.py`** gained `run_fixed_params(model_name, params)` —
same pipeline as `run()`, but trains one model with a given configuration instead of
searching. Also: every promotion now logs the preprocessing artifacts (scaler,
regime normalizer, feature list) onto the MLflow run itself
(`mlflow.log_artifact(..., artifact_path="preprocessing")`), not just the model.
This is what makes File 1 possible — without it, "predict from MLflow" would still
secretly depend on local files for anything except the model itself.

**`src/experiments/mlflow_tracker.py`** gained `get_champion_run_info()` — returns
the champion's run ID, model family, and logged hyperparameters in one call. This is
what File 2 uses to know both *which* model to retrain and *what* configuration to
use.

## File 1 — `pipeline/predict.py`

```python
from pipeline.predict import MLflowInferencePipeline

pipeline = MLflowInferencePipeline()
predictions = pipeline.predict(raw_engine_readings)
```

Loads the model via the registry's `champion` alias, and the regime normalizer /
scaler / feature list from that same run's `preprocessing/` artifacts — everything
sourced from MLflow, nothing from local files. A different machine with access to
the same tracking store could run this with zero local setup beyond the code itself.

## File 2 — `pipeline/train_with_best_params.py`

Fetches the champion's `model_family` and hyperparameters from MLflow, casts the
params back from MLflow's string-only storage to their real types (int/float/bool,
via a small type-sniffing helper — MLflow has no native typed-param storage), and
retrains on freshly-ingested raw data. Promotes only if the retrained model still
beats the current champion on the test set.

## File 3 — `pipeline/train_with_tuning.py`

Thin CLI wrapper around `TrainingPipeline.run()` — the real logic already existed
from Sprint 20; this just exposes it as a standalone, argument-parsed script.

---

# Real, End-to-End Verification

Every piece was tested against real data and a real (if lightly-tuned, for speed)
MLflow registry state — not just written and assumed to work:

1. **Artifact logging confirmed**: after a real promotion, checked the run directly
   — `preprocessing/feature_scaler.pkl`, `preprocessing/regime_normalizer.pkl`, and
   `preprocessing/selected_features.json` were all actually present, plus a
   `requires_regime_normalizer` tag.
2. **File 1 exact-match test**: ran `MLflowInferencePipeline` against the real
   official `test_FD004`, computed MAE against `RUL_FD004` — **19.3066046178343**,
   an exact match (to 10 decimal places) against both `InferencePipeline` (Phase 1,
   local-file-based) and `TrainingPipeline`'s own reported metric. Three independent
   code paths, identical answer.
3. **File 2 correctness test**: fetched real champion params from MLflow
   (`{'iterations': '982', 'learning_rate': '0.190...', 'depth': '4', ...}` — all
   strings, as MLflow stores them), correctly cast to native types, retrained
   deterministically on the same data -> identical MAE to the champion -> correctly
   **declined** to promote a tie. Proves both the type-casting and the
   promotion-guard work together correctly, not just in isolation.
4. **File 3 CLI test**: ran the actual command-line script (not just the underlying
   class), confirmed it reaches the same result through the full argument-parsing
   path.

A found-and-fixed bug during this work, stated plainly: an earlier attempt to reset
the test registry for a clean bootstrap test silently failed because the cleanup
script never called `MLflowTracker()` first to set the correct tracking URI — it was
deleting from the wrong (default) store while the real registry stayed untouched.
Caught by checking the registry's actual state directly rather than trusting the
cleanup script's silence.

---

# Decisions

- **Registry model name confirmed as `predictive-maintenance-rul-model`** (matching
  what was already manually registered), not the earlier draft name
  `predictive-maintenance-rul` used in Sprint 20's initial code — fixed across all
  three files and `TrainingPipeline`'s default.
- File 1 is MLflow-only by design — it does not fall back to local files if
  artifacts are missing from the run, since that would silently defeat the point.
  A champion registered before this sprint's artifact-logging existed (e.g. the
  original manually-bootstrapped Version 1) will not have `preprocessing/`
  artifacts attached — re-run a promotion (either training script) to fix this for
  that model, or manually attach them following the same `mlflow.log_artifact`
  pattern shown in `TrainingPipeline._promote_if_better`.
- Shared `prepare_training_data()` is the one place this logic should be extended
  going forward — do not add a fourth copy in a future script.

---

# What's Next

With all three of predict/retrain-fixed/retrain-with-search callable as real
functions (not scripts with embedded logic), FastAPI is now a reasonable next step:
thin endpoints that call `MLflowInferencePipeline().predict(...)` synchronously, and
`TrainingPipeline().run(...)` / `.run_fixed_params(...)` as background tasks (both
take real time — minutes, not milliseconds).
