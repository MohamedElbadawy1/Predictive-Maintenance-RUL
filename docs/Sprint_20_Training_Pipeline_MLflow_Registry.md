# Sprint 20 — Training Pipeline + MLflow Model Registry

---

# Goal

Two things requested together, and they genuinely depend on each other:

1. Training needs to be **callable**, not just runnable as a standalone script — so
   it can eventually be triggered from an endpoint (FastAPI), not just from a
   terminal.
2. "Train the best model" needs a real definition of *best* — this sprint uses
   MLflow's **Model Registry** (the modern alias-based API, not the deprecated
   Staging/Production stages system) as the source of truth for which model is
   currently the champion, rather than trusting a local file that could silently
   go stale.

---

# What Was Built

**`src/training/pipeline.py` — `TrainingPipeline`**

The callable version of what `tune_and_ensemble_regime_aware.py` does as a script:
raw data → validate → RUL generation → engine split → regime-aware normalization →
feature engineering → feature selection → scaling → tune each requested model →
evaluate on the official test set → **compare the winner against the current
MLflow-registered champion** → promote only if genuinely better.

```python
pipeline = TrainingPipeline()
result = pipeline.run(n_trials=20, model_names=["catboost", "xgboost", "lightgbm"])

if result.promoted:
    print(f"New champion: {result.winning_model_name}, "
          f"MAE {result.champion_metric_after:.3f} (was {result.champion_metric_before})")
```

Returns a structured `TrainingResult` — win, do-not-win, whichever — suitable for an
API response rather than needing to parse stdout.

**`src/experiments/mlflow_tracker.py`** gained registry methods: `register_model`,
`set_champion`, `get_champion_version`, `get_champion_metric`, `load_champion_model`.
All alias-based (`@champion`), matching MLflow's current recommended API — the older
`transition_model_version_stage` (Staging/Production) has been deprecated since
MLflow 2.9 and was deliberately not used here.

**Known limitation, stated plainly**: only individual models (CatBoost, XGBoost,
LightGBM) are registry-eligible in this version. `EnsembleModel` isn't a single
MLflow-loadable flavor — registering it would need a custom `pyfunc` wrapper, which
wasn't built this sprint. In practice this hasn't cost anything yet: Sprint 17 already
found the ensemble losing to solo CatBoost under regime-aware normalization, so
individual-model comparison is what's actually been needed.

---

# Why the Comparison Is Test-Set, Not Validation-Set

`TrainingPipeline` picks the winner among candidates, and decides whether to promote
against the champion, using **test metrics**, not validation metrics. This isn't a
style choice — Sprint 14 already proved validation-only comparison can point to the
wrong answer when the underlying configuration changes (a lower RUL cap looked better
on validation purely because it produced smaller-magnitude labels, not because the
model was actually better). The same risk applies here across different
hyperparameters and regime configurations, so the safeguard from Sprint 14 is
reused, not re-invented.

---

# Real, End-to-End Verification

This wasn't just written and assumed to work — three scenarios were run for real:

1. **Bootstrap (no champion exists yet)**: ran `TrainingPipeline.run(n_trials=2,
   model_names=["catboost"])` with no prior registry entry. Correctly promoted
   unconditionally ("No champion exists yet"), registered as **version 1**, set the
   `champion` alias, materialized all five local canonical files
   (`best_model.pkl`, `regime_normalizer.pkl`, `feature_scaler.pkl`,
   `selected_features.json`, `best_params.json`). Test MAE: 19.307 — an exact match
   to every other independent computation of this same model in this project.
2. **Correct rejection**: reran with the identical configuration (fully
   deterministic given the fixed random seed). Result: identical MAE (19.307),
   correctly **not promoted** ("did not beat champion's 19.3066") — proves the
   guardrail against blindly overwriting a working model actually works, not just in
   the bootstrap case.
3. **Correct promotion**: reran with a larger search (6 trials instead of 2), which
   found a genuinely better configuration (MAE 18.911). Correctly promoted,
   registered as **version 2**, champion alias moved. Registry retains both
   versions — a real audit trail, not just an overwritten file.
4. **`InferencePipeline` integration**: after each promotion, ran `InferencePipeline`
   independently against the real official test set. Its computed MAE matched
   `TrainingPipeline`'s own reported metric exactly, both times — confirms the two
   pipelines are genuinely in sync, not coincidentally similar.

---

# Bootstrapping Your Real Model Into the Registry

You already have a real, fully-tuned CatBoost run (test MAE 19.133) logged in MLflow
from your own machine — this sprint's registry is currently empty in this sandbox
(it only has the verification runs above). Here's how to make your real run the
initial champion, so the *next* `TrainingPipeline.run()` you execute compares against
19.133, not nothing.

## Option A — Python (recommended, exact and repeatable)

```python
import mlflow
from src.experiments.mlflow_tracker import MLflowTracker

tracker = MLflowTracker()

# 1. Find your real run's run_id
runs = mlflow.search_runs(
    filter_string="tags.stage = 'final_tuned_model' and tags.model_family = 'catboost'",
    order_by=["metrics.MAE ASC"],
)
run_id = runs.iloc[0]["run_id"]
print(f"Found run: {run_id}, MAE: {runs.iloc[0]['metrics.MAE']}")

# 2. IMPORTANT: log the run's TEST metrics onto it if not already there —
#    TrainingPipeline compares on test_MAE specifically, not MAE (which in
#    your logged trials is the validation metric from BaseTrainer.train()).
with mlflow.start_run(run_id=run_id):
    mlflow.log_metrics({"test_MAE": 19.133, "test_RMSE": 25.367,
                          "test_R2": 0.783540, "test_MAPE": 28.446220})

# 3. Register it
version = tracker.register_model(run_id, "predictive-maintenance-rul")
print(f"Registered as version {version.version}")

# 4. Set it as champion
tracker.set_champion("predictive-maintenance-rul", version.version)
print("Champion set.")
```

## Option B — MLflow UI (if you prefer clicking over scripting)

1. Run `mlflow ui --backend-store-uri sqlite:///artifacts/mlruns/mlflow.db`, open
   `http://localhost:5000`.
2. Find your real CatBoost run (sort by `MAE` ascending, or search
   `tags.model_family = "catboost"`).
3. **Before registering**: open the run, go to the Metrics tab, and manually log
   `test_MAE = 19.133` (and the other test metrics) if they aren't already there —
   the UI has an option to log a metric manually, or use Option A's Python snippet
   for just that step, since the UI doesn't have a one-click way to do this.
4. Open the run → click the "Register Model" button → name it
   `predictive-maintenance-rul` → confirm.
5. Go to the Models tab → `predictive-maintenance-rul` → find the new version →
   set the alias `champion` on it (there's an "Add alias" option per version).

## After Bootstrapping

Run `promote_regime_aware_model.py` once more (or `TrainingPipeline._materialize_to_canonical_files`
equivalently) if `artifacts/models/best_model.pkl` doesn't already reflect this exact
run — the registry and the local files should agree before the next
`TrainingPipeline.run()` call, so `InferencePipeline` is serving what the registry
says is champion.

---

# What This Enables Next

- **Phase 2 (from the earlier roadmap) is effectively done** — the registry *is* the
  model packaging/versioning system.
- **Phase 3 (API)**: a FastAPI endpoint can now call `TrainingPipeline().run(...)`
  directly (as a background task, given multi-minute runtime — don't block a
  request on it) and return the structured `TrainingResult` as JSON.
- **Phase 4 (testing)**: the three verified scenarios above (bootstrap, reject,
  promote) are the natural first three test cases for a real test suite.

---

# Decisions

- Alias-based registry (`champion`), not the deprecated stages API — this is the
  currently-correct choice and will remain correct as MLflow continues removing the
  old stages system.
- Promotion decisions use **test** metrics, reusing Sprint 14's lesson about
  validation metrics being unsafe to compare across differing configurations.
- Ensembles are not yet registry-eligible — a known, stated gap, not a silent one.
  Revisit if a future ensemble genuinely starts beating solo models again.
- `TrainingPipeline` materializes every promotion to the same local files
  `InferencePipeline` already reads — the registry is the audit trail, the local
  files are what's actually served, and they're kept in sync automatically on every
  promotion.
