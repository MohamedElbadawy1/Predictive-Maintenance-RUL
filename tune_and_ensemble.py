"""
Full pipeline, start to finish: raw data ingestion -> validation -> RUL
generation -> feature engineering -> feature selection -> train/val
split -> scaling -> hyperparameter tuning (CatBoost + XGBoost +
LightGBM) -> weighted ensemble -> evaluation on validation AND the
official test set.

Self-contained on purpose: does not assume train_prepared.csv,
selected_features.json, or feature_scaler.pkl already exist — every
artifact is (re)built from the raw NASA files in this run, and freshly
saved at the end. Safe to run after a clean checkout or a deleted
artifacts/ folder.

Every tuning trial is automatically logged to MLflow (via BaseTrainer's
built-in tracking) — no separate logging step.

Usage:
    python tune_and_ensemble.py --n-trials 20
    python tune_and_ensemble.py --n-trials 20 --skip-test-eval   # faster, val only
"""
import argparse
import json
import sys
import time

sys.path.insert(0, "/home/claude/Predictive-Maintenance-RUL")

import joblib
import pandas as pd

from src.config.config import (
    TRAIN_DATA_PATH, TEST_DATA_PATH, RUL_DATA_PATH,
    MODELS_DIR, REPORTS_DIR, SCALERS_DIR, SELECTED_FEATURES_PATH,
    VALIDATION_SIZE, RANDOM_STATE, DEFAULT_RUL_CAP,
    ROLLING_WINDOW, LAGS, ENGINE_COLUMN, TARGET_COLUMN,
)
from src.utils.constant import SENSOR_COLUMNS
from src.data.loader import DataLoader
from src.data.validator import DataValidator
from src.preprocessing.rul_generator import RULGenerator
from src.preprocessing.feature_engineer import FeatureEngineer
from src.preprocessing.data_splitter import DataSplitter
from src.preprocessing.feature_scaler import FeatureScaler
from src.explainability.feature_selector import FeatureCategorySelector
from src.explainability.feature_reducer import FeatureReducer
from src.optimization.hyperparameter_tuner import ModelTuner
from src.models.model_factory import ModelFactory
from src.models.base_trainer import BaseTrainer
from src.models.ensemble import EnsembleModel
from src.evaluation.evaluator import RegressionEvaluator

parser = argparse.ArgumentParser()
parser.add_argument("--n-trials", type=int, default=20, help="Optuna trials per model")
parser.add_argument("--skip-test-eval", action="store_true", help="Skip official test-set evaluation (faster)")
args = parser.parse_args()

MODEL_NAMES = ["catboost", "xgboost", "lightgbm"]
ENSEMBLE_DIR = MODELS_DIR / "ensemble"

print("=" * 70)
print("FULL PIPELINE: raw data -> tuned multi-model ensemble")
print("=" * 70)

# --- 1. Load raw data ---
print("\n--- 1. Loading raw data ---")
loader = DataLoader(train_path=TRAIN_DATA_PATH, test_path=TEST_DATA_PATH, rul_path=RUL_DATA_PATH)
train_raw = loader.load_train()
test_raw = loader.load_test()
rul_raw = loader.load_rul()
print(f"train_FD004: {train_raw.shape}  test_FD004: {test_raw.shape}  RUL_FD004: {rul_raw.shape}")

# --- 2. Validate ---
print("\n--- 2. Validating ---")
report = DataValidator(train_raw, test_raw, rul_raw).validate_all()
print(report)

# --- 3. RUL generation ---
print(f"\n--- 3. Generating RUL (cap={DEFAULT_RUL_CAP}) ---")
train_with_rul = RULGenerator(train_raw).generate(cap=DEFAULT_RUL_CAP)
print(f"RUL range: [{train_with_rul[TARGET_COLUMN].min()}, {train_with_rul[TARGET_COLUMN].max()}]")

# --- 4. Feature engineering ---
print("\n--- 4. Feature engineering ---")
engineer = FeatureEngineer(sensor_columns=SENSOR_COLUMNS, rolling_window=ROLLING_WINDOW, lags=LAGS)
feature_df = engineer.transform(train_with_rul)
print(f"Engineered: {feature_df.shape[1]} columns ({feature_df.shape[1] - 2} features)")

# --- 5. Feature selection (reproduces Sprint 10's real, already-validated decision) ---
print("\n--- 5. Feature selection ---")
all_columns = [c for c in feature_df.columns if c not in (ENGINE_COLUMN, TARGET_COLUMN)]
final_features = FeatureCategorySelector.exclude(all_columns, categories=["rolling"])
print(f"Selected {len(final_features)} features (dropped rolling features — Sprint 10's validated result)")

MODELS_DIR.mkdir(parents=True, exist_ok=True)
reducer = FeatureReducer(keep_features=final_features)
reducer.fit(feature_df[all_columns])
reducer.save_selected_features(SELECTED_FEATURES_PATH)

# --- 6. Train/validation split (by engine, never by row) ---
print("\n--- 6. Train/validation split ---")
splitter = DataSplitter(test_size=VALIDATION_SIZE, engine_column=ENGINE_COLUMN, random_state=RANDOM_STATE)
train_split, val_split = splitter.split(feature_df)
print(f"Train: {train_split[ENGINE_COLUMN].nunique()} engines ({train_split.shape[0]} rows) | "
      f"Val: {val_split[ENGINE_COLUMN].nunique()} engines ({val_split.shape[0]} rows)")

# --- 7. Scaling (fit on train only) ---
print("\n--- 7. Scaling ---")
scaler = FeatureScaler()
X_train = scaler.fit_transform(train_split[final_features])
X_val = scaler.transform(val_split[final_features])
y_train = train_split[TARGET_COLUMN].reset_index(drop=True)
y_val = val_split[TARGET_COLUMN].reset_index(drop=True)

SCALERS_DIR.mkdir(parents=True, exist_ok=True)
scaler.save(SCALERS_DIR / "feature_scaler.pkl")
print(f"X_train: {X_train.shape}  X_val: {X_val.shape}")

# --- 8. Tune each model ---
print(f"\n--- 8. Hyperparameter tuning ({args.n_trials} trials/model) ---")
evaluator = RegressionEvaluator()
tuned_models = {}
val_results = []

for model_name in MODEL_NAMES:

    print(f"\n  Tuning {model_name}...")
    tuner = ModelTuner(model_name, X_train, y_train, X_val, y_val, random_state=RANDOM_STATE)
    tuner.run(n_trials=args.n_trials, show_progress_bar=False)

    best_params = tuner.best_params()
    print(f"  {model_name} best params: {best_params}")

    final_model = ModelFactory.create(model_name, **best_params)
    final_trainer = BaseTrainer(
        final_model,
        run_name=f"{model_name}_final_tuned",
        tags={"model_family": model_name, "stage": "final_tuned_model"},
    )
    metrics = final_trainer.train(X_train, y_train, X_val, y_val)
    print(f"  {model_name} final validation metrics: {metrics}")

    tuned_models[model_name] = final_model
    val_results.append({"Model": model_name, "Stage": "tuned_individual", **metrics})

    joblib.dump(final_model, MODELS_DIR / f"{model_name}_tuned.pkl")
    with open(MODELS_DIR / f"{model_name}_best_params.json", "w") as f:
        json.dump({"params": best_params, "metrics": metrics}, f, indent=2)

# --- 9. Build the ensemble ---
print("\n--- 9. Building ensemble ---")
val_mae_by_model = {r["Model"]: r["MAE"] for r in val_results}
ensemble = EnsembleModel.from_inverse_mae(tuned_models, val_mae_by_model)

ensemble_val_preds = ensemble.predict(X_val)
ensemble_val_metrics = evaluator.evaluate(y_val, ensemble_val_preds)
print(f"Ensemble weights: {ensemble.weights}")
print(f"Ensemble validation metrics: {ensemble_val_metrics}")
val_results.append({"Model": "ensemble", "Stage": "tuned_ensemble", **ensemble_val_metrics})

ENSEMBLE_DIR.mkdir(parents=True, exist_ok=True)
ensemble.save(ENSEMBLE_DIR)
print(f"Ensemble saved -> {ENSEMBLE_DIR}")

val_results_df = pd.DataFrame(val_results)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)
val_results_df.to_csv(REPORTS_DIR / "tuned_ensemble_validation_results.csv", index=False)
print("\nValidation comparison:")
print(val_results_df)

# --- 10. Official test-set evaluation ---
if not args.skip_test_eval:

    print("\n--- 10. Official test-set evaluation ---")
    # test_raw/rul_raw were already loaded in step 1 -- reused here, not reloaded.
    test_features_df = engineer.transform(test_raw)
    last_rows = (
        test_features_df.sort_values([ENGINE_COLUMN, "time_in_cycles"])
        .groupby(ENGINE_COLUMN).tail(1).sort_values(ENGINE_COLUMN).reset_index(drop=True)
    )

    X_test = scaler.transform(last_rows[final_features])
    y_test_true = rul_raw["RUL"].to_numpy()

    test_results = []
    for model_name, model in tuned_models.items():
        preds = model.predict(X_test)
        metrics = evaluator.evaluate(y_test_true, preds)
        test_results.append({"Model": model_name, "Stage": "tuned_individual", **metrics})

    ensemble_test_preds = ensemble.predict(X_test)
    ensemble_test_metrics = evaluator.evaluate(y_test_true, ensemble_test_preds)
    test_results.append({"Model": "ensemble", "Stage": "tuned_ensemble", **ensemble_test_metrics})

    test_results_df = pd.DataFrame(test_results)
    test_results_df.to_csv(REPORTS_DIR / "tuned_ensemble_test_results.csv", index=False)
    print("\nTest set comparison:")
    print(test_results_df)

print("\nDONE")
