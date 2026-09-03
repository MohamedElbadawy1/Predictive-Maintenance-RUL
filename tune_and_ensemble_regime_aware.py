"""
Full pipeline with regime-aware normalization: raw data ingestion ->
validation -> RUL generation -> engine split -> regime detection &
per-regime sensor normalization -> feature engineering -> feature
selection -> global scaling -> hyperparameter tuning (CatBoost + XGBoost
+ LightGBM, all three) -> weighted ensemble -> evaluation on validation
AND the official test set.

Builds on tune_and_ensemble.py with one structural change, driven by
RegimeNormalizer's leakage requirement: engines are split into
train/validation BEFORE feature engineering (not after), because the
regime detector (K-Means) and per-regime scalers must only ever be
fit on training engines. This was verified in a controlled A/B
experiment (regime_normalization_experiment.py) before this fuller
version was built — that experiment held hyperparameters fixed to
isolate normalization strategy alone; this script re-tunes every model
fresh against the new (regime-normalized) feature representation, since
the best hyperparameters for one feature representation aren't
guaranteed to be best for another.

Every tuning trial is automatically logged to MLflow (BaseTrainer's
built-in tracking).

Usage:
    python tune_and_ensemble_regime_aware.py --n-trials 20
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
from src.preprocessing.regime_normalizer import RegimeNormalizer
from src.explainability.feature_selector import FeatureCategorySelector
from src.explainability.feature_reducer import FeatureReducer
from src.optimization.hyperparameter_tuner import ModelTuner
from src.models.model_factory import ModelFactory
from src.models.base_trainer import BaseTrainer
from src.models.ensemble import EnsembleModel
from src.evaluation.evaluator import RegressionEvaluator

parser = argparse.ArgumentParser()
parser.add_argument("--n-trials", type=int, default=20, help="Optuna trials per model")
parser.add_argument("--n-regimes", type=int, default=6, help="Operating regimes to detect")
parser.add_argument("--skip-test-eval", action="store_true")
args = parser.parse_args()

MODEL_NAMES = ["catboost", "xgboost", "lightgbm"]
ENSEMBLE_DIR = MODELS_DIR / "ensemble_regime_aware"
REGIME_NORMALIZER_PATH = MODELS_DIR / "regime_normalizer.pkl"

print("=" * 70)
print("REGIME-AWARE PIPELINE: raw data -> tuned multi-model ensemble")
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
print(DataValidator(train_raw, test_raw, rul_raw).validate_all())

# --- 3. RUL generation ---
print(f"\n--- 3. Generating RUL (cap={DEFAULT_RUL_CAP}) ---")
train_with_rul = RULGenerator(train_raw).generate(cap=DEFAULT_RUL_CAP)

# --- 4. Split by engine FIRST — required so the regime normalizer (step 5)
# only ever fits on training engines, never validation/test ---
print("\n--- 4. Train/validation split (before feature engineering, for regime-fit safety) ---")
splitter = DataSplitter(test_size=VALIDATION_SIZE, engine_column=ENGINE_COLUMN, random_state=RANDOM_STATE)
train_split, val_split = splitter.split(train_with_rul)
print(f"Train: {train_split[ENGINE_COLUMN].nunique()} engines ({train_split.shape[0]} rows) | "
      f"Val: {val_split[ENGINE_COLUMN].nunique()} engines ({val_split.shape[0]} rows)")

# --- 5. Regime detection + per-regime sensor normalization ---
print(f"\n--- 5. Regime-aware normalization ({args.n_regimes} regimes) ---")
regime_normalizer = RegimeNormalizer(n_regimes=args.n_regimes, sensor_columns=SENSOR_COLUMNS, random_state=RANDOM_STATE)
train_split = regime_normalizer.fit(train_split).transform(train_split)
val_split = regime_normalizer.transform(val_split)
test_raw = regime_normalizer.transform(test_raw)  # predict() only, fit on train above
print(f"Regime distribution (train): {train_split['regime'].value_counts().sort_index().to_dict()}")

MODELS_DIR.mkdir(parents=True, exist_ok=True)
regime_normalizer.save(REGIME_NORMALIZER_PATH)

# --- 6. Feature engineering on regime-normalized sensor values ---
print("\n--- 6. Feature engineering ---")
engineer = FeatureEngineer(sensor_columns=SENSOR_COLUMNS, rolling_window=ROLLING_WINDOW, lags=LAGS)
train_features = engineer.transform(train_split)
val_features = engineer.transform(val_split)
print(f"Engineered: {train_features.shape[1]} columns")

# --- 7. Feature selection (same categorical decision as the real pipeline) ---
print("\n--- 7. Feature selection ---")
all_columns = [c for c in train_features.columns if c not in (ENGINE_COLUMN, TARGET_COLUMN, "regime")]
final_features = FeatureCategorySelector.exclude(all_columns, categories=["rolling"])
print(f"Selected {len(final_features)} features")

reducer = FeatureReducer(keep_features=final_features)
reducer.fit(train_features[all_columns])
reducer.save_selected_features(MODELS_DIR / "selected_features_regime_aware.json")

# --- 8. Final global scaling of the selected features (fit on train only) ---
print("\n--- 8. Scaling ---")
scaler = FeatureScaler()
X_train = scaler.fit_transform(train_features[final_features])
X_val = scaler.transform(val_features[final_features])
y_train = train_features[TARGET_COLUMN].reset_index(drop=True)
y_val = val_features[TARGET_COLUMN].reset_index(drop=True)

SCALERS_DIR.mkdir(parents=True, exist_ok=True)
scaler.save(SCALERS_DIR / "feature_scaler_regime_aware.pkl")
print(f"X_train: {X_train.shape}  X_val: {X_val.shape}")

# --- 9. Tune each model fresh against the new feature representation ---
print(f"\n--- 9. Hyperparameter tuning ({args.n_trials} trials/model) ---")
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
        run_name=f"{model_name}_regime_aware_final_tuned",
        tags={"model_family": model_name, "stage": "final_tuned_model", "normalization": "regime_aware"},
    )
    metrics = final_trainer.train(X_train, y_train, X_val, y_val)
    print(f"  {model_name} final validation metrics: {metrics}")

    tuned_models[model_name] = final_model
    val_results.append({"Model": model_name, "Stage": "tuned_individual", **metrics})

    joblib.dump(final_model, MODELS_DIR / f"{model_name}_tuned_regime_aware.pkl")
    with open(MODELS_DIR / f"{model_name}_best_params_regime_aware.json", "w") as f:
        json.dump({"params": best_params, "metrics": metrics}, f, indent=2)

# --- 10. Build the ensemble ---
print("\n--- 10. Building ensemble ---")
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
val_results_df.to_csv(REPORTS_DIR / "regime_aware_ensemble_validation_results.csv", index=False)
print("\nValidation comparison:")
print(val_results_df)

# --- 11. Official test-set evaluation ---
if not args.skip_test_eval:

    print("\n--- 11. Official test-set evaluation ---")
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
    test_results_df.to_csv(REPORTS_DIR / "regime_aware_ensemble_test_results.csv", index=False)
    print("\nTest set comparison:")
    print(test_results_df)

    print("\n--- Comparison to prior best (non-regime-aware ensemble) ---")
    print("Prior best (global scaling): ensemble test MAE = 19.226")
    best_new_mae = test_results_df["MAE"].min()
    best_new_model = test_results_df.loc[test_results_df["MAE"].idxmin(), "Model"]
    print(f"This run's best: {best_new_model} test MAE = {best_new_mae:.3f}")

print("\nDONE")
