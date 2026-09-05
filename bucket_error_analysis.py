"""
Bucket-level error analysis for the canonical model, on the official
test set. Same methodology as Sprints 13/14: aggregate MAE hides where
the model actually struggles or excels — breaking it down by true RUL
range is what actually revealed the RUL-cap problem back in Sprint 13,
and it's the fastest way to find out whether the regime-aware
normalization win (Sprint 17) is spread evenly or concentrated somewhere
specific.

IMPORTANT: this script loads whatever is currently at
artifacts/models/best_model.pkl. If you've run promote_regime_aware_model.py
with YOUR real (fully-tuned) results, this analyzes YOUR real model. Numbers
will differ from a lightly-tuned verification run.

Usage:
    python bucket_error_analysis.py
"""
import sys
sys.path.insert(0, "/home/claude/Predictive-Maintenance-RUL")

import json
import joblib
import numpy as np
import pandas as pd

from src.config.config import (
    TRAIN_DATA_PATH, TEST_DATA_PATH, RUL_DATA_PATH,
    MODELS_DIR, SCALERS_DIR, SELECTED_FEATURES_PATH, REPORTS_DIR,
    ROLLING_WINDOW, LAGS, ENGINE_COLUMN,
)
from src.utils.constant import SENSOR_COLUMNS
from src.data.loader import DataLoader
from src.preprocessing.feature_engineer import FeatureEngineer
from src.preprocessing.regime_normalizer import RegimeNormalizer
from src.explainability.feature_reducer import FeatureReducer
from src.models.base_trainer import BaseTrainer
from src.evaluation.evaluator import RegressionEvaluator

# --- Load raw test data + official ground truth ---
loader = DataLoader(train_path=TRAIN_DATA_PATH, test_path=TEST_DATA_PATH, rul_path=RUL_DATA_PATH)
test_raw = loader.load_test()
rul_raw = loader.load_rul()
print(f"test_FD004: {test_raw.shape}  RUL_FD004: {rul_raw.shape}")

# --- Check whether the canonical model requires regime normalization ---
with open(MODELS_DIR / "best_params.json") as f:
    best_params_record = json.load(f)
requires_regime = best_params_record.get("requires_regime_normalizer", False)
print(f"Canonical model requires regime normalization: {requires_regime}")

if requires_regime:
    regime_normalizer = RegimeNormalizer.load(MODELS_DIR / "regime_normalizer.pkl")
    test_raw = regime_normalizer.transform(test_raw)  # predict(), never fit, on test
    print(f"Applied regime normalization. Regime distribution: "
          f"{test_raw['regime'].value_counts().sort_index().to_dict()}")

# --- Feature engineering -> last cycle per engine -> scale -> select ---
engineer = FeatureEngineer(sensor_columns=SENSOR_COLUMNS, rolling_window=ROLLING_WINDOW, lags=LAGS)
test_features_df = engineer.transform(test_raw)

last_rows = (
    test_features_df.sort_values([ENGINE_COLUMN, "time_in_cycles"])
    .groupby(ENGINE_COLUMN).tail(1).sort_values(ENGINE_COLUMN).reset_index(drop=True)
)

scaler = joblib.load(SCALERS_DIR / "feature_scaler.pkl")
scaler_columns = list(scaler.feature_names_in_)
X_test_full = pd.DataFrame(scaler.transform(last_rows[scaler_columns]), columns=scaler_columns)

final_features = FeatureReducer.load_selected_features(SELECTED_FEATURES_PATH)
X_test = X_test_full[final_features]

# --- Predict with the canonical model, no retraining ---
trainer = BaseTrainer.load(MODELS_DIR / "best_model.pkl", track_mlflow=False)
y_pred = trainer.predict(X_test)
y_true = rul_raw["RUL"].to_numpy()

evaluator = RegressionEvaluator()
overall_metrics = evaluator.evaluate(y_true, y_pred)
print(f"\nOverall test metrics: {overall_metrics}")

# --- Per-engine results ---
results_df = pd.DataFrame({
    "unit_number": last_rows[ENGINE_COLUMN].values,
    "actual_RUL": y_true,
    "predicted_RUL": y_pred,
    "error": y_pred - y_true,
    "abs_error": np.abs(y_pred - y_true),
})

# --- Bucket analysis: same RUL ranges used since Sprint 13, for direct comparability ---
bins = [0, 25, 50, 75, 125, 1000]
labels = ["0-25 (critical)", "25-50", "50-75", "75-125", "125+ (beyond old cap)"]
results_df["rul_bucket"] = pd.cut(results_df["actual_RUL"], bins=bins, labels=labels)

bucket_stats = results_df.groupby("rul_bucket", observed=True).agg(
    n=("actual_RUL", "size"),
    mae=("abs_error", "mean"),
    mean_error=("error", "mean"),
).round(3)

print("\n=== Bucket-level performance ===")
print(bucket_stats)

print("\n=== Overall bias ===")
print(f"Mean error (bias): {results_df['error'].mean():.3f}")
print(f"Median error: {results_df['error'].median():.3f}")
print(f"Over-predicted: {(results_df['error']>0).mean()*100:.1f}%  |  "
      f"Under-predicted: {(results_df['error']<0).mean()*100:.1f}%")

print("\n=== Top 10 largest errors ===")
print(results_df.nlargest(10, "abs_error")[["unit_number", "actual_RUL", "predicted_RUL", "error"]].to_string(index=False))

# --- Save everything ---
REPORTS_DIR.mkdir(parents=True, exist_ok=True)
results_df.to_csv(REPORTS_DIR / "canonical_model_test_predictions.csv", index=False)
bucket_stats.to_csv(REPORTS_DIR / "canonical_model_bucket_analysis.csv")
print(f"\nSaved -> {REPORTS_DIR / 'canonical_model_test_predictions.csv'}")
print(f"Saved -> {REPORTS_DIR / 'canonical_model_bucket_analysis.csv'}")
