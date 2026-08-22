"""
RUL Cap Investigation (Sprint 14).

Isolates ONE variable: the RUL cap used to generate training targets.
Everything else is held fixed across every experiment:
  - Same 109 frozen features (Sprint 10)
  - Same Optuna hyperparameters (Sprint 11's best_params.json)
  - Same engine-level train/validation split (same random_state)
  - Same feature scaler (fit once on features, which don't depend on cap)
  - Same official test set (test_FD004 / RUL_FD004), same prep pipeline as
    Sprint 13, evaluated against the same fixed, uncapped ground truth

Usage (one cap per call, results appended to a shared CSV so runs are
naturally resumable/checkpointed):

    python rul_cap_experiment.py --cap none   # no cap at all
    python rul_cap_experiment.py --cap 150
    python rul_cap_experiment.py --cap 175
    python rul_cap_experiment.py --cap 195

cap=125 is NOT rerun here — it's the existing Sprint 11/13 model and
results (val MAE 12.836, test MAE 20.256), reused as-is per the "don't
retrain what you don't have to" principle from Sprint 13.
"""
import argparse
import json
import sys
import time

sys.path.insert(0, "/home/claude/Predictive-Maintenance-RUL")

import joblib
import numpy as np
import pandas as pd

from src.config.config import (
    TRAIN_DATA_PATH, TEST_DATA_PATH, RUL_DATA_PATH,
    SCALERS_DIR, MODELS_DIR, SELECTED_FEATURES_PATH, REPORTS_DIR,
    VALIDATION_SIZE, RANDOM_STATE, TARGET_COLUMN, ENGINE_COLUMN,
    ROLLING_WINDOW, LAGS,
)
from src.utils.constant import SENSOR_COLUMNS
from src.data.loader import DataLoader
from src.preprocessing.rul_generator import RULGenerator
from src.preprocessing.feature_engineer import FeatureEngineer
from src.preprocessing.data_splitter import DataSplitter
from src.explainability.feature_reducer import FeatureReducer
from src.models.base_trainer import BaseTrainer
from src.models.model_factory import ModelFactory
from src.evaluation.evaluator import RegressionEvaluator

parser = argparse.ArgumentParser()
parser.add_argument("--cap", type=str, required=True, help="RUL cap, or 'none' for uncapped")
args = parser.parse_args()
cap = None if args.cap.lower() == "none" else int(args.cap)
cap_label = "no_cap" if cap is None else f"cap_{cap}"

print(f"=== RUL Cap Experiment: {cap_label} ===")

RESULTS_PATH = REPORTS_DIR / "rul_cap_experiment_results.csv"
CAP_MODELS_DIR = MODELS_DIR / "cap_experiments"
CAP_MODELS_DIR.mkdir(parents=True, exist_ok=True)

# --- 1. Build training features with this cap's RUL target ---
loader = DataLoader(train_path=TRAIN_DATA_PATH, test_path=TEST_DATA_PATH, rul_path=RUL_DATA_PATH)
train_raw = loader.load_train()

rul_gen = RULGenerator(train_raw)
train_with_rul = rul_gen.generate(cap=cap)
print(f"Training RUL range with cap={cap}: [{train_with_rul[TARGET_COLUMN].min()}, {train_with_rul[TARGET_COLUMN].max()}]")

engineer = FeatureEngineer(sensor_columns=SENSOR_COLUMNS, rolling_window=ROLLING_WINDOW, lags=LAGS)
feature_df = engineer.transform(train_with_rul)
print(f"Engineered: {feature_df.shape}")

# --- 2. Same engine split as every prior sprint ---
splitter = DataSplitter(test_size=VALIDATION_SIZE, engine_column=ENGINE_COLUMN, random_state=RANDOM_STATE)
train_df, val_df = splitter.split(feature_df)
print(f"Train engines: {train_df[ENGINE_COLUMN].nunique()} | Val engines: {val_df[ENGINE_COLUMN].nunique()}")

final_features = FeatureReducer.load_selected_features(SELECTED_FEATURES_PATH)

# --- 3. Same scaler as every prior sprint (features don't depend on cap) ---
scaler = joblib.load(SCALERS_DIR / "feature_scaler.pkl")
scaler_columns = list(scaler.feature_names_in_)

X_train_full = pd.DataFrame(scaler.transform(train_df[scaler_columns]), columns=scaler_columns, index=train_df.index)
X_val_full = pd.DataFrame(scaler.transform(val_df[scaler_columns]), columns=scaler_columns, index=val_df.index)

X_train = X_train_full[final_features]
y_train = train_df[TARGET_COLUMN].reset_index(drop=True)
X_val = X_val_full[final_features]
y_val = val_df[TARGET_COLUMN].reset_index(drop=True)

# --- 4. Same frozen Optuna hyperparameters ---
best_params = json.loads((MODELS_DIR / "best_params.json").read_text())["params"]
print(f"Training CatBoost with frozen Sprint 11 hyperparameters: {best_params}")

model = ModelFactory.create("catboost", **best_params)
trainer = BaseTrainer(model)

start = time.time()
trainer.train(X_train, y_train)
training_time = time.time() - start
print(f"Training took {training_time:.1f}s")

model_path = CAP_MODELS_DIR / f"catboost_{cap_label}.pkl"
trainer.save(model_path)
print(f"Saved -> {model_path}")

# --- 5. Internal validation (cap-consistent labels) ---
evaluator = RegressionEvaluator()
val_preds = trainer.predict(X_val)
val_metrics = evaluator.evaluate(y_val, val_preds)
print(f"Validation metrics: {val_metrics}")

# --- 6. Official test set (Sprint 13's exact protocol, fixed uncapped ground truth) ---
test_df = loader.load_test()
rul_df = loader.load_rul()

test_features_df = engineer.transform(test_df)
last_rows = (
    test_features_df.sort_values([ENGINE_COLUMN, "time_in_cycles"])
    .groupby(ENGINE_COLUMN).tail(1)
    .sort_values(ENGINE_COLUMN).reset_index(drop=True)
)
X_test_full = pd.DataFrame(scaler.transform(last_rows[scaler_columns]), columns=scaler_columns, index=last_rows.index)
X_test = X_test_full[final_features]

y_test_true = rul_df["RUL"].to_numpy()
y_test_pred = trainer.predict(X_test)

test_metrics = evaluator.evaluate(y_test_true, y_test_pred)
print(f"Test metrics: {test_metrics}")
print(f"Test y_pred range: [{y_test_pred.min():.1f}, {y_test_pred.max():.1f}] (y_true range: [{y_test_true.min()}, {y_test_true.max()}])")

# --- 7. Append to shared results CSV (resumable/checkpointed by design) ---
row = {
    "cap": cap if cap is not None else "none",
    "train_rul_max": train_with_rul[TARGET_COLUMN].max(),
    "val_MAE": val_metrics["MAE"], "val_RMSE": val_metrics["RMSE"], "val_R2": val_metrics["R2"],
    "test_MAE": test_metrics["MAE"], "test_RMSE": test_metrics["RMSE"],
    "test_R2": test_metrics["R2"], "test_MAPE": test_metrics["MAPE"],
    "pred_max": y_test_pred.max(), "training_time_s": round(training_time, 1),
}

if RESULTS_PATH.exists():
    results_df = pd.read_csv(RESULTS_PATH)
    results_df = results_df[results_df["cap"].astype(str) != str(row["cap"])]  # replace if rerun
    results_df = pd.concat([results_df, pd.DataFrame([row])], ignore_index=True)
else:
    results_df = pd.DataFrame([row])

results_df.to_csv(RESULTS_PATH, index=False)
print(f"\nAppended to {RESULTS_PATH}")
print(results_df)
print("DONE")
