"""
Regime-Aware Normalization — Controlled A/B Experiment

Isolates ONE variable: whether sensors are normalized globally (current
pipeline) or per-operating-regime (proposed fix for FD004's 6 operating
conditions confounding raw sensor values). Everything else is held
fixed: same frozen CatBoost hyperparameters (from the real tuning run),
same engine split, same feature-selection decision (exclude rolling),
same official test-set protocol.

Correct fit/transform ordering to avoid leakage: engines are split into
train/validation BEFORE feature engineering (not after, as the original
pipeline does) — RegimeNormalizer's K-Means and per-regime scalers must
only ever be fit on training engines' rows, and only fitting AFTER the
split guarantees that. Splitting first doesn't change the rolling/lag
features' correctness, since those are computed per-engine
(groupby(unit_number)) regardless of which rows are physically split
into which DataFrame.

Usage:
    python regime_normalization_experiment.py
"""
import sys
sys.path.insert(0, "/home/claude/Predictive-Maintenance-RUL")

import json
import joblib
import pandas as pd

from src.config.config import (
    TRAIN_DATA_PATH, TEST_DATA_PATH, RUL_DATA_PATH,
    MODELS_DIR, REPORTS_DIR,
    VALIDATION_SIZE, RANDOM_STATE, DEFAULT_RUL_CAP,
    ROLLING_WINDOW, LAGS, ENGINE_COLUMN, TARGET_COLUMN,
)
from src.utils.constant import SENSOR_COLUMNS
from src.data.loader import DataLoader
from src.preprocessing.rul_generator import RULGenerator
from src.preprocessing.feature_engineer import FeatureEngineer
from src.preprocessing.data_splitter import DataSplitter
from src.preprocessing.feature_scaler import FeatureScaler
from src.preprocessing.regime_normalizer import RegimeNormalizer
from src.explainability.feature_selector import FeatureCategorySelector
from src.models.model_factory import ModelFactory
from src.models.base_trainer import BaseTrainer
from src.evaluation.evaluator import RegressionEvaluator

# Frozen hyperparameters from the real tuning run — held IDENTICAL across
# both arms of this experiment, so any difference in results is caused
# by normalization strategy alone, not a confound from different params.
FROZEN_PARAMS = json.load(open(MODELS_DIR / "catboost_best_params.json"))["params"]
print(f"Using frozen CatBoost params: {FROZEN_PARAMS}\n")

evaluator = RegressionEvaluator()


def run_arm(regime_aware: bool):

    label = "regime_normalized" if regime_aware else "baseline_global_scaling"
    print(f"\n{'='*70}\nARM: {label}\n{'='*70}")

    # --- Load + RUL (identical for both arms) ---
    loader = DataLoader(train_path=TRAIN_DATA_PATH, test_path=TEST_DATA_PATH, rul_path=RUL_DATA_PATH)
    train_raw = loader.load_train()
    test_raw = loader.load_test()
    rul_raw = loader.load_rul()
    train_with_rul = RULGenerator(train_raw).generate(cap=DEFAULT_RUL_CAP)

    # --- Split by engine FIRST (before feature engineering), so any
    # regime-fitting below only ever sees training engines ---
    splitter = DataSplitter(test_size=VALIDATION_SIZE, engine_column=ENGINE_COLUMN, random_state=RANDOM_STATE)
    train_split, val_split = splitter.split(train_with_rul)
    print(f"Train: {train_split[ENGINE_COLUMN].nunique()} engines | Val: {val_split[ENGINE_COLUMN].nunique()} engines")

    regime_normalizer = None
    if regime_aware:
        regime_normalizer = RegimeNormalizer(n_regimes=6, sensor_columns=SENSOR_COLUMNS, random_state=RANDOM_STATE)
        train_split = regime_normalizer.fit(train_split).transform(train_split)
        val_split = regime_normalizer.transform(val_split)
        test_raw = regime_normalizer.transform(test_raw)
        print(f"Regime distribution (train): {train_split['regime'].value_counts().sort_index().to_dict()}")

    # --- Feature engineering on (possibly regime-normalized) sensor values ---
    engineer = FeatureEngineer(sensor_columns=SENSOR_COLUMNS, rolling_window=ROLLING_WINDOW, lags=LAGS)
    train_features = engineer.transform(train_split)
    val_features = engineer.transform(val_split)

    # --- Feature selection: same categorical decision as the real pipeline ---
    all_columns = [c for c in train_features.columns if c not in (ENGINE_COLUMN, TARGET_COLUMN, "regime")]
    final_features = FeatureCategorySelector.exclude(all_columns, categories=["rolling"])
    print(f"Selected {len(final_features)} features")

    # --- Final global scaling of the selected features (fit on train only) ---
    scaler = FeatureScaler()
    X_train = scaler.fit_transform(train_features[final_features])
    X_val = scaler.transform(val_features[final_features])
    y_train = train_features[TARGET_COLUMN].reset_index(drop=True)
    y_val = val_features[TARGET_COLUMN].reset_index(drop=True)

    # --- Train with the FROZEN hyperparameters (no re-tuning here — that
    # would confound the comparison) ---
    model = ModelFactory.create("catboost", **FROZEN_PARAMS)
    trainer = BaseTrainer(model, track_mlflow=True, run_name=f"regime_ab_{label}",
                           tags={"experiment": "regime_normalization_ab", "arm": label})
    val_metrics = trainer.train(X_train, y_train, X_val, y_val)
    print(f"Validation metrics: {val_metrics}")

    # --- Official test-set evaluation ---
    test_features_df = engineer.transform(test_raw)
    last_rows = (
        test_features_df.sort_values([ENGINE_COLUMN, "time_in_cycles"])
        .groupby(ENGINE_COLUMN).tail(1).sort_values(ENGINE_COLUMN).reset_index(drop=True)
    )
    X_test = scaler.transform(last_rows[final_features])
    y_test_true = rul_raw["RUL"].to_numpy()

    y_test_pred = trainer.predict(X_test)
    test_metrics = evaluator.evaluate(y_test_true, y_test_pred)
    print(f"Test metrics: {test_metrics}")

    return {"Arm": label, "Stage": "validation", **val_metrics}, {"Arm": label, "Stage": "test", **test_metrics}


results = []
for regime_aware in (False, True):
    val_row, test_row = run_arm(regime_aware)
    results.append(val_row)
    results.append(test_row)

results_df = pd.DataFrame(results)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)
results_df.to_csv(REPORTS_DIR / "regime_normalization_ab_results.csv", index=False)

print("\n" + "=" * 70)
print("FINAL COMPARISON")
print("=" * 70)
print(results_df.to_string(index=False))
print(f"\nSaved -> {REPORTS_DIR / 'regime_normalization_ab_results.csv'}")
