"""
Pipeline/ensemble_evaluate.py — blend CatBoost (MLflow champion), LSTM,
and GRU predictions on the same 237-engine official test subset, and
report whether any combination beats every model individually.

Deliberately NOT importing MLflowInferencePipeline from Pipeline/
predict.py — cross-importing between two scripts in the same Pipeline/
package is exactly the pattern that caused the intermittent
`ModuleNotFoundError: No module named 'Pipeline'` on Windows for
train_gru.py (see docs/Sprint_15_LSTM_Cap150_Test_Evaluation.md). The
handful of lines needed to load the champion here are inlined instead.

Requires, all already trained on THIS machine (predictions must come
from real trained models, not be re-derived elsewhere):
- A CatBoost model registered as "champion" in MLflow (TrainingPipeline
  / train_with_tuning.py / train_with_best_params.py must have run at
  least once).
- artifacts/models/lstm_baseline_seed42_final.keras (python
  Pipeline/train_lstm.py --seed 42)
- artifacts/models/gru_baseline_seed42_final.keras (python
  Pipeline/train_gru.py --seed 42)

Usage:
    python Pipeline/ensemble_evaluate.py
    python Pipeline/ensemble_evaluate.py --seed 7   # if you trained LSTM/GRU with a different seed
"""
import argparse
import json
import sys
import tempfile
from pathlib import Path

import joblib
import mlflow
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config.config import TEST_DATA_PATH, TRAIN_DATA_PATH, RUL_DATA_PATH, ENGINE_COLUMN, ROLLING_WINDOW, LAGS, MODELS_DIR, RANDOM_STATE
from src.utils.constant import SENSOR_COLUMNS
from src.data.loader import DataLoader
from src.preprocessing.feature_engineer import FeatureEngineer
from src.deep_learning.data_prep import prepare_lstm_sequences
from src.deep_learning.dl_trainer import DLTrainer
from src.evaluation.evaluator import RegressionEvaluator
from src.experiments.mlflow_tracker import MLflowTracker
from src.exceptions.custom_exception import CustomException
from src.logger.logger import logger

TIME_COLUMN = "time_in_cycles"
REGISTRY_MODEL_NAME = "predictive-maintenance-rul-model"


def get_catboost_test_predictions(test_raw: pd.DataFrame, registry_model_name: str = REGISTRY_MODEL_NAME) -> pd.DataFrame:
    """Minimal, inlined version of Pipeline/predict.py's
    MLflowInferencePipeline: load the champion + its logged
    preprocessing artifacts from MLflow, predict one RUL per engine
    from its last cycle. Returns a (unit_number, predicted_RUL) frame
    for every engine in test_raw."""

    tracker = MLflowTracker()
    champion_info = tracker.get_champion_run_info(registry_model_name)
    if champion_info is None:
        raise CustomException(
            f"No champion set for '{registry_model_name}' — train CatBoost first "
            "(Pipeline/train_with_tuning.py), then re-run this script.", sys,
        )

    run_id = champion_info["run_id"]
    requires_regime_normalization = (
        mlflow.get_run(run_id).data.tags.get("requires_regime_normalizer") == "True"
    )
    model = mlflow.pyfunc.load_model(f"models:/{registry_model_name}@champion")

    with tempfile.TemporaryDirectory() as tmp_dir:
        local_dir = Path(mlflow.artifacts.download_artifacts(run_id=run_id, artifact_path="preprocessing", dst_path=tmp_dir))
        with open(local_dir / "selected_features.json") as f:
            final_features = json.load(f)["selected_features"]
        scaler = joblib.load(local_dir / "feature_scaler.pkl")
        regime_normalizer = joblib.load(local_dir / "regime_normalizer.pkl") if (local_dir / "regime_normalizer.pkl").exists() else None

    df = test_raw.copy()
    if requires_regime_normalization:
        df = regime_normalizer.transform(df)

    engineered = FeatureEngineer(sensor_columns=SENSOR_COLUMNS, group_column=ENGINE_COLUMN, rolling_window=ROLLING_WINDOW, lags=LAGS).transform(df)

    last_cycles = (
        engineered.sort_values([ENGINE_COLUMN, TIME_COLUMN])
        .groupby(ENGINE_COLUMN).tail(1)
        .sort_values(ENGINE_COLUMN).reset_index(drop=True)
    )

    scaled = pd.DataFrame(scaler.transform(last_cycles[list(scaler.feature_names_in_)]), columns=scaler.feature_names_in_)
    predictions = np.asarray(model.predict(scaled[final_features])).flatten()

    logger.info(f"CatBoost champion (version {champion_info['version']}) predicted {len(predictions)} engines.")

    return pd.DataFrame({ENGINE_COLUMN: last_cycles[ENGINE_COLUMN].values, "catboost": predictions})


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=RANDOM_STATE, help="Seed the LSTM/GRU checkpoints were trained with")
    parser.add_argument("--window-size", type=int, default=30)
    args = parser.parse_args()

    # LSTM/GRU test predictions, restricted to the 237 engines with >= window_size cycles
    seq_data = prepare_lstm_sequences(window_size=args.window_size)
    lstm = DLTrainer.load(MODELS_DIR / f"lstm_baseline_seed{args.seed}_final.keras")
    gru = DLTrainer.load(MODELS_DIR / f"gru_baseline_seed{args.seed}_final.keras")

    preds = pd.DataFrame({
        ENGINE_COLUMN: seq_data["test_engine_ids"],
        "lstm": lstm.predict(seq_data["X_test"]),
        "gru": gru.predict(seq_data["X_test"]),
        "y_true": seq_data["y_test_true"],
    })

    # CatBoost predicts on every engine — restrict to the same 237
    loader = DataLoader(train_path=TRAIN_DATA_PATH, test_path=TEST_DATA_PATH, rul_path=RUL_DATA_PATH)
    catboost_preds = get_catboost_test_predictions(loader.load_test())
    preds = preds.merge(catboost_preds, on=ENGINE_COLUMN, how="left")

    if preds["catboost"].isna().any():
        raise CustomException("Some engines are missing a CatBoost prediction — check champion/engine alignment.", sys)

    evaluator = RegressionEvaluator()
    y_true = preds["y_true"].to_numpy()

    combos = {
        "CatBoost (champion, solo)": preds["catboost"],
        "LSTM (solo)": preds["lstm"],
        "GRU (solo)": preds["gru"],
        "Ensemble: CatBoost + LSTM": (preds["catboost"] + preds["lstm"]) / 2,
        "Ensemble: CatBoost + GRU": (preds["catboost"] + preds["gru"]) / 2,
        "Ensemble: LSTM + GRU": (preds["lstm"] + preds["gru"]) / 2,
        "Ensemble: all three (equal weight)": (preds["catboost"] + preds["lstm"] + preds["gru"]) / 3,
    }

    print("\n" + "=" * 70)
    print(f"ENSEMBLE COMPARISON — {len(preds)} test engines, seed={args.seed}")
    print("=" * 70)
    print(f"{'Combination':<38}{'MAE':>10}{'RMSE':>10}{'R2':>10}")
    for name, prediction in combos.items():
        metrics = evaluator.evaluate(y_true, prediction.to_numpy())
        print(f"{name:<38}{metrics['MAE']:>10.2f}{metrics['RMSE']:>10.2f}{metrics['R2']:>10.3f}")
