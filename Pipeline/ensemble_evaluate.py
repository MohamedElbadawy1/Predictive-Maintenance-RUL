"""
Pipeline/ensemble_evaluate.py — blend CatBoost (MLflow champion), LSTM,
and GRU predictions on the same 237-engine official test subset, and
report whether any combination beats every model individually.

CatBoost's test predictions come from src/training/data_prep.py's
prepare_training_data() — the SAME feature-preparation path
TrainingPipeline itself uses to score and log the champion's metrics —
rather than from Pipeline/predict.py's MLflowInferencePipeline.
Deliberate: MLflowInferencePipeline currently reproduces a much worse
score (R^2 < 0) than the champion's own logged metrics on this same
test set, confirmed against the *unmodified* Pipeline/predict.py, not
just this script — a real pre-existing training/serving skew bug
(training-time and serving-time feature reconstruction disagree
somewhere), unrelated to this reconstruction pass. See the "Known
issue" note in docs/Sprint_15_LSTM_Cap150_Test_Evaluation.md. Until
that's root-caused, prepare_training_data is the trustworthy path for
CatBoost's predictions here; it is not a fix for Pipeline/predict.py,
which is what actually needs to serve predictions from a fresh raw
file in production.

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
import sys
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config.config import ENGINE_COLUMN, MODELS_DIR, RANDOM_STATE
from src.training.data_prep import prepare_training_data
from src.deep_learning.data_prep import prepare_lstm_sequences
from src.deep_learning.dl_trainer import DLTrainer
from src.evaluation.evaluator import RegressionEvaluator
from src.experiments.mlflow_tracker import MLflowTracker
from src.exceptions.custom_exception import CustomException
from src.logger.logger import logger

REGISTRY_MODEL_NAME = "predictive-maintenance-rul-model"


def get_catboost_test_predictions(registry_model_name: str = REGISTRY_MODEL_NAME) -> pd.DataFrame:
    """Get CatBoost's test predictions via the SAME path TrainingPipeline
    itself uses to evaluate and log the champion's metrics
    (prepare_training_data), rather than reconstructing features
    through Pipeline/predict.py's separate MLflowInferencePipeline.

    This matters: MLflowInferencePipeline currently reproduces a much
    worse score than the champion's own logged metrics on this same
    test set (a real "training/serving skew" bug — training-time and
    serving-time feature reconstruction disagree somewhere). Until
    that's root-caused, prepare_training_data is the trustworthy path,
    since its output is exactly what produced the champion's logged
    MAE. Fine for this ensemble script; NOT a substitute for fixing
    Pipeline/predict.py, which is what actually serves predictions
    from a fresh raw-data file in production."""

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

    training_data = prepare_training_data(regime_aware=requires_regime_normalization)
    predictions = np.asarray(model.predict(training_data["X_test"])).flatten()

    # prepare_training_data's X_test rows are already sorted by engine
    # number ascending (1..N) — unit_number isn't a column of X_test
    # itself, so rebuild it from the row position.
    engine_ids = np.arange(1, len(predictions) + 1)

    logger.info(f"CatBoost champion (version {champion_info['version']}, logged test MAE={champion_info['metrics'].get('MAE')}) predicted {len(predictions)} engines via prepare_training_data.")
    logger.info(f"[DEBUG] predictions stats: min={predictions.min():.2f} max={predictions.max():.2f} mean={predictions.mean():.2f}")

    return pd.DataFrame({ENGINE_COLUMN: engine_ids, "catboost": predictions})


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
    catboost_preds = get_catboost_test_predictions()
    preds = preds.merge(catboost_preds, on=ENGINE_COLUMN, how="left")

    if preds["catboost"].isna().any():
        raise CustomException("Some engines are missing a CatBoost prediction — check champion/engine alignment.", sys)

    print("\n[DEBUG] y_true stats:", preds["y_true"].describe().to_dict())
    print("[DEBUG] first 5 rows (engine, catboost_pred, lstm_pred, gru_pred, y_true):")
    print(preds[[ENGINE_COLUMN, "catboost", "lstm", "gru", "y_true"]].head(5).to_string(index=False))

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
