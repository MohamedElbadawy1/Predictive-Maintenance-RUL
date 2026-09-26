"""
src/pipelines/ensemble_inference_pipeline.py — EnsembleInferencePipeline:
serves the LSTM+GRU equal-weight average (this project's best result,
per README's "Current Best Model" -- test MAE 16.39 vs CatBoost's
18.72-19.13), reading everything it needs from
artifacts/models/ensemble/ (written by Pipeline/train_ensemble.py):
manifest.json, feature_scaler.pkl, and (when the manifest says
regime_aware=True) regime_normalizer.pkl.

Deliberately reads local files rather than MLflow, unlike
MLflowInferencePipeline: Keras models aren't registered in the MLflow
Model Registry here (train_lstm.py / train_gru.py log them as
comparison runs only, never as a "champion"), so there is no registry
alias to load them from. artifacts/ is already the shared, volume-
mounted location both serving paths use (see docker-compose.yml).

Preprocessing here intentionally mirrors
src/deep_learning/data_prep.py's prepare_lstm_sequences() step for
step (regime-normalize -> scale raw features -> window) rather than
reusing any CatBoost-path preprocessing. That match matters: this
project has a documented training/serving skew bug on the CatBoost
path specifically (see Pipeline/ensemble_evaluate.py's module
docstring and docs/Sprint_15_LSTM_Cap150_Test_Evaluation.md) caused by
training-time and serving-time feature reconstruction disagreeing.
Reusing the exact same raw-feature-columns + scaler + sequence-window
logic here (rather than re-deriving it) is what avoids repeating that
bug on this path.
"""
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.config.config import (
    ENSEMBLE_MANIFEST_PATH, ENSEMBLE_SCALER_PATH, ENSEMBLE_REGIME_NORMALIZER_PATH,
    MODELS_DIR, ENGINE_COLUMN,
)
from src.utils.constant import SENSOR_COLUMNS
from src.exceptions.custom_exception import CustomException
from src.deep_learning.dl_trainer import DLTrainer
from src.logger.logger import logger

TIME_COLUMN = "time_in_cycles"

REQUIRED_RAW_COLUMNS = (
    [ENGINE_COLUMN, TIME_COLUMN]
    + [f"operational_setting_{i}" for i in (1, 2, 3)]
    + SENSOR_COLUMNS
)


class EnsembleInferencePipeline:
    """
    Example
    -------
    pipeline = EnsembleInferencePipeline()
    predictions = pipeline.predict(raw_engine_readings)
    """

    def __init__(self):

        if not ENSEMBLE_MANIFEST_PATH.exists():
            raise CustomException(
                f"No ensemble manifest found at {ENSEMBLE_MANIFEST_PATH}. "
                "Run Pipeline/train_ensemble.py first to train the LSTM+GRU "
                "pair and write serving artifacts.", sys,
            )

        with open(ENSEMBLE_MANIFEST_PATH) as f:
            self.manifest = json.load(f)

        self.window_size = self.manifest["window_size"]
        self.regime_aware = self.manifest["regime_aware"]
        self.feature_columns = self.manifest["feature_columns"]
        self.version = f"lstm_gru_ensemble-seed{self.manifest['seed']}"

        logger.info(f"Loading ensemble serving artifacts from {ENSEMBLE_MANIFEST_PATH.parent}...")

        self.scaler = joblib.load(ENSEMBLE_SCALER_PATH)

        self.regime_normalizer = None
        if self.regime_aware:
            if not ENSEMBLE_REGIME_NORMALIZER_PATH.exists():
                raise CustomException(
                    f"manifest.json says regime_aware=True but "
                    f"{ENSEMBLE_REGIME_NORMALIZER_PATH} is missing.", sys,
                )
            self.regime_normalizer = joblib.load(ENSEMBLE_REGIME_NORMALIZER_PATH)

        lstm_path = MODELS_DIR / self.manifest["lstm_checkpoint"]
        gru_path = MODELS_DIR / self.manifest["gru_checkpoint"]
        for path, name in [(lstm_path, "LSTM"), (gru_path, "GRU")]:
            if not path.exists():
                raise CustomException(f"{name} checkpoint not found at {path}.", sys)

        self.lstm = DLTrainer.load(lstm_path)
        self.gru = DLTrainer.load(gru_path)

        logger.info(
            f"Loaded ensemble: seed={self.manifest['seed']}, "
            f"window_size={self.window_size}, regime_aware={self.regime_aware}."
        )

    def predict(self, raw_df: pd.DataFrame) -> pd.DataFrame:

        self._validate_input(raw_df)
        df = raw_df.copy()

        if self.regime_aware:
            df = self.regime_normalizer.transform(df)

        cycle_counts = df.groupby(ENGINE_COLUMN).size()

        last_windows = []
        engine_ids = []
        engines_with_short_history = []

        for engine_id, engine_df in df.groupby(ENGINE_COLUMN):

            engine_df = engine_df.sort_values(TIME_COLUMN)

            if len(engine_df) < self.window_size:
                engines_with_short_history.append(int(engine_id))
                continue

            scaled = pd.DataFrame(
                self.scaler.transform(engine_df[self.feature_columns]),
                columns=self.feature_columns,
            )
            last_windows.append(scaled.to_numpy()[-self.window_size:])
            engine_ids.append(engine_id)

        if not last_windows:
            raise CustomException(
                "No engine had enough cycles for a full "
                f"window_size={self.window_size} sequence -- every engine "
                f"needs at least {self.window_size} rows of history.", sys,
            )

        if engines_with_short_history:
            logger.info(
                f"{len(engines_with_short_history)} engine(s) skipped "
                f"(fewer than {self.window_size} cycles): {engines_with_short_history}"
            )

        X = np.asarray(last_windows, dtype=np.float32)

        lstm_preds = self.lstm.predict(X)
        gru_preds = self.gru.predict(X)
        predictions = (lstm_preds + gru_preds) / 2

        result = pd.DataFrame({
            ENGINE_COLUMN: engine_ids,
            "predicted_RUL": predictions,
            "n_cycles_seen": [cycle_counts[eid] for eid in engine_ids],
            "champion_version": self.version,
        })
        result["short_history_warning"] = False

        if engines_with_short_history:
            skipped = pd.DataFrame({
                ENGINE_COLUMN: engines_with_short_history,
                "predicted_RUL": np.nan,
                "n_cycles_seen": [cycle_counts[eid] for eid in engines_with_short_history],
                "champion_version": self.version,
                "short_history_warning": True,
            })
            result = pd.concat([result, skipped], ignore_index=True).sort_values(ENGINE_COLUMN).reset_index(drop=True)

        if self.regime_aware:
            last_regimes = df.sort_values([ENGINE_COLUMN, TIME_COLUMN]).groupby(ENGINE_COLUMN).tail(1)
            result = result.merge(
                last_regimes[[ENGINE_COLUMN, "regime"]], on=ENGINE_COLUMN, how="left",
            )

        return result

    def _validate_input(self, raw_df: pd.DataFrame) -> None:

        if raw_df.empty:
            raise CustomException("Input DataFrame is empty.", sys)

        missing = [c for c in REQUIRED_RAW_COLUMNS if c not in raw_df.columns]
        if missing:
            raise CustomException(f"Input is missing required columns: {missing}", sys)
