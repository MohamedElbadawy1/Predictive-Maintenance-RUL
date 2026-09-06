import sys
from pathlib import Path
from typing import List, Optional

import joblib
import json
import numpy as np
import pandas as pd

from src.config.config import (
    MODELS_DIR, SCALERS_DIR, SELECTED_FEATURES_PATH,
    ROLLING_WINDOW, LAGS, ENGINE_COLUMN,
)
from src.utils.constant import SENSOR_COLUMNS
from src.exceptions.custom_exception import CustomException
from src.preprocessing.feature_engineer import FeatureEngineer
from src.preprocessing.regime_normalizer import RegimeNormalizer
from src.explainability.feature_reducer import FeatureReducer
from src.models.base_trainer import BaseTrainer
from src.logger.logger import logger

TIME_COLUMN = "time_in_cycles"

REQUIRED_RAW_COLUMNS = (
    [ENGINE_COLUMN, TIME_COLUMN]
    + [f"operational_setting_{i}" for i in (1, 2, 3)]
    + SENSOR_COLUMNS
)


class InferencePipeline:
    """
    The single place "raw sensor readings in, RUL prediction out" lives.

    Everything this project's analysis scripts each re-implemented
    separately (regime normalization -> feature engineering -> last-cycle
    extraction -> scaling -> feature selection -> prediction) is wrapped
    here as one class with one method that matters: predict(). Every
    artifact (model, regime normalizer, scaler, feature list) is loaded
    ONCE at construction, not reloaded per call.

    Auto-detects whether the loaded model needs regime-aware
    normalization by reading the `requires_regime_normalizer` flag saved
    in best_params.json (see promote_regime_aware_model.py) — callers
    don't need to know or care which preprocessing path the current
    canonical model expects.

    Input contract: a DataFrame with one row per (engine, cycle), covering
    each engine's full cycle history up to "now" — the same shape as
    test_FD004. The prediction returned is for each engine's most recent
    cycle (its last row), matching the official evaluation protocol used
    throughout this project (Sprint 13 onward). An engine with fewer
    cycles than the longest lag (3) will still get a prediction — CatBoost
    handles the resulting NaNs natively — but is flagged in the output
    for visibility.

    Example
    -------
    pipeline = InferencePipeline()
    predictions_df = pipeline.predict(raw_engine_readings)
    # columns: unit_number, predicted_RUL, regime (if applicable), n_cycles_seen
    """

    def __init__(
        self,
        models_dir: Optional[Path] = None,
        scalers_dir: Optional[Path] = None,
        selected_features_path: Optional[Path] = None,
    ):

        self.models_dir = Path(models_dir) if models_dir else MODELS_DIR
        self.scalers_dir = Path(scalers_dir) if scalers_dir else SCALERS_DIR
        self.selected_features_path = (
            Path(selected_features_path) if selected_features_path else SELECTED_FEATURES_PATH
        )

        logger.info("Loading inference artifacts...")

        self.trainer = self._load_model()
        self.final_features = self._load_selected_features()
        self.scaler = self._load_scaler()
        self.requires_regime_normalization = self._check_requires_regime_normalization()
        self.regime_normalizer = self._load_regime_normalizer() if self.requires_regime_normalization else None

        self.feature_engineer = FeatureEngineer(
            sensor_columns=SENSOR_COLUMNS,
            group_column=ENGINE_COLUMN,
            rolling_window=ROLLING_WINDOW,
            lags=LAGS,
        )

        logger.info(
            f"InferencePipeline ready. Model expects {len(self.final_features)} features. "
            f"Regime normalization required: {self.requires_regime_normalization}."
        )

    def predict(self, raw_df: pd.DataFrame) -> pd.DataFrame:
        """
        raw_df: one row per (engine, cycle), same schema as test_FD004
        (unit_number, time_in_cycles, 3 operational settings, 21 sensors).
        Covers each engine's full history up to "now" — not just the
        latest cycle — since lag/rolling/diff features need prior cycles.

        Returns one row per engine: unit_number, predicted_RUL, n_cycles_seen,
        and regime (only if the loaded model requires regime normalization).
        """

        self._validate_input(raw_df)

        df = raw_df.copy()

        if self.requires_regime_normalization:
            df = self.regime_normalizer.transform(df)  # predict(), never fit, at inference time

        engineered = self.feature_engineer.transform(df)

        last_cycles = (
            engineered.sort_values([ENGINE_COLUMN, TIME_COLUMN])
            .groupby(ENGINE_COLUMN)
            .tail(1)
            .sort_values(ENGINE_COLUMN)
            .reset_index(drop=True)
        )

        cycle_counts = df.groupby(ENGINE_COLUMN).size()

        scaled = pd.DataFrame(
            self.scaler.transform(last_cycles[list(self.scaler.feature_names_in_)]),
            columns=self.scaler.feature_names_in_,
        )

        X = scaled[self.final_features]
        predictions = self.trainer.predict(X)

        result = pd.DataFrame({
            ENGINE_COLUMN: last_cycles[ENGINE_COLUMN].values,
            "predicted_RUL": predictions,
            "n_cycles_seen": last_cycles[ENGINE_COLUMN].map(cycle_counts).values,
        })

        min_cycles_needed = max(LAGS) + 1
        result["short_history_warning"] = result["n_cycles_seen"] < min_cycles_needed

        if self.requires_regime_normalization:
            result["regime"] = last_cycles["regime"].values

        if result["short_history_warning"].any():
            n_short = result["short_history_warning"].sum()
            logger.warning(
                f"{n_short} engine(s) have fewer than {min_cycles_needed} cycles of "
                "history — lag features are undefined for them (NaN). The model "
                "still produces a prediction, but treat it with reduced confidence."
            )

        return result

    def _validate_input(self, raw_df: pd.DataFrame) -> None:

        if raw_df.empty:
            raise CustomException("Input DataFrame is empty.", sys)

        missing = [c for c in REQUIRED_RAW_COLUMNS if c not in raw_df.columns]
        if missing:
            raise CustomException(f"Input is missing required columns: {missing}", sys)

        if raw_df[ENGINE_COLUMN].isna().any():
            raise CustomException(f"Null values found in '{ENGINE_COLUMN}'.", sys)

    def _load_model(self) -> BaseTrainer:

        model_path = self.models_dir / "best_model.pkl"
        if not model_path.exists():
            raise CustomException(f"No canonical model found at {model_path}.", sys)

        return BaseTrainer.load(model_path, track_mlflow=False)

    def _load_selected_features(self) -> List[str]:

        return FeatureReducer.load_selected_features(self.selected_features_path)

    def _load_scaler(self):

        scaler_path = self.scalers_dir / "feature_scaler.pkl"
        if not scaler_path.exists():
            raise CustomException(f"No feature scaler found at {scaler_path}.", sys)

        return joblib.load(scaler_path)

    def _check_requires_regime_normalization(self) -> bool:

        params_path = self.models_dir / "best_params.json"
        if not params_path.exists():
            return False

        with open(params_path) as f:
            record = json.load(f)

        return bool(record.get("requires_regime_normalizer", False))

    def _load_regime_normalizer(self) -> RegimeNormalizer:

        path = self.models_dir / "regime_normalizer.pkl"
        if not path.exists():
            raise CustomException(
                f"best_params.json says this model requires regime "
                f"normalization, but {path} was not found.",
                sys,
            )

        return RegimeNormalizer.load(path)