"""
src/pipelines/inference_pipeline.py — MLflowInferencePipeline: loads
the current champion model AND its preprocessing artifacts (regime
normalizer, scaler, feature list) entirely from MLflow — nothing read
from local files. TrainingPipeline logs those preprocessing artifacts
onto every promoted run (see src/training/pipeline.py's
_promote_if_better), so this can serve predictions from any machine
with access to the same MLflow tracking store.

This used to live inside Pipeline/predict.py and get imported from
api/dependencies.py via a cross-folder import (`from pipeline.predict
import ...` / `from Pipeline.predict import ...`) that broke depending
on the checked-out folder's exact case (works differently across
Windows, macOS, and Linux, and inside a Linux-based Docker container
specifically — where this failed outright). Moving the class here
means both Pipeline/predict.py (the CLI) and api/dependencies.py (the
FastAPI app) import from the same stable, lowercase src/ path, on any
OS, with no ambiguity.
"""
import json
import sys
import tempfile
from pathlib import Path

import joblib
import mlflow
import numpy as np
import pandas as pd

from src.config.config import ROLLING_WINDOW, LAGS, ENGINE_COLUMN
from src.utils.constant import SENSOR_COLUMNS
from src.exceptions.custom_exception import CustomException
from src.preprocessing.feature_engineer import FeatureEngineer
from src.experiments.mlflow_tracker import MLflowTracker
from src.logger.logger import logger

TIME_COLUMN = "time_in_cycles"
REGISTRY_MODEL_NAME = "predictive-maintenance-rul-model"

REQUIRED_RAW_COLUMNS = (
    [ENGINE_COLUMN, TIME_COLUMN]
    + [f"operational_setting_{i}" for i in (1, 2, 3)]
    + SENSOR_COLUMNS
)


class MLflowInferencePipeline:
    """
    Example
    -------
    pipeline = MLflowInferencePipeline()
    predictions = pipeline.predict(raw_engine_readings)
    """

    def __init__(self, registry_model_name: str = REGISTRY_MODEL_NAME):

        self.registry_model_name = registry_model_name
        self.tracker = MLflowTracker()

        logger.info(f"Loading champion for '{registry_model_name}' from MLflow...")

        champion_info = self.tracker.get_champion_run_info(registry_model_name)
        if champion_info is None:
            raise CustomException(
                f"No champion set for '{registry_model_name}'. Run a training "
                "pipeline first, or see docs/Sprint_20_Training_Pipeline_MLflow_Registry.md "
                "to bootstrap an existing run as champion.",
                sys,
            )

        self.run_id = champion_info["run_id"]
        self.version = champion_info["version"]
        self.requires_regime_normalization = (
            mlflow.get_run(self.run_id).data.tags.get("requires_regime_normalizer") == "True"
        )

        self.model = mlflow.pyfunc.load_model(f"models:/{registry_model_name}@champion")

        self.final_features, self.scaler, self.regime_normalizer = self._load_preprocessing_artifacts()

        self.feature_engineer = FeatureEngineer(
            sensor_columns=SENSOR_COLUMNS, group_column=ENGINE_COLUMN,
            rolling_window=ROLLING_WINDOW, lags=LAGS,
        )

        logger.info(
            f"Loaded champion: version {self.version}, run {self.run_id}. "
            f"Regime normalization required: {self.requires_regime_normalization}."
        )

    def predict(self, raw_df: pd.DataFrame) -> pd.DataFrame:

        self._validate_input(raw_df)
        df = raw_df.copy()

        if self.requires_regime_normalization:
            df = self.regime_normalizer.transform(df)

        engineered = self.feature_engineer.transform(df)

        last_cycles = (
            engineered.sort_values([ENGINE_COLUMN, TIME_COLUMN])
            .groupby(ENGINE_COLUMN).tail(1)
            .sort_values(ENGINE_COLUMN).reset_index(drop=True)
        )

        cycle_counts = df.groupby(ENGINE_COLUMN).size()

        scaled = pd.DataFrame(
            self.scaler.transform(last_cycles[list(self.scaler.feature_names_in_)]),
            columns=self.scaler.feature_names_in_,
        )
        X = scaled[self.final_features]

        predictions = self.model.predict(X)
        predictions = np.asarray(predictions).flatten()

        result = pd.DataFrame({
            ENGINE_COLUMN: last_cycles[ENGINE_COLUMN].values,
            "predicted_RUL": predictions,
            "n_cycles_seen": last_cycles[ENGINE_COLUMN].map(cycle_counts).values,
            "champion_version": self.version,
        })

        min_cycles_needed = max(LAGS) + 1
        result["short_history_warning"] = result["n_cycles_seen"] < min_cycles_needed

        if self.requires_regime_normalization:
            result["regime"] = last_cycles["regime"].values

        return result

    def _load_preprocessing_artifacts(self):

        with tempfile.TemporaryDirectory() as tmp_dir:

            local_dir = mlflow.artifacts.download_artifacts(
                run_id=self.run_id, artifact_path="preprocessing", dst_path=tmp_dir,
            )
            local_dir = Path(local_dir)

            with open(local_dir / "selected_features.json") as f:
                final_features = json.load(f)["selected_features"]

            scaler = joblib.load(local_dir / "feature_scaler.pkl")

            regime_normalizer = None
            if (local_dir / "regime_normalizer.pkl").exists():
                regime_normalizer = joblib.load(local_dir / "regime_normalizer.pkl")

        return final_features, scaler, regime_normalizer

    def _validate_input(self, raw_df: pd.DataFrame) -> None:

        if raw_df.empty:
            raise CustomException("Input DataFrame is empty.", sys)

        missing = [c for c in REQUIRED_RAW_COLUMNS if c not in raw_df.columns]
        if missing:
            raise CustomException(f"Input is missing required columns: {missing}", sys)
