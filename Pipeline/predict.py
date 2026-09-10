"""
pipeline/predict.py — use the champion model directly from MLflow.

Unlike src/inference/pipeline.py (InferencePipeline, Phase 1), which
reads the model and preprocessing artifacts from local files, this loads
EVERYTHING from MLflow: the model via the registry's "champion" alias,
and the preprocessing artifacts (regime normalizer, scaler, feature
list) from that same run's logged "preprocessing/" artifacts —
TrainingPipeline logs those onto every promoted run (see
src/training/pipeline.py's _promote_if_better).

No dependency on local files staying in sync with the registry — if the
registry says a run is champion, this can serve predictions from it
directly, from any machine with access to the same MLflow tracking
store.

Usage:
    python pipeline/predict.py --input path/to/raw_engine_data.csv
"""
import argparse
import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, "/home/claude/Predictive-Maintenance-RUL")

import joblib
import mlflow
import numpy as np
import pandas as pd

from src.config.config import ROLLING_WINDOW, LAGS, ENGINE_COLUMN
from src.utils.constant import SENSOR_COLUMNS
from src.exceptions.custom_exception import CustomException
from src.preprocessing.feature_engineer import FeatureEngineer
from src.preprocessing.regime_normalizer import RegimeNormalizer
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
    Loads the current champion model AND its preprocessing artifacts
    entirely from MLflow — nothing read from local files.

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


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="CSV of raw engine readings")
    parser.add_argument("--registry-model-name", default=REGISTRY_MODEL_NAME)
    args = parser.parse_args()

    raw_df = pd.read_csv(args.input)

    pipeline = MLflowInferencePipeline(registry_model_name=args.registry_model_name)
    predictions = pipeline.predict(raw_df)

    print(predictions.to_string(index=False))
