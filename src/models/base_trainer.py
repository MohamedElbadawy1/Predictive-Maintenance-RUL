import time
from pathlib import Path
from typing import Dict, Optional

import joblib
import pandas as pd

from src.evaluation.evaluator import RegressionEvaluator
from src.logger.logger import logger


class BaseTrainer:
    """
    Generic trainer for any scikit-learn compatible regression model —
    with MLflow tracking built in, not bolted on. Every train() call logs
    its own params, training time, and (if validation data is given)
    validation metrics and the model itself automatically. No separate
    manual logging call needed for a normal training run.

    Tracking is on by default and fails soft: if MLflow logging has a
    problem, training still completes and a warning is logged rather
    than the run being lost.

    Example
    -------
    trainer = BaseTrainer(model, run_name="catboost_tuned_v1",
                           tags={"feature_set": "109", "rul_cap": "150"})
    metrics = trainer.train(X_train, y_train, X_val, y_val)
    trainer.save(MODELS_DIR / "best_model.pkl")
    """

    def __init__(
        self,
        model,
        track_mlflow: bool = True,
        run_name: Optional[str] = None,
        tags: Optional[Dict] = None,
    ):

        self.model = model
        self.track_mlflow = track_mlflow
        self.run_name = run_name or f"{model.__class__.__name__}_run"
        self.tags = tags
        self.last_run_id = None

        self.evaluator = RegressionEvaluator()

        self.tracker = None
        if self.track_mlflow:
            try:
                from src.experiments.mlflow_tracker import MLflowTracker
                self.tracker = MLflowTracker()
            except Exception as exc:
                logger.warning(
                    f"MLflow tracking unavailable ({exc}) — training will "
                    "proceed without it."
                )
                self.track_mlflow = False

    def train(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[pd.Series] = None,
    ) -> Dict:
        """
        Fits the model. If X_val/y_val are provided, also evaluates on
        them and returns the metrics dict — and, if tracking is on, logs
        params + training time + these metrics + the model to MLflow in
        the same call. Metrics dict is empty if no validation data given.
        """

        logger.info(f"Training {self.model.__class__.__name__}...")

        start = time.time()
        self.model.fit(X_train, y_train)
        training_time = time.time() - start

        logger.info(f"Training completed successfully in {training_time:.2f}s.")

        metrics = {}
        if X_val is not None and y_val is not None:
            predictions = self.predict(X_val)
            metrics = self.evaluator.evaluate(y_val, predictions)
            metrics["Training Time (s)"] = round(training_time, 2)

        if self.track_mlflow and self.tracker is not None:
            try:
                params = self._get_model_params()
                self.last_run_id = self.tracker.log_run(
                    run_name=self.run_name,
                    params=params,
                    metrics=metrics,
                    model=self.model,
                    model_flavor=self._detect_mlflow_flavor(),
                    tags=self.tags,
                )
                logger.info(f"Logged to MLflow: run_id={self.last_run_id}")
            except Exception as exc:
                logger.warning(f"MLflow logging failed ({exc}) — model was still trained.")

        return metrics

    def _detect_mlflow_flavor(self) -> str:
        """
        MLflow logs models differently per library (mlflow.catboost vs
        mlflow.xgboost vs mlflow.lightgbm vs generic mlflow.sklearn) —
        detect the right one from the model's module rather than assume
        every trained model is CatBoost.
        """

        module_name = type(self.model).__module__.lower()

        if "catboost" in module_name:
            return "catboost"
        if "xgboost" in module_name:
            return "xgboost"
        if "lightgbm" in module_name:
            return "lightgbm"

        return "sklearn"

    def predict(
        self,
        X: pd.DataFrame,
    ):

        logger.info(f"Generating predictions using {self.model.__class__.__name__}...")

        return self.model.predict(X)

    def save(
        self,
        path: str,
    ) -> None:

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        joblib.dump(self.model, path)

        logger.info(f"Model saved to {path}")

    @staticmethod
    def load(path: str, track_mlflow: bool = False) -> "BaseTrainer":

        logger.info(f"Loading model from {path}")

        model = joblib.load(path)

        # Loading is inference, not a training run — tracking defaults
        # off here so reloading a model for prediction doesn't create an
        # empty/misleading MLflow run.
        return BaseTrainer(model, track_mlflow=track_mlflow)

    def _get_model_params(self) -> Dict:

        if hasattr(self.model, "get_params"):
            return self.model.get_params()

        return {}
