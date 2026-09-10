"""
TrainingPipeline: the callable, API-triggerable version of what
tune_and_ensemble_regime_aware.py does as a standalone script.

Why this exists as a class and not just "call the script": a FastAPI
endpoint can't cleanly shell out to a script and get a structured result
back. This wraps the same real logic (raw data -> regime normalization
-> feature engineering -> feature selection -> tuning -> evaluation) as
one method call that returns a structured result, suitable for an API
response.

MLflow Model Registry integration (alias-based "champion", not the
deprecated stages API): after tuning, the winning individual model
(ensembles are NOT registry-eligible yet — see the note in
_promote_if_better) is compared against the CURRENT CHAMPION's logged
test metric. Only promoted — registered as a new version, "champion"
alias moved, and materialized to the local files InferencePipeline
reads — if it's actually better. Nothing is overwritten blindly.

Bootstrapping (first-ever use, no champion set yet): see
docs/Sprint_20_Training_Pipeline_MLflow_Registry.md for the exact steps
to register your existing best real run as the initial champion.
"""
import shutil
import time
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import joblib
import pandas as pd

from src.config.config import (
    TRAIN_DATA_PATH, TEST_DATA_PATH, RUL_DATA_PATH,
    MODELS_DIR, SCALERS_DIR, SELECTED_FEATURES_PATH,
    VALIDATION_SIZE, RANDOM_STATE, DEFAULT_RUL_CAP,
    ROLLING_WINDOW, LAGS, ENGINE_COLUMN, TARGET_COLUMN,
)
from src.utils.constant import SENSOR_COLUMNS
from src.training.data_prep import prepare_training_data
from src.data.loader import DataLoader
from src.data.validator import DataValidator
from src.preprocessing.rul_generator import RULGenerator
from src.preprocessing.feature_engineer import FeatureEngineer
from src.preprocessing.data_splitter import DataSplitter
from src.preprocessing.feature_scaler import FeatureScaler
from src.preprocessing.regime_normalizer import RegimeNormalizer
from src.explainability.feature_selector import FeatureCategorySelector
from src.explainability.feature_reducer import FeatureReducer
from src.optimization.hyperparameter_tuner import ModelTuner
from src.models.model_factory import ModelFactory
from src.models.base_trainer import BaseTrainer
from src.evaluation.evaluator import RegressionEvaluator
from src.experiments.mlflow_tracker import MLflowTracker
from src.logger.logger import logger

REGISTRY_MODEL_NAME = "predictive-maintenance-rul-model"


@dataclass
class TrainingResult:
    """Structured, API-response-friendly result of a training run."""

    winning_model_name: str
    winning_test_metrics: Dict[str, float]
    all_model_results: List[Dict]
    promoted: bool
    promotion_reason: str
    champion_metric_before: Optional[float]
    champion_metric_after: float
    run_id: str
    registered_version: Optional[str] = None
    elapsed_seconds: float = field(default=0.0)


class TrainingPipeline:
    """
    Example
    -------
    pipeline = TrainingPipeline()
    result = pipeline.run(n_trials=20)

    if result.promoted:
        print(f"New champion: {result.winning_model_name}, "
              f"MAE {result.champion_metric_after:.3f} "
              f"(was {result.champion_metric_before})")
    """

    def __init__(
        self,
        registry_model_name: str = REGISTRY_MODEL_NAME,
        promotion_metric: str = "MAE",
        lower_is_better: bool = True,
        n_regimes: int = 6,
    ):

        self.registry_model_name = registry_model_name
        self.promotion_metric = promotion_metric
        self.lower_is_better = lower_is_better
        self.n_regimes = n_regimes

        self.tracker = MLflowTracker()
        self.evaluator = RegressionEvaluator()

    def run(
        self,
        n_trials: int = 20,
        model_names: Sequence[str] = ("catboost", "xgboost", "lightgbm"),
        regime_aware: bool = True,
    ) -> TrainingResult:

        start = time.time()

        data = prepare_training_data(regime_aware=regime_aware, n_regimes=self.n_regimes)
        X_train, y_train = data["X_train"], data["y_train"]
        X_val, y_val = data["X_val"], data["y_val"]
        X_test, y_test_true = data["X_test"], data["y_test_true"]
        regime_normalizer, scaler, final_features = data["regime_normalizer"], data["scaler"], data["final_features"]

        all_results = []
        trained_models = {}

        for model_name in model_names:

            logger.info(f"[TrainingPipeline] Tuning {model_name} ({n_trials} trials)...")
            tuner = ModelTuner(model_name, X_train, y_train, X_val, y_val, random_state=RANDOM_STATE)
            tuner.run(n_trials=n_trials, show_progress_bar=False)
            best_params = tuner.best_params()

            model = ModelFactory.create(model_name, **best_params)
            trainer = BaseTrainer(
                model,
                run_name=f"{model_name}_training_pipeline",
                tags={"model_family": model_name, "stage": "training_pipeline_final",
                      "normalization": "regime_aware" if regime_aware else "global"},
            )
            val_metrics = trainer.train(X_train, y_train, X_val, y_val)

            test_preds = trainer.predict(X_test)
            test_metrics = self.evaluator.evaluate(y_test_true, test_preds)

            trained_models[model_name] = trainer
            all_results.append({
                "model": model_name,
                "run_id": trainer.last_run_id,
                "val_metrics": val_metrics,
                "test_metrics": test_metrics,
            })

        # Pick the winner by TEST metric — validation-only comparison is
        # not trustworthy across configs (see Sprint 14's finding).
        winner = min(
            all_results,
            key=lambda r: r["test_metrics"][self.promotion_metric] if self.lower_is_better
            else -r["test_metrics"][self.promotion_metric],
        )

        result = self._promote_if_better(
            winner=winner,
            trainer=trained_models[winner["model"]],
            regime_normalizer=regime_normalizer if regime_aware else None,
            scaler=scaler,
            final_features=final_features,
            all_results=all_results,
        )

        result.elapsed_seconds = round(time.time() - start, 1)
        return result

    def run_fixed_params(
        self,
        model_name: str,
        params: Dict,
        regime_aware: bool = True,
    ) -> TrainingResult:
        """
        Same pipeline as run(), but trains ONE model with an already-known
        hyperparameter configuration instead of running an Optuna search —
        for periodic retraining on fresh data where you don't want to pay
        the cost of a full search every time, only confirm the existing
        best-known configuration is still good on current data.
        """

        start = time.time()

        data = prepare_training_data(regime_aware=regime_aware, n_regimes=self.n_regimes)
        X_train, y_train = data["X_train"], data["y_train"]
        X_val, y_val = data["X_val"], data["y_val"]
        X_test, y_test_true = data["X_test"], data["y_test_true"]
        regime_normalizer, scaler, final_features = data["regime_normalizer"], data["scaler"], data["final_features"]

        logger.info(f"[TrainingPipeline] Training {model_name} with fixed params (no search): {params}")

        model = ModelFactory.create(model_name, **params)
        trainer = BaseTrainer(
            model,
            run_name=f"{model_name}_fixed_params_retrain",
            tags={"model_family": model_name, "stage": "fixed_params_retrain",
                  "normalization": "regime_aware" if regime_aware else "global"},
        )
        val_metrics = trainer.train(X_train, y_train, X_val, y_val)

        test_preds = trainer.predict(X_test)
        test_metrics = self.evaluator.evaluate(y_test_true, test_preds)

        winner = {
            "model": model_name,
            "run_id": trainer.last_run_id,
            "val_metrics": val_metrics,
            "test_metrics": test_metrics,
        }

        result = self._promote_if_better(
            winner=winner,
            trainer=trainer,
            regime_normalizer=regime_normalizer if regime_aware else None,
            scaler=scaler,
            final_features=final_features,
            all_results=[winner],
        )

        result.elapsed_seconds = round(time.time() - start, 1)
        return result

    def _promote_if_better(self, winner, trainer, regime_normalizer, scaler, final_features, all_results) -> TrainingResult:

        winner_metric = winner["test_metrics"][self.promotion_metric]
        champion_metric = self.tracker.get_champion_metric(self.registry_model_name, f"test_{self.promotion_metric}")

        if champion_metric is None:
            should_promote = True
            reason = "No champion exists yet — promoting unconditionally (bootstrap)."
        elif self.lower_is_better:
            should_promote = winner_metric < champion_metric
            reason = (f"New {self.promotion_metric} {winner_metric:.4f} < champion's {champion_metric:.4f}"
                       if should_promote else
                       f"New {self.promotion_metric} {winner_metric:.4f} did not beat champion's {champion_metric:.4f}")
        else:
            should_promote = winner_metric > champion_metric
            reason = (f"New {self.promotion_metric} {winner_metric:.4f} > champion's {champion_metric:.4f}"
                       if should_promote else
                       f"New {self.promotion_metric} {winner_metric:.4f} did not beat champion's {champion_metric:.4f}")

        registered_version = None

        if should_promote:

            logger.info(f"[TrainingPipeline] Promoting {winner['model']}: {reason}")

            # Log the winner's test metrics AND its preprocessing artifacts
            # (scaler, regime normalizer, feature list) onto its own run
            # before registering — this is what makes pipeline/predict.py's
            # "load everything from MLflow" approach possible: the run
            # becomes fully self-contained, not just the model.
            import mlflow
            import json
            import tempfile

            with mlflow.start_run(run_id=winner["run_id"]):
                mlflow.log_metrics({f"test_{k}": v for k, v in winner["test_metrics"].items()
                                     if isinstance(v, (int, float))})

                with tempfile.TemporaryDirectory() as tmp_dir:
                    tmp_dir = Path(tmp_dir)

                    raw_scaler = scaler.scaler if hasattr(scaler, "scaler") else scaler
                    joblib.dump(raw_scaler, tmp_dir / "feature_scaler.pkl")
                    mlflow.log_artifact(str(tmp_dir / "feature_scaler.pkl"), artifact_path="preprocessing")

                    with open(tmp_dir / "selected_features.json", "w") as f:
                        json.dump({"selected_features": final_features}, f, indent=2)
                    mlflow.log_artifact(str(tmp_dir / "selected_features.json"), artifact_path="preprocessing")

                    requires_regime = regime_normalizer is not None
                    if requires_regime:
                        regime_normalizer.save(tmp_dir / "regime_normalizer.pkl")
                        mlflow.log_artifact(str(tmp_dir / "regime_normalizer.pkl"), artifact_path="preprocessing")

                    mlflow.set_tag("requires_regime_normalizer", str(requires_regime))

            model_version = self.tracker.register_model(winner["run_id"], self.registry_model_name)
            self.tracker.set_champion(self.registry_model_name, model_version.version)
            registered_version = model_version.version

            self._materialize_to_canonical_files(trainer, regime_normalizer, scaler, final_features, winner)

        else:
            logger.info(f"[TrainingPipeline] Not promoting {winner['model']}: {reason}")

        return TrainingResult(
            winning_model_name=winner["model"],
            winning_test_metrics=winner["test_metrics"],
            all_model_results=all_results,
            promoted=should_promote,
            promotion_reason=reason,
            champion_metric_before=champion_metric,
            champion_metric_after=winner_metric if should_promote else champion_metric,
            run_id=winner["run_id"],
            registered_version=registered_version,
        )

    def _materialize_to_canonical_files(self, trainer, regime_normalizer, scaler, final_features, winner):
        """
        The MLflow registry is the audit trail; InferencePipeline reads
        local files for simplicity and speed at serving time. This keeps
        both in sync on every promotion.
        """

        import json

        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        SCALERS_DIR.mkdir(parents=True, exist_ok=True)

        trainer.save(MODELS_DIR / "best_model.pkl")

        reducer = FeatureReducer(keep_features=final_features)
        # keep_features-mode fit() only needs the columns, not real data
        reducer.selected_features_ = final_features
        reducer.removed_features_ = []
        reducer.save_selected_features(SELECTED_FEATURES_PATH)

        joblib.dump(scaler.scaler if hasattr(scaler, "scaler") else scaler, SCALERS_DIR / "feature_scaler.pkl")

        requires_regime = regime_normalizer is not None
        if requires_regime:
            regime_normalizer.save(MODELS_DIR / "regime_normalizer.pkl")

        with open(MODELS_DIR / "best_model_name.txt", "w") as f:
            f.write(f"{winner['model']}_training_pipeline")

        with open(MODELS_DIR / "best_params.json", "w") as f:
            json.dump({
                "params": self._get_model_params(trainer),
                "metrics": winner["test_metrics"],
                "requires_regime_normalizer": requires_regime,
                "registry_model_name": self.registry_model_name,
                "run_id": winner["run_id"],
                "note": "Promoted by TrainingPipeline after beating the MLflow-tracked champion.",
            }, f, indent=2)

    @staticmethod
    def _get_model_params(trainer) -> Dict:

        model = trainer.model
        return model.get_params() if hasattr(model, "get_params") else {}
