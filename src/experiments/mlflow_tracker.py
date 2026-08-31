from pathlib import Path
from typing import Dict, List, Optional

import mlflow

from src.config.config import ARTIFACTS_DIR


class MLflowTracker:
    """
    One place for everything experiment-tracking related: setup and
    logging params/metrics/model/artifacts in a single call. Local
    SQLite backend — no tracking server to run or depend on.
    """

    def __init__(
        self,
        experiment_name: str = "predictive-maintenance-rul",
        tracking_dir: Optional[Path] = None,
    ):

        self.tracking_dir = tracking_dir or (ARTIFACTS_DIR / "mlruns")
        self.tracking_dir.mkdir(parents=True, exist_ok=True)

        mlflow.set_tracking_uri(f"sqlite:///{self.tracking_dir / 'mlflow.db'}")
        mlflow.set_experiment(experiment_name)

    def log_run(
        self,
        run_name: str,
        params: Optional[Dict] = None,
        metrics: Optional[Dict] = None,
        model=None,
        model_flavor: str = "catboost",
        artifact_paths: Optional[List[Path]] = None,
        tags: Optional[Dict] = None,
    ) -> str:

        with mlflow.start_run(run_name=run_name) as run:

            if tags:
                mlflow.set_tags(tags)

            if params:
                # MLflow rejects non-primitive param values (e.g. bool is
                # fine, but anything unhashable/complex isn't) — stringify
                # defensively so a training run never fails purely because
                # of a logging call.
                mlflow.log_params({k: str(v) for k, v in params.items()})

            if metrics:
                mlflow.log_metrics(self._sanitize_metrics(metrics))

            if model is not None:
                log_fn = getattr(mlflow, model_flavor, None)
                if log_fn is not None and hasattr(log_fn, "log_model"):
                    log_fn.log_model(model, "model")
                else:
                    # Unknown flavor — sklearn's logger works for any
                    # scikit-learn-compatible model (fit/predict interface),
                    # which covers RandomForest and anything else not
                    # explicitly handled above. Better than silently
                    # skipping the model artifact.
                    mlflow.sklearn.log_model(model, "model")

            if artifact_paths:
                for path in artifact_paths:
                    mlflow.log_artifact(str(path))

            return run.info.run_id

    def compare_runs(self, order_by: str = "metrics.MAE ASC"):

        return mlflow.search_runs(order_by=[order_by])

    @staticmethod
    def _sanitize_metrics(metrics: Dict) -> Dict:
        """
        MLflow metric names only allow alphanumerics, underscores, dashes,
        periods, spaces, colons, and slashes — e.g. "Training Time (s)"
        (a real key used elsewhere in this project) is rejected outright.
        Sanitize rather than let a logging call silently fail training.
        """

        import re

        clean = {}
        for key, value in metrics.items():
            if not isinstance(value, (int, float)):
                continue
            clean_key = re.sub(r"[^A-Za-z0-9_\-. :/]", "", key).strip().replace(" ", "_")
            clean[clean_key] = value

        return clean
