import json
import shutil
from datetime import datetime
from pathlib import Path

import pandas as pd

from src.logger.logger import logger
from src.config.config import EXPERIMENTS_DIR, MODELS_DIR


class ExperimentTracker:
    """
    Save experiment artifacts including metrics, model,
    feature importance, and experiment summary.

    Every experiment is stored in its own timestamped directory
    under EXPERIMENTS_DIR, while the latest best model is also
    copied to MODELS_DIR for easy loading.
    """

    def __init__(
        self,
        root_dir: Path = EXPERIMENTS_DIR,
        models_dir: Path = MODELS_DIR,
    ):

        self.root_dir = Path(root_dir)
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        timestamp = datetime.now().strftime(
            "%Y-%m-%d_%H-%M-%S"
        )

        self.experiment_dir = (
            self.root_dir / timestamp
        )

        self.experiment_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        logger.info(
            f"Experiment directory created at {self.experiment_dir}"
        )

    def save_results(
        self,
        results: pd.DataFrame,
    ):

        path = (
            self.experiment_dir /
            "benchmark_results.csv"
        )

        results.to_csv(
            path,
            index=False,
        )

        logger.info(
            "Benchmark results saved."
        )

    def save_best_model(
        self,
        trainer,
        model_name: str,
    ):

        # Save inside the experiment directory
        experiment_model_path = (
            self.experiment_dir /
            "best_model.pkl"
        )

        trainer.save(
            experiment_model_path
        )

        with open(
            self.experiment_dir /
            "best_model_name.txt",
            "w",
        ) as f:

            f.write(model_name)

        # Copy the latest model to the models directory
        latest_model_path = (
            self.models_dir /
            "best_model.pkl"
        )

        shutil.copy2(
            experiment_model_path,
            latest_model_path,
        )

        with open(
            self.models_dir /
            "best_model_name.txt",
            "w",
        ) as f:

            f.write(model_name)

        with open(
            self.models_dir /
            "latest_experiment.txt",
            "w",
        ) as f:

            f.write(
                self.experiment_dir.name
            )

        logger.info(
            "Best model saved."
        )

    def save_feature_importance(
        self,
        trainer,
        feature_names,
    ):

        model = trainer.model

        if not hasattr(
            model,
            "feature_importances_",
        ):

            logger.warning(
                "Model does not support feature importance."
            )

            return

        importance = pd.DataFrame(
            {
                "Feature": feature_names,
                "Importance": model.feature_importances_,
            }
        )

        importance = (
            importance
            .sort_values(
                by="Importance",
                ascending=False,
            )
        )

        importance.to_csv(
            self.experiment_dir /
            "feature_importance.csv",
            index=False,
        )

        logger.info(
            "Feature importance saved."
        )

    def save_summary(
        self,
        summary: dict,
    ):

        with open(
            self.experiment_dir /
            "summary.json",
            "w",
        ) as f:

            json.dump(
                summary,
                f,
                indent=4,
            )

        logger.info(
            "Summary saved."
        )