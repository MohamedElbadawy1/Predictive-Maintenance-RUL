import time
from typing import Dict, Optional

import pandas as pd

from src.evaluation.evaluator import RegressionEvaluator
from src.models.base_trainer import BaseTrainer
from src.models.model_factory import ModelFactory
from src.logger.logger import logger


class ExperimentRunner:
    """
    Train and evaluate the same model type across several feature
    subsets, so different feature-reduction strategies can be
    compared on equal footing (same model, same params, same split).

    Every run is recorded as a row: experiment name, feature count,
    MAE, RMSE, R2, and training time.

    Example
    -------
    runner = ExperimentRunner(model_name="catboost")

    runner.run(
        "Baseline",
        X_train, y_train, X_val, y_val,
    )

    runner.run(
        "Remove Zero Importance",
        X_train[reduced_features], y_train,
        X_val[reduced_features], y_val,
    )

    results_df = runner.get_results()
    """

    def __init__(
        self,
        model_name: str = "catboost",
        model_params: Optional[Dict] = None,
    ):

        self.model_name = model_name
        self.model_params = model_params or {}
        self.evaluator = RegressionEvaluator()

        self.results = []
        self.trainers: Dict[str, BaseTrainer] = {}

    def run(
        self,
        experiment_name: str,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series,
    ) -> BaseTrainer:

        logger.info(
            f"Running experiment '{experiment_name}' "
            f"with {X_train.shape[1]} features..."
        )

        model = ModelFactory.create(
            self.model_name,
            **self.model_params,
        )
        trainer = BaseTrainer(model)

        start = time.time()
        trainer.train(X_train, y_train)
        training_time = time.time() - start

        predictions = trainer.predict(X_val)
        metrics = self.evaluator.evaluate(y_val, predictions)

        row = {
            "Experiment": experiment_name,
            "Features": X_train.shape[1],
            "MAE": metrics["MAE"],
            "RMSE": metrics["RMSE"],
            "R2": metrics["R2"],
            "Training Time (s)": round(training_time, 2),
        }

        self.results.append(row)
        self.trainers[experiment_name] = trainer

        logger.info(
            f"{experiment_name} | Features={row['Features']} | "
            f"MAE={row['MAE']:.4f} | RMSE={row['RMSE']:.4f} | "
            f"R2={row['R2']:.4f} | Time={row['Training Time (s)']}s"
        )

        return trainer

    def get_results(self) -> pd.DataFrame:

        return pd.DataFrame(self.results)

    def get_trainer(self, experiment_name: str) -> BaseTrainer:

        return self.trainers[experiment_name]

    def best_experiment(
        self,
        metric: str = "MAE",
        minimize: bool = True,
    ) -> pd.Series:

        results_df = self.get_results()

        return (
            results_df
            .sort_values(metric, ascending=minimize)
            .iloc[0]
        )
