import time
from typing import List, Tuple

import pandas as pd

from src.evaluation.evaluator import RegressionEvaluator
from src.models.base_trainer import BaseTrainer
from src.models.model_factory import ModelFactory
from src.logger.logger import logger


class TraditionalMLBenchmark:
    """
    Benchmark multiple traditional machine learning regression models.
    """

    def __init__(self,models: List[str] | None = None,):

        self.models = (
            models
            if models is not None
            else ModelFactory.available_models()
        )

        self.evaluator = RegressionEvaluator()

    def run(
        self,
        X_train,
        y_train,
        X_val,
        y_val,
    ) -> Tuple[pd.DataFrame, BaseTrainer]:

        logger.info("Starting Traditional ML Benchmark...")

        results = []

        best_trainer = None
        best_score = float("-inf")

        for model_name in self.models:

            logger.info(f"Training {model_name}...")

            model = ModelFactory.create(model_name)
            trainer = BaseTrainer(model)
            start = time.time()

            trainer.train(
                X_train,
                y_train,
            )

            training_time = time.time() - start
            predictions = trainer.predict(X_val)
            metrics = self.evaluator.evaluate(
                y_val,
                predictions,
            )

            metrics["Model"] = model_name
            metrics["Training Time"] = training_time

            results.append(metrics)

            logger.info(
                f"{model_name} | "
                f"RMSE={metrics['RMSE']:.4f} | "
                f"R2={metrics['R2']:.4f}"
            )

            if metrics["R2"] > best_score:
                best_score = metrics["R2"]
                best_trainer = trainer

        results_df = (
            pd.DataFrame(results)
            .sort_values(
                by="R2",
                ascending=False,
            )
            .reset_index(drop=True)
        )

        logger.info("Benchmark completed successfully.")

        return results_df, best_trainer