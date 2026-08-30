from pathlib import Path

import joblib
import pandas as pd

from src.logger.logger import logger


class BaseTrainer:
    """
    Generic trainer for any scikit-learn compatible regression model.
    """

    def __init__(self, model):

        self.model = model

    def train(self,X_train: pd.DataFrame,y_train: pd.Series,) -> None:

        logger.info(
            f"Training {self.model.__class__.__name__}..."
        )

        self.model.fit(X_train, y_train)

        logger.info("Training completed successfully.")

    def predict(
        self,
        X: pd.DataFrame,
    ):

        logger.info(
            f"Generating predictions using {self.model.__class__.__name__}..."
        )

        return self.model.predict(X)

    def save(self,path: str,) -> None:

        path = Path(path)
        path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        joblib.dump(
            self.model,
            path,
        )

        logger.info(
            f"Model saved to {path}"
        )

    @staticmethod
    def load(path: str):

        logger.info(
            f"Loading model from {path}"
        )

        return joblib.load(path)