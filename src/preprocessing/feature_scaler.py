from pathlib import Path

import joblib
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.preprocessing import StandardScaler

from src.exceptions.custom_exception import CustomException
from src.logger.logger import logger

import sys


class FeatureScaler:
    """
    Scale numerical features using a scikit-learn scaler.

    Default scaler: StandardScaler
    """

    def __init__(
        self,
        scaler: BaseEstimator | None = None,
    ):
        self.scaler = scaler or StandardScaler()
        self._is_fitted = False

    def fit(self, X: pd.DataFrame) -> None:
        """
        Fit the scaler on training data only.
        """

        logger.info("Fitting Feature Scaler...")
        self._validate_input(X)
        self.scaler.fit(X)
        self._is_fitted = True
        logger.info("Feature Scaler fitted successfully.")

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transform features using the fitted scaler.
        """

        logger.info("Transforming features...")

        self._validate_input(X)

        if not self._is_fitted:
            raise CustomException(
                "Scaler has not been fitted.",
                sys,
            )

        X_scaled = pd.DataFrame(
            self.scaler.transform(X),
            columns=X.columns,
            index=X.index,
        )

        logger.info("Feature transformation completed.")

        return X_scaled

    def fit_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Fit then transform the training data.
        """

        self.fit(X)

        return self.transform(X)

    def save(self, save_path: str | Path) -> None:
        """
        Save fitted scaler.
        """

        if not self._is_fitted:
            raise CustomException(
                "Cannot save an unfitted scaler.",
                sys,
            )

        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        joblib.dump(self.scaler, save_path)

        logger.info(f"Scaler saved to {save_path}")

    def _validate_input(self, X: pd.DataFrame) -> None:

        if X.empty:
            raise CustomException(
                "Input DataFrame is empty.",
                sys,
            )

        if not isinstance(X, pd.DataFrame):
            raise CustomException(
                "Input must be a pandas DataFrame.",
                sys,
            )