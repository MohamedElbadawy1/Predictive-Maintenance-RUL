from __future__ import annotations

import sys
from typing import Dict

import numpy as np
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)

from src.exceptions.custom_exception import CustomException
from src.logger.logger import logger


class RegressionEvaluator:
    """
    Evaluate regression models using common regression metrics.

    Supported Metrics
    -----------------
    - MAE
    - RMSE
    - R²
    - MAPE
    """

    def evaluate(self,y_true,y_pred,) -> Dict[str, float]:
        """
        Compute regression metrics.

        Parameters
        ----------
        y_true : array-like
            Ground truth values.

        y_pred : array-like
            Predicted values.

        Returns
        -------
        Dict[str, float]
            Dictionary containing regression metrics.
        """

        logger.info("Starting regression evaluation...")

        self._validate_inputs(y_true, y_pred)

        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)

        mae = mean_absolute_error(y_true, y_pred)

        rmse = np.sqrt(
            mean_squared_error(
                y_true,
                y_pred,
            )
        )

        r2 = r2_score(
            y_true,
            y_pred,
        )

        mape = self._mean_absolute_percentage_error(
            y_true,
            y_pred,
        )

        metrics = {
            "MAE": float(mae),
            "RMSE": float(rmse),
            "R2": float(r2),
            "MAPE": float(mape),
        }

        logger.info("Evaluation completed successfully.")

        return metrics

    @staticmethod
    def _mean_absolute_percentage_error(
        y_true: np.ndarray,
        y_pred: np.ndarray,
    ) -> float:
        """
        Compute Mean Absolute Percentage Error (MAPE).

        Samples with y_true == 0 are ignored to avoid division by zero.
        """

        mask = y_true != 0

        if mask.sum() == 0:
            return np.nan

        return (
            np.mean(
                np.abs(
                    (y_true[mask] - y_pred[mask])
                    / y_true[mask]
                )
            )
            * 100
        )

    @staticmethod
    def _validate_inputs(
        y_true,
        y_pred,
    ) -> None:

        if y_true is None or y_pred is None:
            raise CustomException(
                "Inputs cannot be None.",
                sys,
            )

        if len(y_true) == 0:
            raise CustomException(
                "Ground truth array is empty.",
                sys,
            )

        if len(y_pred) == 0:
            raise CustomException(
                "Prediction array is empty.",
                sys,
            )

        if len(y_true) != len(y_pred):
            raise CustomException(
                "y_true and y_pred must have the same length.",
                sys,
            )