from typing import List, Tuple

import pandas as pd

from src.exceptions.custom_exception import CustomException
from src.logger.logger import logger
import sys


class FeatureSelector:
    """
    Separate features (X) and target (y) for model training.
    """

    def __init__(
        self,
        target_column: str = "RUL",
        drop_columns: List[str] | None = None,
    ):

        self.target_column = target_column
        self.drop_columns = drop_columns or []

    def transform(
        self,
        df: pd.DataFrame,
    ) -> Tuple[pd.DataFrame, pd.Series]:

        logger.info("Starting Feature Selection...")

        self._validate_input(df)

        data = df.copy()
        y = data[self.target_column].copy()

        X = data.drop(
            columns=[self.target_column] + self.drop_columns
        )

        logger.info("Feature Selection completed successfully.")

        return X, y

    def _validate_input(
        self,
        df: pd.DataFrame,
    ) -> None:

        if df.empty:
            raise CustomException(
                "Input DataFrame is empty.",
                sys,
            )

        if self.target_column not in df.columns:
            raise CustomException(
                f"Target column '{self.target_column}' not found.",
                sys,
            )

        if self.target_column in self.drop_columns:
            raise CustomException(
                "Target column cannot be included in drop_columns.",
                sys,
            )

        missing = [
            col for col in self.drop_columns
            if col not in df.columns
        ]

        if missing:
            raise CustomException(
                f"Missing columns: {missing}",
                sys,
            )

        if df[self.target_column].isna().any():
            raise CustomException(
                "Target column contains missing values.",
                sys,
            )