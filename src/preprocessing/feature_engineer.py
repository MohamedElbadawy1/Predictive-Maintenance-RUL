from __future__ import annotations

from typing import List
import sys
import pandas as pd

from src.exceptions.custom_exception import CustomException
from src.logger.logger import logger


class FeatureEngineer:
    """
    Feature engineering pipeline for traditional machine learning models.

    This class is responsible for creating time-series features
    independently for each engine.

    Supported Features
    ------------------
    - Rolling Mean
    - Rolling Standard Deviation
    - Lag Features
    - Rate of Change (Difference)

    Notes
    -----
    - The original DataFrame is never modified.
    - NaN values generated during feature creation are intentionally preserved.
    - All temporal features are computed independently for each engine.
    """

    def __init__(
        self,
        sensor_columns: List[str],
        group_column: str = "unit_number",
        rolling_window: int = 5,
        lags: List[int] | None = None,) -> None:

        self.sensor_columns = sensor_columns
        self.group_column = group_column
        self.rolling_window = rolling_window
        self.lags = lags if lags is not None else [1]


    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply all feature engineering transformations.
        """

        logger.info("Starting Feature Engineering...")

        self._validate_input(df)

        engineered_df = df.copy()

        engineered_df = self._rolling_mean(engineered_df)
        engineered_df = self._rolling_std(engineered_df)
        engineered_df = self._lag_features(engineered_df)
        engineered_df = self._rate_of_change(engineered_df)

        logger.info("Feature Engineering completed successfully.")

        return engineered_df


    ## Validate Input
    def _validate_input(self, df: pd.DataFrame) -> None:
        """
        Validate input dataframe and configuration.
        """
        if df.empty:
            raise CustomException("Input DataFrame is empty.",sys)

        if self.group_column not in df.columns:
            raise CustomException(
                f"Group column '{self.group_column}' not found.",sys
            )

        missing_columns = [
            col for col in self.sensor_columns
            if col not in df.columns
        ]

        if missing_columns:
            raise CustomException(
                f"Missing sensor columns: {missing_columns}",sys
            )

        if self.rolling_window <= 0:
            raise CustomException(
                "Rolling window must be greater than zero.",sys
            )

        if not self.lags:
            raise CustomException(
                "Lag list cannot be empty.",sys
            )

        for lag in self.lags:

            if not isinstance(lag, int):
                raise CustomException(
                    f"Lag '{lag}' is not an integer.",sys
                )

            if lag <= 0:
                raise CustomException(
                    "Lag values must be greater than zero.",sys
                )


    # Rolling Mean

    def _rolling_mean(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate rolling mean features.
        """

        logger.info("Generating Rolling Mean features...")

        grouped = df.groupby(self.group_column)

        for sensor in self.sensor_columns:
            feature_name = (
                f"{sensor}_roll_mean_{self.rolling_window}"
            )

            df[feature_name] = (
                grouped[sensor]
                .transform(
                    lambda x: x.rolling(
                        window=self.rolling_window,
                        min_periods=1,
                    ).mean()
                )
            )

        return df


    # Rolling Std

    def _rolling_std(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate rolling standard deviation features.
        """

        logger.info("Generating Rolling Std features...")

        grouped = df.groupby(self.group_column)

        for sensor in self.sensor_columns:

            feature_name = (
                f"{sensor}_roll_std_{self.rolling_window}"
            )

            df[feature_name] = (
                grouped[sensor]
                .transform(
                    lambda x: x.rolling(
                        window=self.rolling_window,
                        min_periods=1,
                    ).std()
                )
            )

        return df

    

    # Lag Features

    def _lag_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate lag features.
        """

        logger.info("Generating Lag Features...")

        grouped = df.groupby(self.group_column)

        for sensor in self.sensor_columns:

            for lag in self.lags:

                feature_name = f"{sensor}_lag_{lag}"

                df[feature_name] = (
                    grouped[sensor]
                    .shift(lag)
                )
        return df



    # Rate of Change


    def _rate_of_change(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate first-order difference features.
        """

        logger.info("Generating Rate of Change features...")

        grouped = df.groupby(self.group_column)

        for sensor in self.sensor_columns:

            feature_name = f"{sensor}_diff"

            df[feature_name] = (
                grouped[sensor]
                .diff()
            )
        return df