from typing import Tuple
import sys

import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

from src.exceptions.custom_exception import CustomException
from src.logger.logger import logger


class DataSplitter:
    """
    Split the dataset into training and validation sets based on engine IDs.

    Each engine is treated as a single group to prevent data leakage while
    preserving the temporal order of every engine trajectory.
    """

    def __init__(
        self,
        test_size: float = 0.2,
        engine_column: str = "unit_number",
        random_state: int = 42,
    ) -> None:

        self.test_size = test_size
        self.engine_column = engine_column
        self.random_state = random_state

    def split(
        self,
        df: pd.DataFrame,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:

        logger.info("Starting engine-based train/validation split...")

        self._validate_input(df)

        splitter = GroupShuffleSplit(
            n_splits=1,
            test_size=self.test_size,
            random_state=self.random_state,
        )

        train_idx, val_idx = next(
            splitter.split(
                df,
                groups=df[self.engine_column],
            )
        )

        train_df = (
            df.iloc[train_idx]
            .copy()
            .reset_index(drop=True)
        )

        val_df = (
            df.iloc[val_idx]
            .copy()
            .reset_index(drop=True)
        )

        train_engines = train_df[self.engine_column].nunique()
        val_engines = val_df[self.engine_column].nunique()

        logger.info(
            f"Train Engines: {train_engines} | "
            f"Validation Engines: {val_engines}"
        )

        logger.info("Data splitting completed successfully.")

        return train_df, val_df

    def _validate_input(self, df: pd.DataFrame) -> None:

        if df.empty:
            raise CustomException(
                "Input DataFrame is empty.",
                sys,
            )

        if self.engine_column not in df.columns:
            raise CustomException(
                f"'{self.engine_column}' column not found.",
                sys,
            )

        if not (0 < self.test_size < 1):
            raise CustomException(
                "test_size must be between 0 and 1.",
                sys,
            )