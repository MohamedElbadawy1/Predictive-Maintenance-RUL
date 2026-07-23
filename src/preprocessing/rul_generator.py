"""
RUL Generator Module.

This module provides the RULGenerator class responsible for generating
Remaining Useful Life (RUL) targets for the NASA CMAPSS training dataset.

"""

from __future__ import annotations

import pandas as pd

from src.logger.logger import logger
from src.exceptions.custom_exception import CustomException


class RULGenerator:
    """
    Generate Remaining Useful Life (RUL) targets.

    Responsibilities
    ----------------
    - Compute raw RUL.
    - Optionally apply RUL capping.
    - Never modify the original DataFrame.
    """

    REQUIRED_COLUMNS = ["unit_number", "time_in_cycles"]

    def __init__(self, train_df: pd.DataFrame) -> None:
        self.train_df = train_df
        self._validate_input()


    def _validate_input(self) -> None:
        if self.train_df is None:
            raise CustomException("Training DataFrame cannot be None.")

        if self.train_df.empty:
            raise CustomException("Training DataFrame is empty.")

        missing_columns = [
            col
            for col in self.REQUIRED_COLUMNS
            if col not in self.train_df.columns
        ]

        if missing_columns:
            raise CustomException(
                f"Missing required columns: {missing_columns}"
            )

    def _compute_raw_rul(self, df: pd.DataFrame) -> pd.Series:
        ## Compute raw Remaining Useful Life.

        max_cycles = (
            df.groupby("unit_number")["time_in_cycles"]
            .transform("max")
        )
        return max_cycles - df["time_in_cycles"]

    @staticmethod
    def _apply_cap(rul: pd.Series,cap: int,) -> pd.Series:
        ## Apply upper cap to RUL.
        return rul.clip(upper=cap)

    def generate(self,cap: int | None = None,) -> pd.DataFrame:
        """
        Generate RUL.

        Parameters
        ----------
        cap : int | None, default=None
            Optional maximum RUL.

        Returns
        -------
        pd.DataFrame
            New DataFrame containing the generated RUL column.
        """

        logger.info("Generating Remaining Useful Life (RUL)...")

        df = self.train_df.copy()
        rul = self._compute_raw_rul(df)

        if cap is not None:
            if cap <= 0:
                raise CustomException(
                    "RUL cap must be greater than zero."
                )
            logger.info(f"Applying RUL cap = {cap}")
            rul = self._apply_cap(rul, cap)

        df["RUL"] = rul
        logger.info("RUL generated successfully.")
        return df