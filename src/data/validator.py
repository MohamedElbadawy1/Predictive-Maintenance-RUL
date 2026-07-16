from __future__ import annotations
from src.logger.logger import logger
from src.exceptions.custom_exception import CustomException
from src.utils.constant import EXPECTED_COLUMNS

import pandas as pd
from pandas.api.types import is_numeric_dtype
import numpy as np

from typing import Dict, List


class DataValidator:
    """
    Validate NASA CMAPSS datasets.

    Responsibilities
    ----------------
    - Validate train/test datasets
    - Validate RUL dataset

    This class DOES NOT:
    --------------------
    - Fix data
    - Remove duplicates
    - Fill missing values
    - Perform preprocessing
    """

    def __init__(self, train_df: pd.DataFrame, test_df: pd.DataFrame, rul_df: pd.DataFrame) -> None:

        self.train_df = train_df
        self.test_df = test_df
        self.rul_df = rul_df

    ## Public Methods

    def validate_train(self) -> Dict:

        logger.info("Validating training dataset...")

        return self._validate_dataset(
            self.train_df,
            dataset_name="Train",
            expected_columns=EXPECTED_COLUMNS,
        )

    def validate_test(self) -> Dict:

        logger.info("Validating testing dataset...")

        return self._validate_dataset(
            self.test_df,
            dataset_name="Test",
            expected_columns=EXPECTED_COLUMNS,
        )

    def validate_rul(self) -> Dict:

        logger.info("Validating RUL dataset...")

        report = {
            "valid": True,
            "errors": [],
            "warnings": [],
        }

        if self.rul_df.shape[1] != 1:
            report["valid"] = False
            report["errors"].append(
                "RUL dataset must contain exactly one column."
            )

        if self.rul_df.isna().sum().sum() > 0:
            report["warnings"].append("Missing values found.")

        if self.rul_df.duplicated().sum() > 0:
            report["warnings"].append("Duplicate rows found.")

        return report

    def validate_all(self) -> Dict:

        return {
            "train": self.validate_train(),
            "test": self.validate_test(),
            "rul": self.validate_rul(),
        }

    # Private Methods

    def _validate_dataset( self, df: pd.DataFrame, dataset_name: str, expected_columns: List[str]) -> Dict:

        report = {
            "valid": True,
            "errors": [],
            "warnings": [],
        }

        # Number of Columns
        if df.shape[1] != len(expected_columns):

            report["valid"] = False

            report["errors"].append(
                f"{dataset_name}: Expected "
                f"{len(expected_columns)} columns "
                f"but found {df.shape[1]}."
            )

        # Column Names
        elif list(df.columns) != expected_columns:

            report["valid"] = False

            report["errors"].append(
                f"{dataset_name}: Column names do not match."
            )

        # Missing Values
        missing = df.isna().sum().sum()
        if missing > 0:
            report["warnings"].append(
                f"{dataset_name}: {missing} missing values found."
            )

        # Duplicate Rows
        duplicates = df.duplicated().sum()
        if duplicates > 0:
            report["warnings"].append(
                f"{dataset_name}: {duplicates} duplicate rows found."
            )

        # Numeric Data Types
        non_numeric = []
        for column in df.columns:
            if not is_numeric_dtype(df[column]):
                non_numeric.append(column)

        if non_numeric:
            report["valid"] = False
            report["errors"].append(
                f"{dataset_name}: Non-numeric columns: {non_numeric}"
            )

        # Positive Values
        if (df["unit_number"] <= 0).any():
            report["valid"] = False
            report["errors"].append(
                "unit_number contains non-positive values."
            )

        if (df["time_in_cycles"] <= 0).any():
            report["valid"] = False
            report["errors"].append(
                "time_in_cycles contains non-positive values."
            )

        # Cycle Starts at 1
        first_cycles = (
            df.groupby("unit_number")["time_in_cycles"]
            .min()
        )
        invalid_units = first_cycles[first_cycles != 1]

        if len(invalid_units) > 0:
            report["warnings"].append(
                f"{len(invalid_units)} engines do not start at cycle 1."
            )
        return report