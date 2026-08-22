from __future__ import annotations

import sys
from typing import List, Tuple

import numpy as np
import pandas as pd

from src.exceptions.custom_exception import CustomException
from src.logger.logger import logger


class SequenceGenerator:
    """
    Convert per-cycle tabular data into fixed-size sequence windows for
    sequence models (LSTM, GRU, ...).

    For each engine, chronologically ordered cycles are grouped into
    fixed-size sliding windows. Each window's target is the RUL value at
    the window's *last* cycle:

        cycles 1-30  -> predict RUL at cycle 30
        cycles 2-31  -> predict RUL at cycle 31
        cycles 3-32  -> predict RUL at cycle 32

    Windows never span two engines: they're built per engine group, in
    chronological order, and an engine with fewer cycles than
    `window_size` contributes no windows at all (it is skipped, not
    padded) — kept as a simple, honest baseline behavior rather than
    hiding short trajectories behind padding.

    Example
    -------
    generator = SequenceGenerator(window_size=30)

    X_seq, y_seq, engine_ids = generator.transform(
        df,
        feature_columns=final_features,
    )

    # X_seq.shape == (n_samples, window_size, n_features)
    """

    def __init__(
        self,
        window_size: int = 30,
        stride: int = 1,
        group_column: str = "unit_number",
        time_column: str = "time_in_cycles",
        target_column: str = "RUL",
    ):

        self.window_size = window_size
        self.stride = stride
        self.group_column = group_column
        self.time_column = time_column
        self.target_column = target_column

    def transform(
        self,
        df: pd.DataFrame,
        feature_columns: List[str],
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:

        self._validate_input(df, feature_columns)

        logger.info(
            f"Generating sequences: window_size={self.window_size}, "
            f"stride={self.stride}, {df[self.group_column].nunique()} engines."
        )

        sequences = []
        targets = []
        engine_ids = []

        skipped_engines = 0

        for engine_id, engine_df in df.groupby(self.group_column):

            # Chronological order is required — never assume the input
            # is already sorted (e.g. after a GroupShuffleSplit).
            engine_df = engine_df.sort_values(self.time_column)

            n_cycles = len(engine_df)

            if n_cycles < self.window_size:
                skipped_engines += 1
                continue

            feature_matrix = engine_df[feature_columns].to_numpy()
            target_values = engine_df[self.target_column].to_numpy()

            for start in range(
                0,
                n_cycles - self.window_size + 1,
                self.stride,
            ):

                end = start + self.window_size

                sequences.append(feature_matrix[start:end])
                targets.append(target_values[end - 1])
                engine_ids.append(engine_id)

        if not sequences:
            raise CustomException(
                "No sequences were generated — every engine had fewer "
                f"cycles than window_size={self.window_size}.",
                sys,
            )

        X = np.array(sequences, dtype=np.float32)
        y = np.array(targets, dtype=np.float32)
        groups = np.array(engine_ids)

        if skipped_engines:
            logger.info(
                f"Skipped {skipped_engines} engine(s) with fewer than "
                f"{self.window_size} cycles (no padding applied)."
            )

        logger.info(
            f"Generated {X.shape[0]} sequences of shape "
            f"({X.shape[1]}, {X.shape[2]})."
        )

        return X, y, groups

    def _validate_input(
        self,
        df: pd.DataFrame,
        feature_columns: List[str],
    ) -> None:

        if df.empty:
            raise CustomException("Input DataFrame is empty.", sys)

        required_columns = (
            [self.group_column, self.time_column, self.target_column]
            + list(feature_columns)
        )

        missing = [c for c in required_columns if c not in df.columns]

        if missing:
            raise CustomException(
                f"Missing required columns: {missing}",
                sys,
            )

        if self.window_size <= 0:
            raise CustomException(
                "window_size must be greater than zero.",
                sys,
            )

        if self.stride <= 0:
            raise CustomException(
                "stride must be greater than zero.",
                sys,
            )
