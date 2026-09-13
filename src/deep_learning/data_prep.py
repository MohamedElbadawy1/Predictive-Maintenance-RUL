"""
src/deep_learning/data_prep.py — shared data preparation for every
sequence model (LSTM, GRU, ...).

This used to live inside Pipeline/train_lstm.py and get imported from
Pipeline/train_gru.py directly. That cross-import between two
executable scripts turned out to be fragile on Windows (intermittent
`ModuleNotFoundError: No module named 'Pipeline'`, most likely a
case-sensitivity interaction between the auto-added script directory
on sys.path and the `Pipeline` package name). Shared logic belongs in
src/, not in another Pipeline/ script -- which is exactly the
convention Pipeline/train_with_tuning.py and
Pipeline/train_with_best_params.py already follow for the CatBoost
path (both import from src/training/pipeline.py, never from each
other). This module brings the LSTM/GRU path in line with that.
"""
from __future__ import annotations

from typing import List

import numpy as np
import pandas as pd

from src.config.config import (
    TRAIN_DATA_PATH, TEST_DATA_PATH, RUL_DATA_PATH,
    VALIDATION_SIZE, RANDOM_STATE, DEFAULT_RUL_CAP, DEFAULT_WINDOW_SIZE,
    ENGINE_COLUMN, TARGET_COLUMN,
)
from src.utils.constant import SENSOR_COLUMNS
from src.data.loader import DataLoader
from src.data.validator import DataValidator
from src.preprocessing.rul_generator import RULGenerator
from src.preprocessing.data_splitter import DataSplitter
from src.preprocessing.feature_scaler import FeatureScaler
from src.preprocessing.regime_normalizer import RegimeNormalizer
from src.preprocessing.sequence_generator import SequenceGenerator
from src.logger.logger import logger

# 21 sensors + 3 operational settings + time_in_cycles = 25 raw features
# (Sprint 12's finding: this beats the 109-feature CatBoost feature set
# for the LSTM by 15.8% MAE).
RAW_FEATURE_COLUMNS: List[str] = [
    *SENSOR_COLUMNS,
    "operational_setting_1",
    "operational_setting_2",
    "operational_setting_3",
    "time_in_cycles",
]

SORT_TIME_COLUMN = "_sort_time"  # unscaled cycle number, used only for
                                   # chronological ordering within an
                                   # engine — the scaled copy of
                                   # time_in_cycles is still one of the
                                   # 25 model features.


def _scale_and_pack(
    df: pd.DataFrame,
    scaler: FeatureScaler,
    fit: bool,
    include_target: bool,
) -> pd.DataFrame:
    """Scale RAW_FEATURE_COLUMNS and pack them back together with the
    engine id, an unscaled sort key, and (optionally) the target — the
    single flat frame SequenceGenerator expects."""

    scaled = (
        scaler.fit_transform(df[RAW_FEATURE_COLUMNS])
        if fit
        else scaler.transform(df[RAW_FEATURE_COLUMNS])
    )

    parts = [
        df[[ENGINE_COLUMN]].reset_index(drop=True),
        df[["time_in_cycles"]].reset_index(drop=True).rename(columns={"time_in_cycles": SORT_TIME_COLUMN}),
        scaled.reset_index(drop=True),
    ]
    if include_target:
        parts.insert(1, df[[TARGET_COLUMN]].reset_index(drop=True))

    return pd.concat(parts, axis=1)


def prepare_lstm_sequences(
    window_size: int = DEFAULT_WINDOW_SIZE,
    regime_aware: bool = True,
    n_regimes: int = 6,
) -> dict:
    """Raw data -> validate -> RUL -> engine split -> [regime normalize]
    -> scale (raw features only) -> sequence windows. Shared by both
    train_lstm.py and train_gru.py — identical preprocessing is what
    makes the LSTM-vs-GRU comparison fair."""

    loader = DataLoader(train_path=TRAIN_DATA_PATH, test_path=TEST_DATA_PATH, rul_path=RUL_DATA_PATH)
    train_raw = loader.load_train()
    test_raw = loader.load_test()
    rul_raw = loader.load_rul()

    DataValidator(train_raw, test_raw, rul_raw).validate_all()

    train_with_rul = RULGenerator(train_raw).generate(cap=DEFAULT_RUL_CAP)

    splitter = DataSplitter(test_size=VALIDATION_SIZE, engine_column=ENGINE_COLUMN, random_state=RANDOM_STATE)
    train_split, val_split = splitter.split(train_with_rul)

    if regime_aware:
        normalizer = RegimeNormalizer(n_regimes=n_regimes, sensor_columns=SENSOR_COLUMNS, random_state=RANDOM_STATE)
        train_split = normalizer.fit(train_split).transform(train_split)
        val_split = normalizer.transform(val_split)
        test_raw = normalizer.transform(test_raw)

    scaler = FeatureScaler()
    train_packed = _scale_and_pack(train_split, scaler, fit=True, include_target=True)
    val_packed = _scale_and_pack(val_split, scaler, fit=False, include_target=True)
    test_packed = _scale_and_pack(test_raw, scaler, fit=False, include_target=False)

    seq_gen = SequenceGenerator(window_size=window_size, time_column=SORT_TIME_COLUMN, target_column=TARGET_COLUMN)

    X_train, y_train, _ = seq_gen.transform(train_packed, feature_columns=RAW_FEATURE_COLUMNS)
    X_val, y_val, _ = seq_gen.transform(val_packed, feature_columns=RAW_FEATURE_COLUMNS)
    X_test, test_engine_ids, skipped = seq_gen.get_last_window(test_packed, feature_columns=RAW_FEATURE_COLUMNS)

    # Official RUL file has one row per engine, ordered by unit_number
    # (1-indexed) — pick out the engines get_last_window actually kept.
    all_test_rul = rul_raw[TARGET_COLUMN].to_numpy()
    y_test_true = np.array([all_test_rul[engine_id - 1] for engine_id in test_engine_ids])

    if skipped:
        logger.info(f"{len(skipped)} test engine(s) skipped (fewer than {window_size} cycles): {skipped}")

    return {
        "X_train": X_train, "y_train": y_train,
        "X_val": X_val, "y_val": y_val,
        "X_test": X_test, "y_test_true": y_test_true,
        "scaler": scaler,
    }
