"""
Shared data-preparation logic for the training pipeline files
(pipeline/train_with_best_params.py, pipeline/train_with_tuning.py) and
TrainingPipeline. Extracted to one place so the exact same raw-data ->
model-ready-features chain is guaranteed identical across all training
paths — duplicating this across three files is exactly the kind of
scattered-logic problem Phase 1 (InferencePipeline) already fixed for
prediction; this does the same for training.
"""
from typing import List, Optional, Tuple

import pandas as pd

from src.config.config import (
    TRAIN_DATA_PATH, TEST_DATA_PATH, RUL_DATA_PATH,
    VALIDATION_SIZE, RANDOM_STATE, DEFAULT_RUL_CAP,
    ROLLING_WINDOW, LAGS, ENGINE_COLUMN, TARGET_COLUMN,
)
from src.utils.constant import SENSOR_COLUMNS
from src.data.loader import DataLoader
from src.data.validator import DataValidator
from src.preprocessing.rul_generator import RULGenerator
from src.preprocessing.feature_engineer import FeatureEngineer
from src.preprocessing.data_splitter import DataSplitter
from src.preprocessing.feature_scaler import FeatureScaler
from src.preprocessing.regime_normalizer import RegimeNormalizer
from src.explainability.feature_selector import FeatureCategorySelector


def prepare_training_data(
    regime_aware: bool = True,
    n_regimes: int = 6,
) -> dict:
    """
    Raw data -> validate -> RUL -> engine split -> [regime normalize] ->
    feature engineer -> feature select -> scale.

    Returns a dict rather than a long tuple, so callers can pick out what
    they need by name instead of tracking positional order.
    """

    loader = DataLoader(train_path=TRAIN_DATA_PATH, test_path=TEST_DATA_PATH, rul_path=RUL_DATA_PATH)
    train_raw = loader.load_train()
    test_raw = loader.load_test()
    rul_raw = loader.load_rul()

    DataValidator(train_raw, test_raw, rul_raw).validate_all()

    train_with_rul = RULGenerator(train_raw).generate(cap=DEFAULT_RUL_CAP)

    splitter = DataSplitter(test_size=VALIDATION_SIZE, engine_column=ENGINE_COLUMN, random_state=RANDOM_STATE)
    train_split, val_split = splitter.split(train_with_rul)

    regime_normalizer = None
    if regime_aware:
        regime_normalizer = RegimeNormalizer(n_regimes=n_regimes, sensor_columns=SENSOR_COLUMNS, random_state=RANDOM_STATE)
        train_split = regime_normalizer.fit(train_split).transform(train_split)
        val_split = regime_normalizer.transform(val_split)
        test_raw = regime_normalizer.transform(test_raw)

    engineer = FeatureEngineer(sensor_columns=SENSOR_COLUMNS, rolling_window=ROLLING_WINDOW, lags=LAGS)
    train_features = engineer.transform(train_split)
    val_features = engineer.transform(val_split)
    test_features = engineer.transform(test_raw)

    exclude_cols = (ENGINE_COLUMN, TARGET_COLUMN, "regime")
    all_columns = [c for c in train_features.columns if c not in exclude_cols]
    final_features = FeatureCategorySelector.exclude(all_columns, categories=["rolling"])

    scaler = FeatureScaler()
    X_train = scaler.fit_transform(train_features[final_features])
    X_val = scaler.transform(val_features[final_features])
    y_train = train_features[TARGET_COLUMN].reset_index(drop=True)
    y_val = val_features[TARGET_COLUMN].reset_index(drop=True)

    last_rows = (
        test_features.sort_values([ENGINE_COLUMN, "time_in_cycles"])
        .groupby(ENGINE_COLUMN).tail(1).sort_values(ENGINE_COLUMN).reset_index(drop=True)
    )
    X_test = scaler.transform(last_rows[final_features])
    y_test_true = rul_raw["RUL"].to_numpy()

    return {
        "X_train": X_train, "y_train": y_train,
        "X_val": X_val, "y_val": y_val,
        "X_test": X_test, "y_test_true": y_test_true,
        "regime_normalizer": regime_normalizer,
        "scaler": scaler,
        "final_features": final_features,
    }
