"""
Pipeline/train_lstm.py — raw data -> sequences -> LSTM baseline ->
validation + official test-set evaluation, logged to MLflow as a
comparison run (NOT registered as champion — see the note below).

Thin CLI wrapper, same convention as train_with_tuning.py /
train_with_best_params.py: the script itself only parses args, prints
the result, and orchestrates calls into src/ — it does not contain
model logic itself.

Reconstructs docs/Sprint_12_LSTM_Baseline.md and
docs/Sprint_15_LSTM_Cap150_Test_Evaluation.md's LSTM baseline after the
original src/deep_learning module was lost to a sandbox reset. Reuses
every existing preprocessing building block as-is (FeatureScaler,
RegimeNormalizer, DataSplitter, SequenceGenerator) — nothing in
src/preprocessing needed to change for this. The only genuinely new
decisions carried over from the original work are: raw features only
(21 sensors + 3 operational settings + time_in_cycles — engineered
lag/rolling/diff features measurably hurt the LSTM, see Sprint 12),
RUL cap=150 (Sprint 14), and window_size=30.

Not registry-eligible: CatBoost beat the LSTM by 18.5% MAE on the
official test set in the original run (Sprint 15), and there is no
reason to expect that to flip here. This script logs an MLflow
comparison run only — it never touches the "champion" alias, unlike
TrainingPipeline.

Usage:
    python Pipeline/train_lstm.py
    python Pipeline/train_lstm.py --max-epochs 40 --no-regime-aware
    python Pipeline/train_lstm.py --resume   # continue an interrupted run
"""
import argparse
import sys
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config.config import (
    TRAIN_DATA_PATH, TEST_DATA_PATH, RUL_DATA_PATH,
    VALIDATION_SIZE, RANDOM_STATE, DEFAULT_RUL_CAP, DEFAULT_WINDOW_SIZE,
    ENGINE_COLUMN, TARGET_COLUMN, MODELS_DIR, LSTM_CHECKPOINT_PATH,
)
from src.utils.constant import SENSOR_COLUMNS
from src.data.loader import DataLoader
from src.data.validator import DataValidator
from src.preprocessing.rul_generator import RULGenerator
from src.preprocessing.data_splitter import DataSplitter
from src.preprocessing.feature_scaler import FeatureScaler
from src.preprocessing.regime_normalizer import RegimeNormalizer
from src.preprocessing.sequence_generator import SequenceGenerator
from src.deep_learning.lstm_model import build_lstm_baseline
from src.deep_learning.dl_trainer import DLTrainer
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
    -> scale (raw features only) -> sequence windows."""

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


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--window-size", type=int, default=DEFAULT_WINDOW_SIZE)
    parser.add_argument("--max-epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--no-regime-aware", action="store_true", help="Disable regime-aware normalization")
    parser.add_argument("--resume", action="store_true", default=True, help="Resume from checkpoint if one exists (default)")
    parser.add_argument("--no-resume", dest="resume", action="store_false", help="Ignore any existing checkpoint and restart at epoch 0")
    args = parser.parse_args()

    data = prepare_lstm_sequences(window_size=args.window_size, regime_aware=not args.no_regime_aware)

    model = build_lstm_baseline(window_size=args.window_size, n_features=len(RAW_FEATURE_COLUMNS))

    trainer = DLTrainer(
        model,
        checkpoint_path=LSTM_CHECKPOINT_PATH,
        run_name="lstm_baseline",
        tags={
            "model_family": "lstm",
            "feature_set": "raw_25",
            "rul_cap": str(DEFAULT_RUL_CAP),
            "normalization": "regime_aware" if not args.no_regime_aware else "global",
        },
    )

    val_metrics = trainer.train(
        data["X_train"], data["y_train"],
        data["X_val"], data["y_val"],
        max_epochs=args.max_epochs,
        batch_size=args.batch_size,
        patience=args.patience,
        resume=args.resume,
    )

    test_preds = trainer.predict(data["X_test"])
    from src.evaluation.evaluator import RegressionEvaluator
    test_metrics = RegressionEvaluator().evaluate(data["y_test_true"], test_preds)

    trainer.save(MODELS_DIR / "lstm_baseline_final.keras")

    print("\n" + "=" * 60)
    print("LSTM BASELINE RESULT")
    print("=" * 60)
    print(f"Validation metrics : {val_metrics}")
    print(f"Test metrics       : {test_metrics}")
    print(f"MLflow run_id      : {trainer.last_run_id}")
    print("Not registered as champion — logged as a comparison run only.")
