"""
Prepare LSTM sequence data (Sprint 12).

Rebuilds train/validation with unit_number preserved (the saved
train_prepared.csv / validation_prepared.csv dropped it, since CatBoost
doesn't need it — but SequenceGenerator does, to group and order cycles
per engine).

Supports two feature sets, since which one is "right" is a genuinely
different question for a sequence model than it was for CatBoost:

  --feature-set full : the 109 features frozen in Sprint 10 (raw + lag +
                        diff + time_in_cycles). These lag/diff features
                        were engineered to give a *memoryless* row-by-row
                        model (CatBoost) temporal context. An LSTM
                        doesn't need that — it already sees the raw
                        sequence directly.

  --feature-set raw  : only the true raw signal (21 sensors + 3
                        operational settings + time_in_cycles = 25
                        features). Lets the LSTM learn temporal patterns
                        itself from the actual sequence, instead of also
                        being fed pre-computed lag/diff versions of the
                        same values (which duplicates information already
                        present across the window's time axis).

Pipeline: engineered features checkpoint -> same engine split as every
prior sprint -> reduce to the chosen feature set -> scale (fit on train
only) -> SequenceGenerator -> cache to .npz for fast, resumable reuse.
"""
import argparse
import sys
sys.path.insert(0, "/home/claude/Predictive-Maintenance-RUL")

import numpy as np
import pandas as pd

from src.config.config import (
    FEATURES_DATA_DIR,
    SELECTED_FEATURES_PATH,
    SCALERS_DIR,
    VALIDATION_SIZE,
    RANDOM_STATE,
    TARGET_COLUMN,
    ENGINE_COLUMN,
)
from src.explainability.feature_reducer import FeatureReducer
from src.explainability.feature_selector import FeatureCategorySelector
from src.preprocessing.data_splitter import DataSplitter
from src.preprocessing.feature_scaler import FeatureScaler
from src.preprocessing.sequence_generator import SequenceGenerator

parser = argparse.ArgumentParser()
parser.add_argument("--feature-set", choices=["full", "raw"], default="full")
parser.add_argument("--window-size", type=int, default=30)
args = parser.parse_args()

WINDOW_SIZE = args.window_size
SEQUENCE_DIR = FEATURES_DATA_DIR.parent / "sequences"
SEQUENCE_DIR.mkdir(parents=True, exist_ok=True)

TIME_COLUMN = "time_in_cycles"

print(f"Feature set: {args.feature_set}")
print("Loading engineered feature checkpoint...")
feature_df = pd.read_csv(FEATURES_DATA_DIR / "train_features.csv")
print(f"  {feature_df.shape}")

if args.feature_set == "full":
    final_features = FeatureReducer.load_selected_features(SELECTED_FEATURES_PATH)
else:
    all_columns = [c for c in feature_df.columns if c not in (ENGINE_COLUMN, TARGET_COLUMN)]
    final_features = FeatureCategorySelector.only(all_columns, categories=["raw", "time"])

print(f"Feature count: {len(final_features)}")

# Same engine-level split as every CatBoost sprint (same config, same
# random_state) so the LSTM comparison is against the identical engines.
splitter = DataSplitter(
    test_size=VALIDATION_SIZE,
    engine_column=ENGINE_COLUMN,
    random_state=RANDOM_STATE,
)
train_df, val_df = splitter.split(feature_df)
print(f"Train engines: {train_df[ENGINE_COLUMN].nunique()} | "
      f"Val engines: {val_df[ENGINE_COLUMN].nunique()}")

scaler = FeatureScaler()
train_df = train_df.copy()
val_df = val_df.copy()

# Lag/diff features (only present in "full") are NaN for the first few
# cycles of every engine (nothing to lag/difference against yet) —
# CatBoost handles NaN natively, but Keras/TensorFlow does not: a
# single NaN input poisons the entire loss to NaN. Fill per engine
# (forward-fill, then back-fill for any engine whose very first cycles
# are still NaN). Harmless no-op for "raw" (no NaNs to begin with).
print("Filling early-cycle NaNs per engine (no-op for raw feature set)...")
train_nan_before = train_df[final_features].isna().sum().sum()
val_nan_before = val_df[final_features].isna().sum().sum()

train_df[final_features] = (
    train_df.groupby(ENGINE_COLUMN)[final_features]
    .transform(lambda g: g.ffill().bfill())
)
val_df[final_features] = (
    val_df.groupby(ENGINE_COLUMN)[final_features]
    .transform(lambda g: g.ffill().bfill())
)
print(f"  Train NaNs: {train_nan_before} -> {train_df[final_features].isna().sum().sum()}")
print(f"  Val NaNs:   {val_nan_before} -> {val_df[final_features].isna().sum().sum()}")

train_df[final_features] = scaler.fit_transform(train_df[final_features])
val_df[final_features] = scaler.transform(val_df[final_features])

SCALERS_DIR.mkdir(parents=True, exist_ok=True)
scaler_path = SCALERS_DIR / f"lstm_feature_scaler_{args.feature_set}.pkl"
scaler.save(scaler_path)
print(f"Saved LSTM feature scaler -> {scaler_path}")

generator = SequenceGenerator(
    window_size=WINDOW_SIZE,
    group_column=ENGINE_COLUMN,
    time_column=TIME_COLUMN,
    target_column=TARGET_COLUMN,
)

print("Generating training sequences...")
X_train_seq, y_train_seq, train_groups = generator.transform(train_df, final_features)
print(f"  X_train_seq: {X_train_seq.shape}  y_train_seq: {y_train_seq.shape}")

print("Generating validation sequences...")
X_val_seq, y_val_seq, val_groups = generator.transform(val_df, final_features)
print(f"  X_val_seq: {X_val_seq.shape}  y_val_seq: {y_val_seq.shape}")

out_path = SEQUENCE_DIR / f"lstm_sequences_{args.feature_set}_w{WINDOW_SIZE}.npz"
np.savez_compressed(
    out_path,
    X_train=X_train_seq, y_train=y_train_seq, train_groups=train_groups,
    X_val=X_val_seq, y_val=y_val_seq, val_groups=val_groups,
    window_size=WINDOW_SIZE, n_features=len(final_features),
)
print(f"Cached sequences -> {out_path}")
print("DONE")
