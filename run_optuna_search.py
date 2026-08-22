import argparse
import sys

sys.path.insert(0, "/home/claude/Predictive-Maintenance-RUL")

import pandas as pd
import optuna
from optuna.samplers import TPESampler

from src.config.config import PROCESSED_DATA_DIR, SELECTED_FEATURES_PATH, TARGET_COLUMN
from src.explainability.feature_reducer import FeatureReducer
from src.optimization.hyperparameter_tuner import CatBoostTuner

STORAGE = "sqlite:////home/claude/Predictive-Maintenance-RUL/optuna_study.db"
STUDY_NAME = "catboost_rul_optimization"

parser = argparse.ArgumentParser()
parser.add_argument("--n-trials", type=int, default=3)
args = parser.parse_args()

import time

final_features = FeatureReducer.load_selected_features(SELECTED_FEATURES_PATH)

train_df = pd.read_csv(PROCESSED_DATA_DIR / "train_prepared.csv")
validation_df = pd.read_csv(PROCESSED_DATA_DIR / "validation_prepared.csv")

X_train = train_df[final_features]
y_train = train_df[TARGET_COLUMN]
X_val = validation_df[final_features]
y_val = validation_df[TARGET_COLUMN]

tuner = CatBoostTuner(X_train, y_train, X_val, y_val, random_state=42)

# NOTE: this script may be invoked multiple times (once per batch of trials)
# so the search can be resumed across separate process runs via SQLite
# storage. A fixed sampler seed would replay the exact same suggestion
# sequence on every invocation (since the RNG resets to the same state),
# producing duplicate trials instead of new ones. Seeding from the current
# wall-clock time avoids that; exact bit-for-bit reproducibility of the
# search sequence is traded for correctness across resumed runs.
sampler_seed = int(time.time() * 1000) % (2**31 - 1)

tuner.study = optuna.create_study(
    direction="minimize",
    sampler=TPESampler(seed=sampler_seed),
    study_name=STUDY_NAME,
    storage=STORAGE,
    load_if_exists=True,
)

n_done_before = len(tuner.study.trials)
print(f"Trials already completed: {n_done_before}", flush=True)

tuner.study.optimize(
    tuner._objective,
    n_trials=args.n_trials,
    show_progress_bar=False,
)

n_done_after = len(tuner.study.trials)
print(f"Trials completed now: {n_done_after}", flush=True)
print(f"Best MAE so far: {tuner.study.best_value:.4f}", flush=True)
print(f"Best params so far: {tuner.study.best_params}", flush=True)
