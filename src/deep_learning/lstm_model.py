"""
LSTM baseline architecture for RUL sequence regression.

Reconstructed from docs/Sprint_12_LSTM_Baseline.md's documented
architecture after the original module was lost to a sandbox reset
(see the "What Was Lost" note in
docs/Sprint_15_LSTM_Cap150_Test_Evaluation.md). The architecture and
feature-set decision below are not new choices -- they are the ones
Sprint 12 already tested and justified:

- Raw sensor + operational-setting columns only (no lag/rolling/diff
  engineered features). Sprint 12 found the 109 CatBoost features
  actively hurt the LSTM (15.8% worse MAE) because a sliding window
  already encodes the temporal information those engineered features
  re-derive by hand.
- A single LSTM layer feeding a Dense(1) regression head, kept
  deliberately simple -- the goal was a fair baseline comparison
  against CatBoost, not a maximally tuned architecture.
"""
from __future__ import annotations

from tensorflow import keras
from tensorflow.keras import layers


def build_lstm_baseline(
    window_size: int,
    n_features: int,
    lstm_units: int = 64,
    dropout: float = 0.2,
    learning_rate: float = 1e-3,
) -> keras.Model:
    """
    Build the Sprint 12 LSTM baseline:

        Input(window_size, n_features) -> LSTM(lstm_units)
            -> Dropout(dropout) -> Dense(1)

    Parameters
    ----------
    window_size : int
        Number of time steps per sequence (Sprint 12/15 used 30).
    n_features : int
        Number of raw features per time step (25 for FD004: 21 sensors
        + 3 operational settings + time_in_cycles).
    """

    model = keras.Sequential(
        [
            keras.Input(shape=(window_size, n_features), name="sequence_input"),
            layers.LSTM(lstm_units, name="lstm"),
            layers.Dropout(dropout, name="dropout"),
            layers.Dense(1, name="rul_output"),
        ],
        name="lstm_baseline",
    )

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss="mse",
        metrics=["mae"],
    )

    return model
