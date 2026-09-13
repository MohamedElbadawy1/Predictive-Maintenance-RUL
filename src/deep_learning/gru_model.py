"""
GRU baseline architecture for RUL sequence regression.

Mirrors src/deep_learning/lstm_model.py exactly -- same raw-25-feature
input, same single-recurrent-layer-into-Dense(1) shape, same cap=150 /
window_size=30 protocol -- so the LSTM-vs-GRU comparison differs only
in the recurrent cell, not in data, preprocessing, or training setup.
This was the comparison Sprint 16 originally set out to make (see
docs/Sprint_15_LSTM_Cap150_Test_Evaluation.md's reconstruction note).
"""
from __future__ import annotations

from tensorflow import keras
from tensorflow.keras import layers


def build_gru_baseline(
    window_size: int,
    n_features: int,
    gru_units: int = 64,
    dropout: float = 0.2,
    learning_rate: float = 1e-3,
) -> keras.Model:
    """
    Build the GRU counterpart of build_lstm_baseline():

        Input(window_size, n_features) -> GRU(gru_units)
            -> Dropout(dropout) -> Dense(1)

    Parameters
    ----------
    window_size : int
        Number of time steps per sequence (30, matching the LSTM run).
    n_features : int
        Number of raw features per time step (25 for FD004: 21 sensors
        + 3 operational settings + time_in_cycles).
    """

    model = keras.Sequential(
        [
            keras.Input(shape=(window_size, n_features), name="sequence_input"),
            layers.GRU(gru_units, name="gru"),
            layers.Dropout(dropout, name="dropout"),
            layers.Dense(1, name="rul_output"),
        ],
        name="gru_baseline",
    )

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss="mse",
        metrics=["mae"],
    )

    return model
