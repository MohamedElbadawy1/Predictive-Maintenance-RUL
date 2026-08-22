from __future__ import annotations

from tensorflow import keras


def build_lstm_baseline(
    window_size: int,
    n_features: int,
    lstm_units: int = 64,
    dropout_rate: float = 0.2,
    learning_rate: float = 1e-3,
) -> keras.Model:
    """
    Simple LSTM baseline for RUL regression:

        Input -> LSTM -> Dropout -> Dense -> RUL

    Intentionally unoptimized (per Sprint 12: "Don't optimize it yet") —
    a single LSTM layer, one dropout layer, one dense output unit. This
    is the starting point Sprint 13+ tunes from, not a tuned model.
    """

    model = keras.Sequential(
        [
            keras.layers.Input(shape=(window_size, n_features)),
            keras.layers.LSTM(lstm_units),
            keras.layers.Dropout(dropout_rate),
            keras.layers.Dense(1),
        ]
    )

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss="mse",
        metrics=["mae"],
    )

    return model
