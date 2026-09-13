"""
DLTrainer: the Keras counterpart of src/models/base_trainer.py.

BaseTrainer assumes a scikit-learn-shaped model (`fit(X, y)`, no epochs,
no callbacks) -- a Keras model's `fit()` needs epochs/batch_size/
callbacks and benefits from checkpointing mid-training, so it gets its
own trainer rather than being forced into BaseTrainer's shape.

Resumability is a first-class feature here, not an afterthought: every
epoch is checkpointed to disk together with the last-completed-epoch
number, so an interrupted `train()` call can pick up where it left off
instead of restarting from epoch 0. This is the exact problem
docs/Sprint_15_LSTM_Cap150_Test_Evaluation.md flagged when the training
environment reset mid-run -- solved properly this time instead of
re-running from scratch.
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Dict, Optional, Union

import numpy as np
from tensorflow import keras

from src.evaluation.evaluator import RegressionEvaluator
from src.logger.logger import logger


class _EpochStateCallback(keras.callbacks.Callback):
    """Writes the last-completed-epoch number to a small JSON file after
    every epoch, so a later train() call knows where to resume from."""

    def __init__(self, state_path: Path):
        super().__init__()
        self.state_path = Path(state_path)

    def on_epoch_end(self, epoch, logs=None):
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "last_completed_epoch": epoch,
            "logs": {k: float(v) for k, v in (logs or {}).items()},
        }
        with open(self.state_path, "w") as f:
            json.dump(payload, f, indent=2)


class DLTrainer:
    """
    Example
    -------
    trainer = DLTrainer(
        model,
        checkpoint_path=MODELS_DIR / "lstm_baseline.keras",
        run_name="lstm_baseline",
        tags={"model_family": "lstm", "feature_set": "raw_25"},
    )
    val_metrics = trainer.train(X_train, y_train, X_val, y_val, max_epochs=40)
    trainer.save(MODELS_DIR / "lstm_baseline_final.keras")
    """

    def __init__(
        self,
        model: keras.Model,
        checkpoint_path: Union[str, Path],
        state_path: Optional[Union[str, Path]] = None,
        track_mlflow: bool = True,
        run_name: Optional[str] = None,
        tags: Optional[Dict] = None,
    ):

        self.model = model
        self.checkpoint_path = Path(checkpoint_path)
        self.state_path = (
            Path(state_path)
            if state_path is not None
            else self.checkpoint_path.with_suffix(".state.json")
        )
        self.track_mlflow = track_mlflow
        self.run_name = run_name or "dl_model_run"
        self.tags = tags
        self.last_run_id = None

        self.evaluator = RegressionEvaluator()

        self.tracker = None
        if self.track_mlflow:
            try:
                from src.experiments.mlflow_tracker import MLflowTracker
                self.tracker = MLflowTracker()
            except Exception as exc:
                logger.warning(
                    f"MLflow tracking unavailable ({exc}) — training will "
                    "proceed without it."
                )
                self.track_mlflow = False

    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        max_epochs: int = 40,
        batch_size: int = 256,
        patience: int = 8,
        resume: bool = True,
    ) -> Dict:
        """
        Fits the model, resuming from a prior checkpoint automatically
        when one exists and resume=True. If X_val/y_val are given, also
        evaluates on them and returns the metrics dict (and logs
        everything to MLflow, same as BaseTrainer.train()).
        """

        initial_epoch = 0

        if resume and self.checkpoint_path.exists() and self.state_path.exists():
            logger.info(f"Resuming {self.model.name} training from checkpoint: {self.checkpoint_path}")
            self.model = keras.models.load_model(self.checkpoint_path)
            with open(self.state_path) as f:
                state = json.load(f)
            initial_epoch = state.get("last_completed_epoch", -1) + 1

        if initial_epoch >= max_epochs:
            logger.info(
                f"Checkpoint already at epoch {initial_epoch} >= "
                f"max_epochs={max_epochs} — nothing left to train."
            )
        else:
            validation_data = (
                (X_val, y_val) if X_val is not None and y_val is not None else None
            )

            callbacks = [
                keras.callbacks.EarlyStopping(
                    monitor="val_loss" if validation_data is not None else "loss",
                    patience=patience,
                    restore_best_weights=True,
                ),
                keras.callbacks.ModelCheckpoint(
                    filepath=str(self.checkpoint_path),
                    save_best_only=False,
                ),
                _EpochStateCallback(self.state_path),
            ]

            self.checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

            logger.info(
                f"Training {self.model.name}: epochs={initial_epoch}->{max_epochs}, "
                f"batch_size={batch_size}, patience={patience}."
            )

            start = time.time()
            self.model.fit(
                X_train,
                y_train,
                validation_data=validation_data,
                epochs=max_epochs,
                initial_epoch=initial_epoch,
                batch_size=batch_size,
                callbacks=callbacks,
                verbose=2,
            )
            logger.info(f"Training completed in {time.time() - start:.2f}s.")

        metrics = {}
        if X_val is not None and y_val is not None:
            predictions = self.predict(X_val)
            metrics = self.evaluator.evaluate(y_val, predictions)

        if self.track_mlflow and self.tracker is not None:
            try:
                self.last_run_id = self.tracker.log_run(
                    run_name=self.run_name,
                    params={
                        "window_size": X_train.shape[1],
                        "n_features": X_train.shape[2],
                        "max_epochs": max_epochs,
                        "batch_size": batch_size,
                        "patience": patience,
                    },
                    metrics=metrics,
                    model=self.model,
                    model_flavor="keras",
                    tags=self.tags,
                )
                logger.info(f"Logged to MLflow: run_id={self.last_run_id}")
            except Exception as exc:
                logger.warning(f"MLflow logging failed ({exc}) — model was still trained.")

        return metrics

    def predict(self, X: np.ndarray) -> np.ndarray:

        logger.info(f"Generating predictions using {self.model.name}...")

        return self.model.predict(X, verbose=0).flatten()

    def save(self, path: Union[str, Path]) -> None:

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        self.model.save(path)

        logger.info(f"Model saved to {path}")

    @staticmethod
    def load(path: Union[str, Path], track_mlflow: bool = False) -> "DLTrainer":

        logger.info(f"Loading model from {path}")

        model = keras.models.load_model(path)

        # Loading is inference, not a training run — tracking defaults
        # off, same convention as BaseTrainer.load().
        return DLTrainer(model, checkpoint_path=path, track_mlflow=track_mlflow)
