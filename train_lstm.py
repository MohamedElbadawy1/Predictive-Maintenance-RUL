"""
Resumable LSTM baseline training (Sprint 12).

Runs in small epoch batches across separate process invocations (same
pattern as Sprint 11's run_optuna_search.py), checkpointing the model
and training state to disk after every epoch so progress survives
across calls. Early stopping is tracked manually in the state file so
it also survives resumption.

Usage:
    python train_lstm.py --feature-set raw --n-epochs 5   # run up to 5 more epochs, exit
    python train_lstm.py --feature-set raw --n-epochs 5   # resumes from the checkpoint

--feature-set selects which prepare_lstm_sequences.py output to train on
("full" = 109 Sprint-10 features incl. lag/diff, "raw" = 25 true raw
features) and keeps each feature set's checkpoint/state/model separate,
so a "full" run and a "raw" run never clobber each other.
"""
import argparse
import json
import sys
import time

sys.path.insert(0, "/home/claude/Predictive-Maintenance-RUL")

import numpy as np
from tensorflow import keras

from src.config.config import ARTIFACTS_DIR, MODELS_DIR
from src.deep_learning.lstm_model import build_lstm_baseline
from src.deep_learning.dl_trainer import DLTrainer

MAX_EPOCHS = 40
PATIENCE = 8
BATCH_SIZE = 256
LSTM_UNITS = 64
DROPOUT_RATE = 0.2

parser = argparse.ArgumentParser()
parser.add_argument("--feature-set", choices=["full", "raw"], default="full")
parser.add_argument("--window-size", type=int, default=30)
parser.add_argument("--n-epochs", type=int, default=5)
args = parser.parse_args()

SEQUENCE_PATH = (
    ARTIFACTS_DIR / "data" / "sequences"
    / f"lstm_sequences_{args.feature_set}_w{args.window_size}.npz"
)
CHECKPOINT_PATH = MODELS_DIR / f"lstm_checkpoint_{args.feature_set}.keras"
STATE_PATH = MODELS_DIR / f"lstm_training_state_{args.feature_set}.json"

data = np.load(SEQUENCE_PATH)
X_train, y_train = data["X_train"], data["y_train"]
X_val, y_val = data["X_val"], data["y_val"]
window_size = int(data["window_size"])
n_features = int(data["n_features"])

MODELS_DIR.mkdir(parents=True, exist_ok=True)

if STATE_PATH.exists() and CHECKPOINT_PATH.exists():
    print("Resuming from checkpoint...")
    model = keras.models.load_model(CHECKPOINT_PATH)
    state = json.loads(STATE_PATH.read_text())
else:
    print("No checkpoint found — starting fresh.")
    model = build_lstm_baseline(
        window_size=window_size,
        n_features=n_features,
        lstm_units=LSTM_UNITS,
        dropout_rate=DROPOUT_RATE,
    )
    state = {
        "epoch": 0,
        "best_val_loss": None,
        "epochs_no_improve": 0,
        "stopped_early": False,
        "history": [],
    }

if state["stopped_early"]:
    print(f"Training already stopped early at epoch {state['epoch']} "
          f"(best val_loss={state['best_val_loss']:.4f}). Nothing to do.")
    sys.exit(0)

if state["epoch"] >= MAX_EPOCHS:
    print(f"Already reached MAX_EPOCHS={MAX_EPOCHS}. Nothing to do.")
    sys.exit(0)


class CheckpointAndLogCallback(keras.callbacks.Callback):
    """Save model + state after every epoch, and apply manual early
    stopping that survives across process restarts."""

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        state["epoch"] += 1

        val_loss = logs.get("val_loss")
        record = {
            "epoch": state["epoch"],
            "loss": logs.get("loss"),
            "mae": logs.get("mae"),
            "val_loss": val_loss,
            "val_mae": logs.get("val_mae"),
        }
        state["history"].append(record)

        if state["best_val_loss"] is None or val_loss < state["best_val_loss"]:
            state["best_val_loss"] = val_loss
            state["epochs_no_improve"] = 0
        else:
            state["epochs_no_improve"] += 1

        self.model.save(CHECKPOINT_PATH)
        STATE_PATH.write_text(json.dumps(state, indent=2))

        print(f"Epoch {state['epoch']}: loss={record['loss']:.4f} "
              f"mae={record['mae']:.4f} val_loss={val_loss:.4f} "
              f"val_mae={record['val_mae']:.4f} "
              f"(no_improve={state['epochs_no_improve']}/{PATIENCE})",
              flush=True)

        if state["epochs_no_improve"] >= PATIENCE:
            print(f"Early stopping triggered at epoch {state['epoch']}.")
            state["stopped_early"] = True
            STATE_PATH.write_text(json.dumps(state, indent=2))
            self.model.stop_training = True

        if state["epoch"] >= MAX_EPOCHS:
            self.model.stop_training = True


target_epoch = min(state["epoch"] + args.n_epochs, MAX_EPOCHS)

print(f"Training epochs {state['epoch']+1} to {target_epoch} "
      f"(max={MAX_EPOCHS}, patience={PATIENCE})...")

start = time.time()
model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    initial_epoch=state["epoch"],
    epochs=target_epoch,
    batch_size=BATCH_SIZE,
    callbacks=[CheckpointAndLogCallback()],
    verbose=0,
)
elapsed = time.time() - start

print(f"Batch took {elapsed:.1f}s ({elapsed/max(1,(target_epoch-state['epoch']+ (target_epoch-state['epoch']==0))):.1f}s/epoch approx)")
print(f"Current epoch: {state['epoch']} / {MAX_EPOCHS} | "
      f"best val_loss so far: {state['best_val_loss']:.4f} | "
      f"stopped_early: {state['stopped_early']}")
