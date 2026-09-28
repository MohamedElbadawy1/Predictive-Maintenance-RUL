"""
Pipeline/train_gru.py — same protocol as Pipeline/train_lstm.py (raw
25 features, cap=150, window_size=30, same train/val split, same
official-test-set evaluation), swapping only the recurrent cell.

Imports prepare_lstm_sequences() and RAW_FEATURE_COLUMNS from
src/deep_learning/data_prep.py (not from Pipeline/train_lstm.py --
that cross-script import was fragile on Windows). Identical
preprocessing via a shared src/ module is what makes the LSTM-vs-GRU
comparison fair.

Not registry-eligible, same reasoning as train_lstm.py: this logs an
MLflow comparison run only, it never touches the "champion" alias.

The checkpoint filename includes the seed, same as train_lstm.py, so
different seeds (or an old run) never collide.

Usage:
    python Pipeline/train_gru.py                    # seed=42 (RANDOM_STATE), full run
    python Pipeline/train_gru.py --seed 7            # a different seed, own checkpoint
    python Pipeline/train_gru.py --no-resume         # restart this seed's run at epoch 0
"""
import argparse
import sys
from pathlib import Path

from tensorflow import keras

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config.config import DEFAULT_RUL_CAP, DEFAULT_WINDOW_SIZE, MODELS_DIR, RANDOM_STATE
from src.deep_learning.data_prep import prepare_lstm_sequences, RAW_FEATURE_COLUMNS
from src.deep_learning.gru_model import build_gru_baseline
from src.deep_learning.dl_trainer import DLTrainer
from src.evaluation.evaluator import RegressionEvaluator

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--window-size", type=int, default=DEFAULT_WINDOW_SIZE)
    parser.add_argument("--max-epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--no-regime-aware", action="store_true", help="Disable regime-aware normalization")
    parser.add_argument("--resume", action="store_true", default=True, help="Resume this seed's checkpoint if one exists (default)")
    parser.add_argument("--no-resume", dest="resume", action="store_false", help="Ignore any existing checkpoint for this seed and restart at epoch 0")
    parser.add_argument("--seed", type=int, default=RANDOM_STATE, help="Random seed for weight init and batch shuffling (reproducibility)")
    args = parser.parse_args()

    keras.utils.set_random_seed(args.seed)

    checkpoint_path = MODELS_DIR / f"gru_baseline_seed{args.seed}.keras"

    data = prepare_lstm_sequences(window_size=args.window_size, regime_aware=not args.no_regime_aware)

    model = build_gru_baseline(window_size=args.window_size, n_features=len(RAW_FEATURE_COLUMNS))

    trainer = DLTrainer(
        model,
        checkpoint_path=checkpoint_path,
        run_name="gru_baseline",
        tags={
            "model_family": "gru",
            "feature_set": "raw_25",
            "rul_cap": str(DEFAULT_RUL_CAP),
            "normalization": "regime_aware" if not args.no_regime_aware else "global",
            "seed": str(args.seed),
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
    test_metrics = RegressionEvaluator().evaluate(data["y_test_true"], test_preds)

    trainer.save(MODELS_DIR / f"gru_baseline_seed{args.seed}_final.keras")

    print("\n" + "=" * 60)
    print(f"GRU BASELINE RESULT (seed={args.seed})")
    print("=" * 60)
    print(f"Validation metrics : {val_metrics}")
    print(f"Test metrics       : {test_metrics}")
    print(f"MLflow run_id      : {trainer.last_run_id}")
    print("Not registered as champion — logged as a comparison run only.")
