"""
Pipeline/train_lstm.py — raw data -> sequences -> LSTM baseline ->
validation + official test-set evaluation, logged to MLflow as a
comparison run (NOT registered as champion — see the note below).

Thin CLI wrapper, same convention as train_with_tuning.py /
train_with_best_params.py: the script itself only parses args, prints
the result, and orchestrates calls into src/ — the data prep and model
logic live in src/deep_learning/, not here.

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

The checkpoint filename includes the seed (e.g. lstm_baseline_seed42.
keras) so that runs with different seeds never resume from each
other's (or an old unseeded run's) checkpoint by accident.

Usage:
    python Pipeline/train_lstm.py                    # seed=42 (RANDOM_STATE), full run
    python Pipeline/train_lstm.py --seed 7            # a different seed, own checkpoint
    python Pipeline/train_lstm.py --no-resume         # restart this seed's run at epoch 0
"""
import argparse
import sys
from pathlib import Path

from tensorflow import keras

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config.config import DEFAULT_RUL_CAP, DEFAULT_WINDOW_SIZE, MODELS_DIR, RANDOM_STATE
from src.deep_learning.data_prep import prepare_lstm_sequences, RAW_FEATURE_COLUMNS
from src.deep_learning.lstm_model import build_lstm_baseline
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

    checkpoint_path = MODELS_DIR / f"lstm_baseline_seed{args.seed}.keras"

    data = prepare_lstm_sequences(window_size=args.window_size, regime_aware=not args.no_regime_aware)

    model = build_lstm_baseline(window_size=args.window_size, n_features=len(RAW_FEATURE_COLUMNS))

    trainer = DLTrainer(
        model,
        checkpoint_path=checkpoint_path,
        run_name="lstm_baseline",
        tags={
            "model_family": "lstm",
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

    trainer.save(MODELS_DIR / f"lstm_baseline_seed{args.seed}_final.keras")

    print("\n" + "=" * 60)
    print(f"LSTM BASELINE RESULT (seed={args.seed})")
    print("=" * 60)
    print(f"Validation metrics : {val_metrics}")
    print(f"Test metrics       : {test_metrics}")
    print(f"MLflow run_id      : {trainer.last_run_id}")
    print("Not registered as champion — logged as a comparison run only.")
