"""
Pipeline/train_ensemble.py — train (or reuse) the LSTM and GRU
baselines, confirm their equal-weight average beats each of them
individually on the official test set, export both to ONNX, and
persist everything EnsembleInferencePipeline needs to serve that
average in production:

    artifacts/models/ensemble/
    ├── manifest.json           window_size, seed, feature columns,
    │                           whether regime normalization was used,
    │                           and the two ONNX filenames
    ├── feature_scaler.pkl      StandardScaler fit on RAW_FEATURE_COLUMNS
    ├── regime_normalizer.pkl   only written when regime_aware=True
    ├── lstm.onnx               ONNX export of the trained LSTM
    └── gru.onnx                ONNX export of the trained GRU

Serving uses ONNX Runtime rather than Keras/TensorFlow directly (see
EnsembleInferencePipeline's module docstring for why -- short version:
importing tensorflow costs ~500-600MB of RSS regardless of model size,
which alone exceeds the memory budget of most free hosting tiers;
onnxruntime with these two models loaded measured under 65MB). The
conversion path here (Keras -> tf.saved_model export -> tf2onnx CLI)
is deliberate, not the more direct `tf2onnx.convert.from_keras(model,
...)`: as of tf2onnx 1.17 / Keras 3, that direct path raises a
KeyError from a stale tensor-name lookup against Keras 3's renamed
internals. Exporting a plain SavedModel first and converting *that*
sidesteps the incompatibility entirely and is exactly the invocation
tf2onnx's own docs recommend for SavedModel-based conversion.

This is the ensemble counterpart of Pipeline/train_with_tuning.py /
train_with_best_params.py for CatBoost: those promote a champion into
MLflow; this promotes a champion pair into artifacts/models/ensemble/.
Nothing here touches the MLflow "champion" alias — the two serving
paths are independent, selected at request time by api/dependencies.py
via the MODEL_BACKEND environment variable (see README).

Per docs/Sprint_15_LSTM_Cap150_Test_Evaluation.md and this project's
README ("Current Best Model"), the LSTM+GRU equal-weight average is
the strongest result in this project (test MAE 16.39 vs CatBoost's
18.72-19.13) — this script is what turns that offline result into
something the API can actually serve.

Usage:
    # Trains LSTM and GRU from scratch (or resumes their checkpoints if
    # they already exist), then writes the serving artifacts.
    python Pipeline/train_ensemble.py

    # Reuse already-trained checkpoints for this seed without retraining
    python Pipeline/train_ensemble.py --skip-training

    python Pipeline/train_ensemble.py --seed 7 --n-trials-note "different seed"
"""
import argparse
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from tensorflow import keras

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config.config import (
    DEFAULT_RUL_CAP, DEFAULT_WINDOW_SIZE, MODELS_DIR, RANDOM_STATE,
    ENSEMBLE_DIR, ENSEMBLE_MANIFEST_PATH, ENSEMBLE_SCALER_PATH,
    ENSEMBLE_REGIME_NORMALIZER_PATH,
)
from src.deep_learning.data_prep import prepare_lstm_sequences, RAW_FEATURE_COLUMNS
from src.deep_learning.lstm_model import build_lstm_baseline
from src.deep_learning.gru_model import build_gru_baseline
from src.deep_learning.dl_trainer import DLTrainer
from src.evaluation.evaluator import RegressionEvaluator
from src.exceptions.custom_exception import CustomException
from src.logger.logger import logger


def _export_to_onnx(model: keras.Model, out_path: Path, opset: int = 13) -> None:
    """
    Keras 3 -> ONNX, via a plain tf.saved_model export rather than
    tf2onnx's `convert.from_keras(model, ...)` -- see this module's
    docstring for why the direct path fails against Keras 3.
    """
    with tempfile.TemporaryDirectory() as tmp_dir:
        saved_model_dir = Path(tmp_dir) / "saved_model"
        model.export(str(saved_model_dir))

        result = subprocess.run(
            [
                sys.executable, "-m", "tf2onnx.convert",
                "--saved-model", str(saved_model_dir),
                "--output", str(out_path),
                "--opset", str(opset),
            ],
            capture_output=True, text=True,
        )
        if result.returncode != 0:
            raise CustomException(
                f"tf2onnx failed converting {saved_model_dir} to {out_path}:\n{result.stderr}", sys,
            )
    logger.info(f"Exported ONNX model to {out_path} ({out_path.stat().st_size / 1024:.0f} KB)")

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--window-size", type=int, default=DEFAULT_WINDOW_SIZE)
    parser.add_argument("--max-epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--no-regime-aware", action="store_true", help="Disable regime-aware normalization")
    parser.add_argument("--seed", type=int, default=RANDOM_STATE)
    parser.add_argument(
        "--skip-training", action="store_true",
        help="Don't call trainer.train() at all -- load each seed's "
             "*_final.keras checkpoint as-is (fails clearly if either "
             "is missing). Use this to (re)build serving artifacts "
             "for models you already trained with train_lstm.py / "
             "train_gru.py.",
    )
    args = parser.parse_args()

    regime_aware = not args.no_regime_aware
    keras.utils.set_random_seed(args.seed)

    lstm_final_path = MODELS_DIR / f"lstm_baseline_seed{args.seed}_final.keras"
    gru_final_path = MODELS_DIR / f"gru_baseline_seed{args.seed}_final.keras"

    data = prepare_lstm_sequences(window_size=args.window_size, regime_aware=regime_aware)

    if args.skip_training:
        for path, name in [(lstm_final_path, "LSTM"), (gru_final_path, "GRU")]:
            if not path.exists():
                raise CustomException(
                    f"--skip-training was passed but {name} checkpoint not "
                    f"found at {path}. Train it first (Pipeline/train_lstm.py "
                    "/ train_gru.py) or drop --skip-training.", sys,
                )
        logger.info("Skipping training -- loading existing final checkpoints.")
        lstm_trainer = DLTrainer.load(lstm_final_path)
        gru_trainer = DLTrainer.load(gru_final_path)
    else:
        lstm_model = build_lstm_baseline(window_size=args.window_size, n_features=len(RAW_FEATURE_COLUMNS))
        lstm_trainer = DLTrainer(
            lstm_model,
            checkpoint_path=MODELS_DIR / f"lstm_baseline_seed{args.seed}.keras",
            run_name="lstm_baseline",
            tags={
                "model_family": "lstm", "feature_set": "raw_25",
                "rul_cap": str(DEFAULT_RUL_CAP),
                "normalization": "regime_aware" if regime_aware else "global",
                "seed": str(args.seed),
            },
        )
        lstm_trainer.train(
            data["X_train"], data["y_train"], data["X_val"], data["y_val"],
            max_epochs=args.max_epochs, batch_size=args.batch_size, patience=args.patience,
        )
        lstm_trainer.save(lstm_final_path)

        gru_model = build_gru_baseline(window_size=args.window_size, n_features=len(RAW_FEATURE_COLUMNS))
        gru_trainer = DLTrainer(
            gru_model,
            checkpoint_path=MODELS_DIR / f"gru_baseline_seed{args.seed}.keras",
            run_name="gru_baseline",
            tags={
                "model_family": "gru", "feature_set": "raw_25",
                "rul_cap": str(DEFAULT_RUL_CAP),
                "normalization": "regime_aware" if regime_aware else "global",
                "seed": str(args.seed),
            },
        )
        gru_trainer.train(
            data["X_train"], data["y_train"], data["X_val"], data["y_val"],
            max_epochs=args.max_epochs, batch_size=args.batch_size, patience=args.patience,
        )
        gru_trainer.save(gru_final_path)

    lstm_preds = lstm_trainer.predict(data["X_test"])
    gru_preds = gru_trainer.predict(data["X_test"])
    ensemble_preds = (lstm_preds + gru_preds) / 2

    evaluator = RegressionEvaluator()
    y_true = data["y_test_true"]
    lstm_metrics = evaluator.evaluate(y_true, lstm_preds)
    gru_metrics = evaluator.evaluate(y_true, gru_preds)
    ensemble_metrics = evaluator.evaluate(y_true, ensemble_preds)

    print("\n" + "=" * 60)
    print(f"ENSEMBLE BUILD RESULT (seed={args.seed}, window_size={args.window_size})")
    print("=" * 60)
    print(f"{'Model':<20}{'MAE':>10}{'RMSE':>10}{'R2':>10}")
    print(f"{'LSTM (solo)':<20}{lstm_metrics['MAE']:>10.2f}{lstm_metrics['RMSE']:>10.2f}{lstm_metrics['R2']:>10.3f}")
    print(f"{'GRU (solo)':<20}{gru_metrics['MAE']:>10.2f}{gru_metrics['RMSE']:>10.2f}{gru_metrics['R2']:>10.3f}")
    print(f"{'LSTM+GRU avg':<20}{ensemble_metrics['MAE']:>10.2f}{ensemble_metrics['RMSE']:>10.2f}{ensemble_metrics['R2']:>10.3f}")

    if not (ensemble_metrics["MAE"] <= lstm_metrics["MAE"] and ensemble_metrics["MAE"] <= gru_metrics["MAE"]):
        logger.warning(
            "The LSTM+GRU average did NOT beat both individual models on "
            "this run (this project's README documents ~±1 MAE point of "
            "seed-to-seed variance) -- serving artifacts are still written "
            "below since the API always serves the average, but you may "
            "want to compare against a different --seed before deploying."
        )

    # --- Persist everything EnsembleInferencePipeline needs -----------
    ENSEMBLE_DIR.mkdir(parents=True, exist_ok=True)

    data["scaler"].save(ENSEMBLE_SCALER_PATH)

    if regime_aware:
        data["regime_normalizer"].save(ENSEMBLE_REGIME_NORMALIZER_PATH)
    elif ENSEMBLE_REGIME_NORMALIZER_PATH.exists():
        # Stale file from a previous regime-aware build -- remove it so
        # EnsembleInferencePipeline doesn't load a normalizer that no
        # longer matches this manifest's regime_aware=False.
        ENSEMBLE_REGIME_NORMALIZER_PATH.unlink()

    lstm_onnx_path = ENSEMBLE_DIR / "lstm.onnx"
    gru_onnx_path = ENSEMBLE_DIR / "gru.onnx"
    _export_to_onnx(lstm_trainer.model, lstm_onnx_path)
    _export_to_onnx(gru_trainer.model, gru_onnx_path)

    manifest = {
        "seed": args.seed,
        "window_size": args.window_size,
        "regime_aware": regime_aware,
        "feature_columns": RAW_FEATURE_COLUMNS,
        "lstm_onnx": lstm_onnx_path.name,
        "gru_onnx": gru_onnx_path.name,
        "test_metrics": {"lstm": lstm_metrics, "gru": gru_metrics, "ensemble": ensemble_metrics},
    }
    with open(ENSEMBLE_MANIFEST_PATH, "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"\nServing artifacts written to {ENSEMBLE_DIR}/")
    print("The API will start serving this ensemble once MODEL_BACKEND=ensemble "
          "(the default) and the process is (re)started, or via POST /predict/reload.")
