"""
pipeline/train_with_tuning.py — full pipeline from raw data ingestion,
WITH a real Optuna hyperparameter search.

Thin CLI wrapper around TrainingPipeline.run() (src/training/pipeline.py)
— the actual logic lives there so it's callable from anywhere (a script,
a notebook, eventually a FastAPI background task), not duplicated here.

Use this when: you want to find the best hyperparameters fresh (new
data might have shifted what's optimal), and are willing to pay the
time cost of a real search (each trial trains a full model).

For periodic retraining where you just want to confirm the ALREADY-KNOWN
best configuration still holds on new data, without re-searching, use
pipeline/train_with_best_params.py instead — much faster.

Usage:
    python pipeline/train_with_tuning.py --n-trials 20
    python pipeline/train_with_tuning.py --n-trials 20 --models catboost xgboost
"""
import argparse
import sys

from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.training.pipeline import TrainingPipeline, REGISTRY_MODEL_NAME

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--n-trials", type=int, default=20, help="Optuna trials per model")
    parser.add_argument("--models", nargs="+", default=["catboost", "xgboost", "lightgbm"])
    parser.add_argument("--no-regime-aware", action="store_true", help="Disable regime-aware normalization")
    parser.add_argument("--registry-model-name", default=REGISTRY_MODEL_NAME)
    args = parser.parse_args()

    pipeline = TrainingPipeline(registry_model_name=args.registry_model_name)

    result = pipeline.run(
        n_trials=args.n_trials,
        model_names=args.models,
        regime_aware=not args.no_regime_aware,
    )

    print("\n" + "=" * 60)
    print("TRAINING RESULT")
    print("=" * 60)
    print(f"Winning model     : {result.winning_model_name}")
    print(f"Test metrics      : {result.winning_test_metrics}")
    print(f"Promoted          : {result.promoted}")
    print(f"Reason            : {result.promotion_reason}")
    print(f"Champion before   : {result.champion_metric_before}")
    print(f"Champion after    : {result.champion_metric_after}")
    print(f"Registered version: {result.registered_version}")
    print(f"Elapsed           : {result.elapsed_seconds}s")

    print("\nAll candidates:")
    for r in result.all_model_results:
        print(f"  {r['model']}: test {r['test_metrics']}")
