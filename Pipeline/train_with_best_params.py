"""
pipeline/train_with_best_params.py — full pipeline from raw data
ingestion, but retrains using the CURRENT CHAMPION's already-known
hyperparameters instead of running a fresh Optuna search.

Use this for periodic/scheduled retraining on new data: you want to
confirm the existing best-known configuration still performs well (data
may have drifted), without paying the cost of a full hyperparameter
search every time. If it's still the best, it gets promoted as the new
champion (replacing the old run with a fresher one trained on more
recent data, same hyperparameters); if not, nothing changes.

For a fresh hyperparameter search (e.g. after a real architecture or
feature change), use pipeline/train_with_tuning.py instead.

Usage:
    python pipeline/train_with_best_params.py
"""
import argparse
import sys

sys.path.insert(0, "/home/claude/Predictive-Maintenance-RUL")

from src.training.pipeline import TrainingPipeline, REGISTRY_MODEL_NAME
from src.experiments.mlflow_tracker import MLflowTracker
from src.exceptions.custom_exception import CustomException


def _cast_param(value: str):
    """
    MLflow logs every param as a string. Cast back to the type the
    model constructor actually expects — int if it parses cleanly as
    one, else float, else a real bool for "True"/"False", else leave as
    the original string (covers things like loss_function="RMSE").
    """

    if value in ("True", "False"):
        return value == "True"

    try:
        return int(value)
    except ValueError:
        pass

    try:
        return float(value)
    except ValueError:
        pass

    return value


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--no-regime-aware", action="store_true", help="Disable regime-aware normalization")
    parser.add_argument("--registry-model-name", default=REGISTRY_MODEL_NAME)
    args = parser.parse_args()

    tracker = MLflowTracker()
    champion_info = tracker.get_champion_run_info(args.registry_model_name)

    if champion_info is None:
        raise CustomException(
            f"No champion set for '{args.registry_model_name}' yet — there's "
            "nothing to reuse the params of. Run pipeline/train_with_tuning.py "
            "first, or bootstrap an existing run per "
            "docs/Sprint_20_Training_Pipeline_MLflow_Registry.md.",
            sys,
        )

    model_name = champion_info["model_family"]
    params = {k: _cast_param(v) for k, v in champion_info["params"].items()}

    print(f"Reusing champion's params (version {champion_info['version']}, "
          f"model_family={model_name}): {params}")

    pipeline = TrainingPipeline(registry_model_name=args.registry_model_name)
    result = pipeline.run_fixed_params(
        model_name=model_name,
        params=params,
        regime_aware=not args.no_regime_aware,
    )

    print("\n" + "=" * 60)
    print("RETRAIN RESULT (fixed params, no search)")
    print("=" * 60)
    print(f"Model             : {result.winning_model_name}")
    print(f"Test metrics      : {result.winning_test_metrics}")
    print(f"Promoted          : {result.promoted}")
    print(f"Reason            : {result.promotion_reason}")
    print(f"Champion before   : {result.champion_metric_before}")
    print(f"Champion after    : {result.champion_metric_after}")
    print(f"Registered version: {result.registered_version}")
    print(f"Elapsed           : {result.elapsed_seconds}s")
