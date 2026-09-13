import sys
import uuid

from fastapi import APIRouter, BackgroundTasks, HTTPException

from api.dependencies import job_store, pipeline_cache
from api.schemas import FixedParamsTrainingRequest, TrainingJobStatus, TuningTrainingRequest
from src.exceptions.custom_exception import CustomException
from src.experiments.mlflow_tracker import MLflowTracker
from src.logger.logger import logger
from src.training.pipeline import TrainingPipeline

router = APIRouter()


def _cast_param(value: str):
    """MLflow stores every logged param as a string — cast back to the
    type a model constructor actually expects. Same logic as
    pipeline/train_with_best_params.py; kept in sync deliberately since
    both need it, not accidentally duplicated."""

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


def _result_to_dict(result) -> dict:

    return {
        "winning_model_name": result.winning_model_name,
        "winning_test_metrics": result.winning_test_metrics,
        "promoted": result.promoted,
        "promotion_reason": result.promotion_reason,
        "champion_metric_before": result.champion_metric_before,
        "champion_metric_after": result.champion_metric_after,
        "registered_version": result.registered_version,
        "elapsed_seconds": result.elapsed_seconds,
    }


def _run_tuning_job(job_id: str, n_trials: int, models: list, regime_aware: bool) -> None:

    try:
        pipeline = TrainingPipeline()
        result = pipeline.run(n_trials=n_trials, model_names=models, regime_aware=regime_aware)

        if result.promoted:
            pipeline_cache.reload()

        job_store.set_result(job_id, _result_to_dict(result))

    except Exception as exc:
        logger.error(f"Training job {job_id} failed: {exc}")
        job_store.set_error(job_id, str(exc))


def _run_fixed_params_job(job_id: str, regime_aware: bool) -> None:

    try:
        tracker = MLflowTracker()
        training_pipeline = TrainingPipeline()

        champion_info = tracker.get_champion_run_info(training_pipeline.registry_model_name)
        if champion_info is None:
            raise CustomException(
                "No champion set yet — nothing to reuse hyperparameters from. "
                "Run a /train/tuning job first.",
                sys,
            )

        model_name = champion_info["model_family"]
        params = {k: _cast_param(v) for k, v in champion_info["params"].items()}

        result = training_pipeline.run_fixed_params(
            model_name=model_name, params=params, regime_aware=regime_aware,
        )

        if result.promoted:
            pipeline_cache.reload()

        job_store.set_result(job_id, _result_to_dict(result))

    except Exception as exc:
        logger.error(f"Training job {job_id} failed: {exc}")
        job_store.set_error(job_id, str(exc))


@router.post("/tuning", response_model=TrainingJobStatus, status_code=202)
def start_tuning_job(
    request: TuningTrainingRequest,
    background_tasks: BackgroundTasks,
) -> TrainingJobStatus:
    """
    Full pipeline from raw data, WITH a real Optuna hyperparameter
    search — takes minutes, so this returns immediately with a job_id
    to poll via GET /train/status/{job_id} rather than blocking the
    request.
    """

    job_id = str(uuid.uuid4())
    job_store.create(job_id)

    background_tasks.add_task(
        _run_tuning_job, job_id, request.n_trials, request.models, request.regime_aware,
    )

    return TrainingJobStatus(job_id=job_id, status="running")


@router.post("/best-params", response_model=TrainingJobStatus, status_code=202)
def start_fixed_params_job(
    request: FixedParamsTrainingRequest,
    background_tasks: BackgroundTasks,
) -> TrainingJobStatus:
    """
    Full pipeline from raw data, reusing the current champion's
    already-known hyperparameters — no search. Still runs as a
    background job (still real data-pipeline + training time), just
    faster than /train/tuning.
    """

    job_id = str(uuid.uuid4())
    job_store.create(job_id)

    background_tasks.add_task(_run_fixed_params_job, job_id, request.regime_aware)

    return TrainingJobStatus(job_id=job_id, status="running")


@router.get("/status/{job_id}", response_model=TrainingJobStatus)
def get_job_status(job_id: str) -> TrainingJobStatus:

    job = job_store.get(job_id)

    if job is None:
        raise HTTPException(status_code=404, detail=f"No job found with id '{job_id}'.")

    return TrainingJobStatus(job_id=job_id, **job)
