"""
Shared, process-wide state for the API — kept out of the route files so
routes stay thin (parse request, call logic, format response).
"""
import os
import threading
from typing import Dict, Optional

from src.logger.logger import logger

# Which serving path /predict uses. "ensemble" (default): the LSTM+GRU
# equal-weight average -- this project's best offline result (README's
# "Current Best Model", test MAE 16.39 vs CatBoost's 18.72-19.13),
# served from artifacts/models/ensemble/ (see
# Pipeline/train_ensemble.py). "catboost": the previous default --
# the MLflow-registered tabular champion. Set once via environment
# variable, read at process start (not per-request), since switching
# backends is meant to be a deploy-time decision.
MODEL_BACKEND = os.environ.get("MODEL_BACKEND", "ensemble").strip().lower()

_SUPPORTED_BACKENDS = {"ensemble", "catboost"}
if MODEL_BACKEND not in _SUPPORTED_BACKENDS:
    raise ValueError(
        f"MODEL_BACKEND={MODEL_BACKEND!r} is not supported -- must be one "
        f"of {sorted(_SUPPORTED_BACKENDS)}."
    )


def _load_pipeline():
    """Imported lazily, not at module load time, so importing this
    module doesn't require MLflow or ensemble artifacts to already
    exist."""

    if MODEL_BACKEND == "ensemble":
        from src.pipelines.ensemble_inference_pipeline import EnsembleInferencePipeline
        return EnsembleInferencePipeline()

    from src.pipelines.inference_pipeline import MLflowInferencePipeline
    return MLflowInferencePipeline()


class PipelineCache:
    """
    The model and its preprocessing artifacts are loaded once (from
    MLflow, or from artifacts/models/ensemble/ -- see MODEL_BACKEND
    above) and reused across requests — reloading them per request
    would mean a slow round-trip on every single prediction. reload()
    lets a training endpoint force a refresh after promoting a new
    champion (or rebuilding the ensemble), without restarting the API
    process.

    Loaded lazily (on first request), not at startup — the API can come
    up and answer /health even if the backing store or artifacts aren't
    ready yet; the first /predict call surfaces any loading problem
    clearly, rather than the whole process failing to start.
    """

    def __init__(self):
        self._pipeline = None
        self._lock = threading.Lock()

    def get(self):

        with self._lock:
            if self._pipeline is None:
                logger.info(f"Loading {MODEL_BACKEND} inference pipeline (first request)...")
                self._pipeline = _load_pipeline()
            return self._pipeline

    def reload(self):

        with self._lock:
            logger.info(f"Reloading {MODEL_BACKEND} inference pipeline (artifacts may have changed)...")
            self._pipeline = _load_pipeline()
            return self._pipeline

    def is_loaded(self) -> bool:

        with self._lock:
            return self._pipeline is not None


class JobStore:
    """
    Minimal in-memory tracker for background training jobs — deliberately
    simple, not a claim of production-scale robustness. A real deployment
    handling meaningful load or requiring restart-durability should use a
    proper task queue (Celery + Redis, or similar); this is a lightweight,
    appropriately-scoped stand-in for where this project currently is.
    Jobs are lost if the API process restarts.
    """

    def __init__(self):
        self._jobs: Dict[str, dict] = {}
        self._lock = threading.Lock()

    def create(self, job_id: str) -> None:

        with self._lock:
            self._jobs[job_id] = {"status": "running", "result": None, "error": None}

    def set_result(self, job_id: str, result: dict) -> None:

        with self._lock:
            self._jobs[job_id] = {"status": "completed", "result": result, "error": None}

    def set_error(self, job_id: str, error: str) -> None:

        with self._lock:
            self._jobs[job_id] = {"status": "failed", "result": None, "error": error}

    def get(self, job_id: str) -> Optional[dict]:

        with self._lock:
            return self._jobs.get(job_id)


pipeline_cache = PipelineCache()
job_store = JobStore()
