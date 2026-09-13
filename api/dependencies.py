"""
Shared, process-wide state for the API — kept out of the route files so
routes stay thin (parse request, call logic, format response).
"""
import threading
from typing import Dict, Optional

from src.logger.logger import logger


class PipelineCache:
    """
    The model, scaler, regime normalizer, and feature list are loaded
    once from MLflow and reused across requests — reloading them per
    request would mean a slow round-trip to the tracking store on every
    single prediction. reload() lets a training endpoint force a
    refresh after promoting a new champion, without restarting the API
    process.

    Loaded lazily (on first request), not at startup — the API can come
    up and answer /health even if MLflow or the champion isn't ready
    yet; the first /predict call surfaces any loading problem clearly,
    rather than the whole process failing to start.
    """

    def __init__(self):
        self._pipeline = None
        self._lock = threading.Lock()

    def get(self):

        with self._lock:
            if self._pipeline is None:
                # Imported here, not at module load time, so importing
                # this module doesn't require MLflow to be reachable.
                from pipeline.predict import MLflowInferencePipeline
                logger.info("Loading MLflowInferencePipeline (first request)...")
                self._pipeline = MLflowInferencePipeline()
            return self._pipeline

    def reload(self):

        with self._lock:
            from pipeline.predict import MLflowInferencePipeline
            logger.info("Reloading MLflowInferencePipeline (champion may have changed)...")
            self._pipeline = MLflowInferencePipeline()
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
