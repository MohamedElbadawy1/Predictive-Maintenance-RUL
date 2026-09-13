from typing import Dict, List, Optional

from pydantic import BaseModel, Field


class EngineCycleReading(BaseModel):
    """One row: a single engine at a single cycle. A prediction request
    sends a full history per engine (multiple rows, one per cycle up to
    "now") — lag/rolling/diff features need the prior cycles."""

    unit_number: int
    time_in_cycles: int
    operational_setting_1: float
    operational_setting_2: float
    operational_setting_3: float
    sensor_1: float
    sensor_2: float
    sensor_3: float
    sensor_4: float
    sensor_5: float
    sensor_6: float
    sensor_7: float
    sensor_8: float
    sensor_9: float
    sensor_10: float
    sensor_11: float
    sensor_12: float
    sensor_13: float
    sensor_14: float
    sensor_15: float
    sensor_16: float
    sensor_17: float
    sensor_18: float
    sensor_19: float
    sensor_20: float
    sensor_21: float


class PredictionRequest(BaseModel):

    readings: List[EngineCycleReading] = Field(
        ..., min_length=1,
        description="One row per (engine, cycle) — full history per engine up to 'now', not just the latest cycle.",
    )


class EnginePrediction(BaseModel):

    unit_number: int
    predicted_RUL: float
    n_cycles_seen: int
    short_history_warning: bool
    champion_version: str
    regime: Optional[int] = None


class PredictionResponse(BaseModel):

    predictions: List[EnginePrediction]
    model_version: str


class TuningTrainingRequest(BaseModel):

    n_trials: int = Field(20, ge=1, le=200, description="Optuna trials per model")
    models: List[str] = Field(default_factory=lambda: ["catboost", "xgboost", "lightgbm"])
    regime_aware: bool = True


class FixedParamsTrainingRequest(BaseModel):

    regime_aware: bool = True


class TrainingJobStatus(BaseModel):

    job_id: str
    status: str  # "running" | "completed" | "failed"
    result: Optional[Dict] = None
    error: Optional[str] = None
