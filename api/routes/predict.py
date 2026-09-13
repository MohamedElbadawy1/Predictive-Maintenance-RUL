import sys

import pandas as pd
from fastapi import APIRouter, HTTPException

from api.dependencies import pipeline_cache
from api.schemas import EnginePrediction, PredictionRequest, PredictionResponse
from src.exceptions.custom_exception import CustomException

router = APIRouter()


@router.post("/", response_model=PredictionResponse)
def predict(request: PredictionRequest) -> PredictionResponse:
    """
    Predict remaining useful life for one or more engines, using
    whichever model is currently the MLflow "champion" — model,
    scaler, regime normalizer, and feature list all sourced from
    MLflow (see pipeline/predict.py), not local files.
    """

    raw_df = pd.DataFrame([reading.model_dump() for reading in request.readings])

    pipeline = pipeline_cache.get()

    try:
        result_df = pipeline.predict(raw_df)
    except CustomException as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    predictions = [
        EnginePrediction(
            unit_number=int(row["unit_number"]),
            predicted_RUL=float(row["predicted_RUL"]),
            n_cycles_seen=int(row["n_cycles_seen"]),
            short_history_warning=bool(row["short_history_warning"]),
            champion_version=str(row["champion_version"]),
            regime=int(row["regime"]) if "regime" in result_df.columns else None,
        )
        for _, row in result_df.iterrows()
    ]

    return PredictionResponse(predictions=predictions, model_version=str(pipeline.version))


@router.post("/reload")
def reload_champion() -> dict:
    """
    Force the cached pipeline to reload from MLflow — call this after a
    training job promotes a new champion, so predictions immediately
    start using it without needing to restart the API process.
    """

    try:
        pipeline = pipeline_cache.reload()
    except CustomException as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    return {"status": "reloaded", "champion_version": pipeline.version}
