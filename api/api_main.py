from contextlib import asynccontextmanager

from fastapi import FastAPI

from api.routes import predict, train
from src.logger.logger import logger


@asynccontextmanager
async def lifespan(app: FastAPI):

    logger.info("Starting Predictive Maintenance RUL API...")
    yield
    logger.info("Shutting down Predictive Maintenance RUL API.")


app = FastAPI(
    title="Predictive Maintenance RUL API",
    description=(
        "Predicts remaining useful life for turbofan engines using the "
        "backend selected by MODEL_BACKEND (default 'ensemble' -- the "
        "LSTM+GRU average, this project's best offline result; 'catboost' "
        "serves the MLflow registry's tabular 'champion' instead), and "
        "exposes endpoints to trigger retraining (with or without a fresh "
        "hyperparameter search)."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

app.include_router(predict.router, prefix="/predict", tags=["Prediction"])
app.include_router(train.router, prefix="/train", tags=["Training"])


@app.get("/health", tags=["Health"])
def health() -> dict:
    """Basic liveness check — does not depend on the model being loaded,
    so this responds even before the first prediction request triggers
    the (lazy) MLflow load."""

    return {"status": "ok"}


@app.get("/health/ready", tags=["Health"])
def readiness() -> dict:
    """Readiness check that DOES reflect whether the model is loaded —
    useful for orchestration (e.g. a load balancer) that wants to know
    "can this instance actually serve a prediction right now", not just
    "is the process alive"."""

    from api.dependencies import pipeline_cache

    return {"model_loaded": pipeline_cache.is_loaded()}
