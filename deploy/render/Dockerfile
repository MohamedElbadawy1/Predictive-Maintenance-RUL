# deploy/render/Dockerfile — Render free-tier variant. Two things set
# this apart from the repo root's Dockerfile (which is for local
# docker-compose) and from deploy/huggingface/Dockerfile:
#
#   1. Installs from requirements-serving.txt, not requirements.txt /
#      pyproject.toml -- no tensorflow, no mlflow/catboost/xgboost/
#      lightgbm/optuna. Those are training-only dependencies the
#      MODEL_BACKEND=ensemble serving path never imports (see
#      src/pipelines/ensemble_inference_pipeline.py's module
#      docstring), and skipping them is most of the difference between
#      the ~700MB RSS a tensorflow-cpu process needs at idle and the
#      ~240MB this image measured under the same load -- comfortably
#      inside Render's free-tier 512MB limit, where the former was not.
#   2. Only copies the ONNX ensemble artifacts (artifacts/models/
#      ensemble/), not the original .keras checkpoints -- the serving
#      path never reads those directly.
#
# Build context for this Dockerfile MUST be the project root (Render's
# "Docker Build Context Directory" setting) -- see README.md's
# "Deploying to Render" for the exact service settings this expects.
FROM python:3.12-slim

WORKDIR /app

COPY requirements-serving.txt ./
RUN pip install --no-cache-dir --timeout 300 --retries 5 -r requirements-serving.txt

COPY api/ ./api/
COPY src/ ./src/
COPY frontend/ ./frontend/

# The pre-trained (and pre-exported-to-ONNX) ensemble this image
# actually serves -- see Pipeline/train_ensemble.py's docstring for
# how these five files are produced from the two Keras checkpoints.
# NOT `COPY . .`: no data/raw, no notebooks/, no docs/, no tests/, no
# artifacts/mlruns/ local MLflow store, and no .keras checkpoints
# either -- keeps the image lean and avoids shipping anything this
# serving path doesn't read.
COPY artifacts/models/ensemble/ ./artifacts/models/ensemble/

ENV MODEL_BACKEND=ensemble

COPY deploy/render/start.sh ./start.sh
RUN chmod +x ./start.sh

# Render sets $PORT itself and routes public traffic to it; start.sh
# reads it (falling back to 10000, Render's own documented default,
# if unset).
CMD ["./start.sh"]
