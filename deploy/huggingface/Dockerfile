# deploy/huggingface/Dockerfile — Hugging Face Spaces variant of the
# project's own Dockerfile (see the repo root's Dockerfile for the
# local docker-compose setup, which this deliberately mirrors as
# closely as possible so the two never drift far apart).
#
# Two real differences from the root Dockerfile, both forced by how a
# free HF Space works:
#   1. A free Space's filesystem is EPHEMERAL -- it does not persist
#      across restarts/rebuilds the way docker-compose's `artifacts:`
#      bind mount does. So this image COPIES a pre-trained
#      artifacts/models/ directory (built by
#      `python Pipeline/train_ensemble.py` before you deploy) straight
#      into the image, instead of expecting to train after the
#      container starts. See deploy/huggingface/README.md for the
#      exact steps.
#   2. A Space exposes exactly one public port. deploy/huggingface/
#      start.sh runs the FastAPI service internally on 127.0.0.1:8000
#      and the Streamlit frontend on the Space's declared app_port --
#      see that script for the full explanation.
#
# Build context for this Dockerfile MUST be the project root (it COPYs
# pyproject.toml, uv.lock, api/, src/, frontend/, artifacts/ from
# there) -- see README.md's "How to deploy" for the exact `docker
# build` / HF Space setup this expects.
FROM python:3.12-slim

RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir --timeout 300 --retries 5 uv

WORKDIR /app

ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy UV_HTTP_TIMEOUT=300

COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-install-project

COPY api/ ./api/
COPY src/ ./src/
COPY frontend/ ./frontend/

# The pre-trained models this image will actually serve -- see this
# file's header comment. NOT `COPY . .`: intentionally excludes
# data/raw (not needed at serving time, and this keeps the image
# small), notebooks/, docs/, tests/, reports/, catboost_info/, and any
# artifacts/mlruns/ local MLflow store (irrelevant to the ensemble
# backend and, unlike the two files below, not something you want a
# public Space image built from).
COPY artifacts/models/*.keras ./artifacts/models/
COPY artifacts/models/ensemble/ ./artifacts/models/ensemble/

RUN uv sync --frozen

ENV PATH="/app/.venv/bin:$PATH"
ENV MODEL_BACKEND=ensemble

COPY deploy/huggingface/start.sh ./start.sh
RUN chmod +x ./start.sh

# Hugging Face sets $PORT itself and routes public traffic to it;
# 7860 is just the conventional default a Space's README.md declares
# via app_port when $PORT isn't set some other way.
EXPOSE 7860

CMD ["./start.sh"]
