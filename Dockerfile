# Uses uv (this project's real dependency manager — pyproject.toml +
# uv.lock, confirmed from "Detected uv project" in this project's own
# MLflow logs) instead of pip, for fast, fully-reproducible installs
# pinned to uv.lock rather than requirements.txt's loose >= ranges.
#
# Single shared image for both services (api, frontend) in
# docker-compose.yml — same dependencies either way, so one image with
# a different `command:` per service is simpler than two near-
# identical Dockerfiles.
FROM python:3.12-slim

# libgomp1: required at runtime by catboost/lightgbm (OpenMP) — the
# container fails at import time without it, not at build time, so
# it's an easy one to miss until someone actually calls /predict.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Installed from PyPI, not copied from ghcr.io/astral-sh/uv — some
# networks (corporate proxies, certain ISPs) block ghcr.io specifically
# while PyPI works fine, which is exactly what happened building this
# image the first time.
#
# --timeout/--retries: this image has been built at least once over a
# very slow connection (~50 KB/s observed), slow enough that pip's
# default read timeout tripped on a 20 MB wheel. Generous values here
# cost nothing on a fast connection and avoid a full rebuild on a slow
# one.
RUN pip install --no-cache-dir --timeout 300 --retries 5 uv

WORKDIR /app

ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy UV_HTTP_TIMEOUT=300

# Install dependencies before copying the rest of the app — this layer
# only rebuilds when pyproject.toml/uv.lock change, not on every code
# edit, so most rebuilds skip straight to the fast COPY below.
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-install-project

COPY . .
RUN uv sync --frozen

ENV PATH="/app/.venv/bin:$PATH"

EXPOSE 8000 8501

# Overridden per-service by docker-compose.yml's `command:` — this
# default only matters if the image is run standalone.
CMD ["uv", "run", "uvicorn", "api.api_main:app", "--host", "0.0.0.0", "--port", "8000"]