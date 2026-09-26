#!/usr/bin/env bash
# deploy/huggingface/start.sh — runs both services in one container, the
# way a free Hugging Face Space needs (Spaces expose exactly one public
# port). docker-compose.yml keeps them as two separate containers for
# local dev; this script is the single-container equivalent for Spaces
# only -- it changes nothing about api/ or frontend/ themselves.
#
# - FastAPI (api/api_main.py) starts first, in the background, on
#   localhost:8000 -- never exposed publicly by the Space itself.
# - Streamlit (frontend/app.py) runs in the foreground on
#   0.0.0.0:${PORT:-7860} -- the port the Space's README.md declares
#   via `app_port`, and the only one Hugging Face routes traffic to.
# - API_URL is hard-set to the internal address so the frontend finds
#   the API without any extra configuration on the Space's side.
set -euo pipefail

export API_URL="http://127.0.0.1:8000"

echo "Starting FastAPI on 127.0.0.1:8000 (internal only)..."
uvicorn api.api_main:app --host 127.0.0.1 --port 8000 &
API_PID=$!

# Give the API a few seconds' head start so the frontend's first
# health check (in its sidebar) doesn't race an empty connection --
# purely cosmetic, /predict itself would still work either order since
# PipelineCache loads lazily on first request.
sleep 3

echo "Starting Streamlit on 0.0.0.0:${PORT:-7860}..."
streamlit run frontend/app.py \
    --server.address 0.0.0.0 \
    --server.port "${PORT:-7860}" \
    --server.headless true \
    --browser.gatherUsageStats false &
FRONTEND_PID=$!

# If either process dies, bring the whole container down so Hugging
# Face's health checks / restart policy notice, instead of silently
# running with only one of the two services alive.
wait -n "$API_PID" "$FRONTEND_PID"
exit $?
