#!/usr/bin/env bash
# deploy/render/start.sh — runs both services in one container, since
# a Render free-tier Web Service (like a free HF Space) exposes exactly
# one public port. See deploy/huggingface/start.sh for the identical
# reasoning; this is the same script, just reading Render's $PORT
# instead of a Space's.
#
# - FastAPI (api/api_main.py) starts first, in the background, on
#   localhost:8000 -- never exposed publicly.
# - Streamlit (frontend/app.py) runs in the foreground on
#   0.0.0.0:${PORT} -- Render sets $PORT itself and routes public
#   traffic to it.
set -euo pipefail

export API_URL="http://127.0.0.1:8000"

echo "Starting FastAPI on 127.0.0.1:8000 (internal only)..."
uvicorn api.api_main:app --host 127.0.0.1 --port 8000 &
API_PID=$!

sleep 3

echo "Starting Streamlit on 0.0.0.0:${PORT:-10000}..."
streamlit run frontend/app.py \
    --server.address 0.0.0.0 \
    --server.port "${PORT:-10000}" \
    --server.headless true \
    --browser.gatherUsageStats false &
FRONTEND_PID=$!

wait -n "$API_PID" "$FRONTEND_PID"
exit $?
