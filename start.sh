#!/bin/sh
# start.sh
#
# Launches FastAPI (backend) and Streamlit (frontend) inside a single
# Hugging Face Spaces Docker container.
#
# Hugging Face Spaces only exposes one port (default 7860) and runs one
# CMD, so the backend is started first as a background process on
# BACKEND_PORT (default 8000, internal only), then the Streamlit frontend
# is started in the background on FRONTEND_PORT (7860), which HF Spaces
# routes external traffic to. A polling loop then supervises both PIDs,
# since the base image's /bin/sh (dash) does not support `wait -n`.

set -e

BACKEND_PORT="${BACKEND_PORT:-8000}"
FRONTEND_PORT="${FRONTEND_PORT:-7860}"

echo "[start.sh] Starting FastAPI backend on port ${BACKEND_PORT}..."
uvicorn src.api.main:app --host 0.0.0.0 --port "${BACKEND_PORT}" &
BACKEND_PID=$!

echo "[start.sh] Waiting for backend /health to respond..."
for i in $(seq 1 30); do
  if curl -sf "http://127.0.0.1:${BACKEND_PORT}/health" > /dev/null 2>&1; then
    echo "[start.sh] Backend is healthy."
    break
  fi
  sleep 1
done

echo "[start.sh] Starting Streamlit frontend on port ${FRONTEND_PORT}..."
export API_BASE_URL="http://127.0.0.1:${BACKEND_PORT}"

streamlit run app.py \
  --server.port="${FRONTEND_PORT}" \
  --server.address=0.0.0.0 \
  --server.headless=true \
  --server.enableXsrfProtection=false \
  --server.enableCORS=false \
  --server.maxUploadSize=200 \
  --browser.gatherUsageStats=false &
FRONTEND_PID=$!

echo "[start.sh] Supervising backend (PID ${BACKEND_PID}) and frontend (PID ${FRONTEND_PID})..."

# dash's `wait` has no -n flag, so poll both PIDs instead.
# If either process dies, bring the whole container down so the
# orchestrator (HF Spaces) restarts it cleanly.
while true; do
  if ! kill -0 "${BACKEND_PID}" 2>/dev/null; then
    echo "[start.sh] Backend process exited unexpectedly."
    EXIT_CODE=1
    break
  fi

  if ! kill -0 "${FRONTEND_PID}" 2>/dev/null; then
    echo "[start.sh] Frontend process exited unexpectedly."
    EXIT_CODE=1
    break
  fi

  sleep 2
done

echo "[start.sh] Shutting down container (exit code ${EXIT_CODE})."
kill "${BACKEND_PID}" "${FRONTEND_PID}" 2>/dev/null || true
exit "${EXIT_CODE}"
