#!/usr/bin/env bash
# Launch the interactive Streamlit demo dashboard.
set -euo pipefail
cd "$(dirname "$0")/.."

PY="${PY:-.venv/bin/python}"
PORT="${PORT:-8501}"
HEADLESS="${HEADLESS:-true}"

echo "==> Open http://localhost:${PORT}  (over SSH: ssh -L ${PORT}:localhost:${PORT} <host>)"
PYTHONPATH=src "$PY" -m streamlit run ui/app.py \
  --server.port "$PORT" \
  --server.headless "$HEADLESS"
