#!/usr/bin/env bash
# Real-data cluster visualization (FastAPI + React from rlviz/).
#   bash scripts/run_viz.sh                 # -> http://localhost:8000
# Env overrides: PORT=8000  PARTS=0-3  VIZ_WARMUP=48  VIZ_CONTAINERS=140
set -euo pipefail
cd "$(dirname "$0")/.."
ROOT="$(pwd)"

PY="$ROOT/.venv/bin/python"
PORT="${PORT:-8000}"

if [[ ! -x "$PY" ]]; then
  echo "==> creating .venv"
  python3 -m venv .venv
fi
if [[ requirements.txt -nt .venv/.req_stamp || rlviz/backend/requirements.txt -nt .venv/.req_stamp ]]; then
  echo "==> installing dependencies"
  "$PY" -m pip install -r requirements.txt
  "$PY" -m pip install -r rlviz/backend/requirements.txt
  touch .venv/.req_stamp
fi

if [[ ! -s data/processed/smoke/instances.parquet ]]; then
  echo "==> ensuring real data"
  bash scripts/download_data.sh "${PARTS:-0-3}"
  PYTHONPATH=src "$PY" -m cluster_rl.data.preprocess --parts "${PARTS:-0-3}" --out data/processed/smoke
fi

echo "==> building frontend"
cd rlviz/frontend
[[ -d node_modules ]] || npm install
npm run build
cd ../..

echo "==> open http://localhost:${PORT}"
cd rlviz/backend
exec "$PY" -m uvicorn main:app --port "$PORT"