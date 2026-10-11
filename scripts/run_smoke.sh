#!/usr/bin/env bash
# End-to-end Phase 1 smoke: download real Borg 2011 data, preprocess, evaluate baselines.
set -euo pipefail
cd "$(dirname "$0")/.."

PY="${PY:-.venv/bin/python}"

echo "==> Preprocessing real Borg 2011 shards 0-3 (downloads if missing)"
PYTHONPATH=src "$PY" -m cluster_rl.data.preprocess --source google2011 --parts 0-3 --out data/processed/smoke

echo "==> Evaluating baseline placement policies"
PYTHONPATH=src "$PY" -m cluster_rl.evaluate --config configs/smoke.yaml --episodes 5 --seed 0
