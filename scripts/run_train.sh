#!/usr/bin/env bash
# Train MaskablePPO on the placement environment.
#   bash scripts/run_train.sh --total-timesteps 100000 --run-name ppo_100k
set -euo pipefail
cd "$(dirname "$0")/.."

PY="${PY:-.venv/bin/python}"
PYTHONPATH=src "$PY" -m cluster_rl.train --config "${CONFIG:-configs/smoke.yaml}" \
  --rl-config "${RL_CONFIG:-configs/rl.yaml}" "$@"