#!/usr/bin/env bash
# One-command runner: environment, data, tests, baselines, RL training,
# evaluation, dashboard and tensorboard. Safe to re-run (idempotent).
#
#   ./run.sh                # full pipeline (setup + test + baseline + quick train + eval)
#   ./run.sh setup          # venv + deps + data only
#   ./run.sh test           # pytest
#   ./run.sh baseline       # heuristic baselines
#   ./run.sh train          # quick PPO (TRAIN_STEPS, default 20000)
#   ./run.sh train-full     # 300k-step PPO using configs/rl_big.yaml
#   ./run.sh eval           # compare latest model vs baselines
#   ./run.sh ui             # Streamlit dashboard (http://localhost:PORT)
#   ./run.sh tb             # TensorBoard (http://localhost:6006)
#
# Environment overrides: PARTS=0-3  CONFIG=configs/smoke.yaml  RL_CONFIG=configs/rl.yaml
#   TRAIN_STEPS=20000  N_ENVS=4  EVAL_EPISODES=5  RUN_NAME=ppo_tmp  RL_MODEL=runs/...zip  PORT=8501
set -euo pipefail
cd "$(dirname "$0")"

PY=".venv/bin/python"
PYTHON="python3"
SRC="src"

CONFIG="${CONFIG:-configs/smoke.yaml}"
RL_CONFIG="${RL_CONFIG:-configs/rl.yaml}"
PARTS="${PARTS:-0-3}"
TRAIN_STEPS="${TRAIN_STEPS:-20000}"
N_ENVS="${N_ENVS:-4}"
EVAL_EPISODES="${EVAL_EPISODES:-5}"
PORT="${PORT:-8501}"

say() { printf '\n\033[1;34m==> %s\033[0m\n' "$*"; }

ensure_venv() {
  say "environment (.venv)"
  if [[ ! -x "$PY" ]]; then
    echo "  creating virtualenv and installing dependencies ..."
    "$PYTHON" -m venv .venv
    "$PY" -m pip install -r requirements.txt
    touch .venv/.req_stamp
  elif [[ requirements.txt -nt .venv/.req_stamp ]]; then
    echo "  requirements changed -> reinstalling ..."
    "$PY" -m pip install -r requirements.txt --quiet
    touch .venv/.req_stamp
  else
    echo "  already set up."
  fi
}

ensure_data() {
  local need=0 p f
  say "data (real Borg 2011 trace)"
  [[ -s data/raw/google2011/machine_events/part-00000-of-00001.csv.gz ]] || need=1
  for p in 0 1 2 3; do
    f="data/raw/google2011/task_events/part-$(printf '%05d' "$p")-of-00500.csv.gz"
    [[ -s "$f" ]] || need=1
  done
  if [[ "$need" == 1 ]]; then
    echo "  raw shards missing -> downloading parts ${PARTS} ..."
    bash scripts/download_data.sh "$PARTS"
  else
    echo "  raw shards present."
  fi
  if [[ ! -s data/processed/smoke/instances.parquet ]]; then
    echo "  preprocessing ..."
    PYTHONPATH="$SRC" "$PY" -m cluster_rl.data.preprocess \
      --source google2011 --parts "$PARTS" --out data/processed/smoke
  else
    echo "  preprocessed data present."
  fi
}

cmd_setup() { ensure_venv; ensure_data; }
cmd_test() { say "test suite"; "$PY" -m pytest; }
cmd_baseline() {
  cmd_setup
  say "baseline evaluation"
  PYTHONPATH="$SRC" "$PY" -m cluster_rl.evaluate --config "$CONFIG" --episodes "$EVAL_EPISODES" --seed 0
}
cmd_train() {
  cmd_setup
  say "train MaskablePPO (${TRAIN_STEPS} steps, ${N_ENVS} envs)"
  PYTHONPATH="$SRC" "$PY" -m cluster_rl.train \
    --config "$CONFIG" --rl-config "$RL_CONFIG" \
    --total-timesteps "$TRAIN_STEPS" --n-envs "$N_ENVS" \
    --run-name "${RUN_NAME:-ppo_tmp}" --seed 0
}
cmd_train_full() {
  cmd_setup
  say "train MaskablePPO (full 300k using configs/rl_big.yaml)"
  PYTHONPATH="$SRC" "$PY" -m cluster_rl.train \
    --config "$CONFIG" --rl-config configs/rl_big.yaml \
    --run-name "${RUN_NAME:-ppo_full}" --seed 0
}
cmd_eval() {
  cmd_setup
  local m="${RL_MODEL:-}"
  if [[ -z "$m" ]]; then
    m="$(ls -t runs/*/ppo_final.zip 2>/dev/null | head -1 || true)"
  fi
  if [[ -z "$m" || ! -s "$m" ]]; then
    echo "  no trained model found. Run './run.sh train' first." >&2
    return 1
  fi
  say "compare RL policy vs baselines: $m"
  PYTHONPATH="$SRC" "$PY" -m cluster_rl.evaluate \
    --config "$CONFIG" --episodes "$EVAL_EPISODES" --rl-model "$m"
}
cmd_ui() {
  cmd_setup
  say "launching dashboard -> http://localhost:${PORT}"
  RL_MODEL="${RL_MODEL:-}" bash scripts/run_ui.sh
}
cmd_tb() {
  say "launching TensorBoard -> http://localhost:6006"
  "$PY" -m tensorboard.main --logdir runs --port 6006
}
cmd_all() {
  cmd_setup
  cmd_test
  cmd_baseline
  cmd_train
  cmd_eval
  say "pipeline complete. For the interactive dashboard run:  ./run.sh ui"
}

usage() {
  grep '^#' "$0" | sed '1d' | sed 's/^# \{0,1\}//'
}

case "${1:-all}" in
  all|pipeline) cmd_all ;;
  setup) cmd_setup ;;
  test) cmd_test ;;
  baseline) cmd_baseline ;;
  train) cmd_train ;;
  train-full) cmd_train_full ;;
  eval) cmd_eval ;;
  ui) cmd_ui ;;
  tb) cmd_tb ;;
  help|-h|--help) usage ;;
  *) echo "unknown command: $1" >&2; echo; usage >&2; exit 1 ;;
esac