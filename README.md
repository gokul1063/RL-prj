# cluster-rl — Real-Data RL Environment for Container Placement

Phase 1 is complete: a Gymnasium environment that replays the **real Google Borg
2011 cluster trace** and lets an agent place dockerized containers onto cluster
nodes to balance load, reduce failure risk, and spread service replicas.

Phase 2 (RL training with Stable-Baselines3) builds on the frozen environment
contract in [`docs/ENV_SPEC.md`](docs/ENV_SPEC.md).

## Quick start

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt

# Download real data, preprocess, and evaluate heuristic baselines
bash scripts/run_smoke.sh
```

Or step by step:

```bash
bash scripts/download_data.sh 0-3
PYTHONPATH=src .venv/bin/python -m cluster_rl.data.preprocess --parts 0-3 --out data/processed/smoke
PYTHONPATH=src .venv/bin/python -m cluster_rl.evaluate --config configs/smoke.yaml --episodes 5
PYTHONPATH=src .venv/bin/python -m pytest
```

## Phase 2 — RL training

`MaskablePPO` (SB3) trained against the Phase-1 env using `env.action_masks()`.
The greedy baselines know the failure model directly; PPO only sees observations.

```bash
pip install -r requirements.txt
# 20k-step smoke run (~30s on CPU)
PYTHONPATH=src .venv/bin/python -m cluster_rl.train \
    --config configs/smoke.yaml --rl-config configs/rl.yaml --total-timesteps 20000 --run-name ppo_20k

# bigger run (300k steps, 8 envs, ~7 min CPU)
PYTHONPATH=src .venv/bin/python -m cluster_rl.train --config configs/smoke.yaml --rl-config configs/rl_big.yaml

# compare a trained model against baselines
PYTHONPATH=src .venv/bin/python -m cluster_rl.evaluate \
    --config configs/smoke.yaml --episodes 20 --rl-model runs/ppo_big/ppo_final.zip
```

Typical results (20 episodes, smoke config): `rl:ppo_final` matches the
hand-tuned `risk_aware` greedy (reward ~ +46 vs +58) and clearly beats
first-fit/best-fit (−270); it achieves the lowest CPU utilisation with the
traffic profile of the best heuristics.

The dashboard can also run the trained policy live: point `RL_MODEL` at
`runs/<run>/ppo_final.zip` (default `runs/ppo_big/ppo_final.zip`) and pick
**RL (PPO)** / *Scenario: RL (PPO)*.

## Full-stack visualization (`rlviz/`)

A React + FastAPI app that live-visualizes the **real-data** environment:
physical cluster nodes as halos, running containers as dots (red when failed),
ingress traffic per node, the RL policy vs a static baseline, and a ⚡ failure
injector. The backend streams `RealClusterEnv` frames (adapting `PlacementEnv`)
over a WebSocket.

```bash
bash scripts/run_viz.sh        # or: ./run.sh viz   -> http://localhost:8000
```

See [`rlviz/README.md`](rlviz/README.md) for the UI↔real-data mapping and config.

## Interactive demo (professor demo)

A Streamlit dashboard to visualize and drive the environment live: a queue of
docker containers, a grid of cluster nodes with CPU/memory bars and container
chips, live KPIs (`imbalance`, `failure rate`, `SLA violations`, `traffic
imbalance`), metric charts, a per-step reward breakdown, an event log, a
**Download metrics CSV** button, and one-click presenter scenarios.

```bash
.venv/bin/pip install -r requirements.txt
bash scripts/run_ui.sh                      # -> http://localhost:8501
```

Controls: pick a policy (Manual / Random / First-fit / Best-fit / Worst-fit /
Least-requested / Risk-aware), **Step** or **Auto-play**, or click a node button
in **Manual** mode to place the incoming container yourself. Presenter buttons
run a *bad* (Random) and *good* (Risk-aware) scenario on the same seed to show
the contrast. Over SSH: `ssh -L 8501:localhost:8501 <host>`.

## What is real vs synthetic

| Real (Borg 2011) | Synthetic overlay (documented) |
|---|---|
| container resource requests (cpu/mem), priorities, scheduling classes | docker image / service names |
| lifecycles, terminal events (finish/fail/evict/kill/lost) | per-service request rate for the traffic proxy |
| 12.5k machine capacities from `machine_events` | episode arrival subsampling (density downscale) |

Google 2019 (BigQuery-only, needs GCP billing) is available behind the same
adapter interface (`cluster_rl.data.google2019_bq`).

## Layout

```
src/cluster_rl/
  data/       trace adapters (google2011, google2019_bq), preprocess CLI, synthetic overlay
  sim/        cluster state, workload replay, failure model, traffic proxy
  env/        Gymnasium env, observation, reward, config
  baselines/  heuristic placement policies
  utils/      metrics
  viz/        read-only UI state adapter (snapshot)
  evaluate.py baseline evaluation harness
ui/app.py          Streamlit demo dashboard
docs/ENV_SPEC.md   full rules / reward / policy contract
configs/           smoke + default configs
tests/             offline unit + API tests
```

## Policy interface (Phase 2 hook)

```python
env = PlacementEnv(EnvConfig.from_yaml("configs/smoke.yaml"))
obs, info = env.reset(seed=0)
mask = env.action_masks()                 # mask for MaskablePPO
obs, reward, terminated, truncated, info = env.step(action)  # action in 0..n_nodes (last = HOLD)
```
