# ClusterRL — real-data cluster placement, live visualised

A Python backend replays the **real Google Borg 2011 cluster trace** through the
`cluster_rl` placement environment and streams state over a WebSocket; a React
canvas UI draws it.

```
rlviz/
├── backend/
│   ├── env.py        BaseClusterEnv contract + RealClusterEnv (real-data bridge)
│   ├── agent.py      tabular Q-learning agent
│   ├── session.py    RL env + static-baseline env on identical workload
│   ├── main.py       FastAPI + WebSocket (/ws), serves frontend/dist
│   └── requirements.txt
└── frontend/         Vite + React + d3-force (canvas)
```

## Run it

From the repo root:

```bash
bash scripts/run_viz.sh          # builds the UI, serves it -> http://localhost:8000
```

or `./run.sh viz`. The script creates the venv, installs deps, ensures the real
trace data is downloaded/preprocessed, builds the frontend, and starts the
backend on one port.

Dev mode (hot reload):

```bash
# terminal 1
cd rlviz/backend && PYTHONPATH=../../src uvicorn main:app --port 8000
# terminal 2
cd rlviz/frontend && npm install && npm run dev     # http://localhost:5173
```

## How the real environment maps onto the UI

`RealClusterEnv` (backend/env.py) adapts `cluster_rl.env.PlacementEnv`:

| UI element | Real-data meaning |
|---|---|
| Cluster halo (A, B, C…) | a physical cluster node (`machine_events` capacity) |
| Container dot | a running container from the Borg trace (`instance_events`) |
| Dot size (`demand`) | the container's CPU + memory request |
| Red dot | the container failed (sampled from the trace-calibrated risk model) |
| Link ingress → cluster | per-node traffic from the traffic proxy (request routing) |
| **Action** | which **placement policy** drives the next window (RR/FF/BF/WF/LR/RA) |
| Failure-rate KPI | RL policy vs a static `Least-requested` baseline on the same workload |
| ⚡ Inject failure | force-fails containers on the selected node in both envs |

Each UI tick places one incoming container with the active policy; every 10 ticks
the Q-learning agent re-chooses a policy based on the current load band
(Low/Moderate/High/Surge). Reward = `-8·failure_rate − 2.5·imbalance − 0.5·migration`.

## Configuration

* `CLUSTER_RL_CONFIG` — env YAML (default `configs/viz.yaml`: 14 nodes, 2500 containers).
* `VIZ_WARMUP` — containers to pre-place before the first frame (default 48).
* `VIZ_CONTAINERS` — max container dots drawn (default 140).
* `N_CONTAINERS` / `PORT` handled by `scripts/run_viz.sh`.

The UI contract (`BaseClusterEnv` + `frame()` schema) is unchanged from the
original scaffold — only `RealClusterEnv` and `session.make_env()` were swapped.
