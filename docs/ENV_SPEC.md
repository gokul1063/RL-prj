# Environment Specification — Container Placement RL

This is the frozen contract for Phase 1. Phase 2 trains an RL policy against
this environment without changing the file. Any change here is a new version.

## 1. Problem

A cluster operator receives dockerized workloads ("containers") over time and
must decide **which node hosts each incoming container**. Good placements:

1. keep load balanced across nodes,
2. reduce the probability that a container fails,
3. spread replicas of the same service so traffic is not concentrated.

## 2. Data

| Item | Source | Notes |
|---|---|---|
| Resources, lifecycles, failures | Google Borg 2011 trace (`clusterdata-2011-2`) | Real, zero-auth download |
| Machines / capacity | `machine_events` | 12.5k machines, normalized capacity |
| Optional | Google Borg 2019 via BigQuery (`google2019_bq`) | Same schema, needs GCP billing |
| Service names, docker images, request rates | Small, calibrated synthetic overlay | Trace has neither |

Preprocess: `python -m cluster_rl.data.preprocess --parts 0-3 --out data/processed/smoke`
produces `machines.parquet`, `instances.parquet`, `meta.json`.

An **episode** uses a contiguous tick window (`episode_ticks`) and a random
subset of machines (`max_nodes`) and containers (`max_arrivals`). Uniform
subsampling preserves marginal distributions but downscales absolute arrival
density — a documented limitation.

## 3. Simulation rules

* **Time** advances in ticks of `tick_seconds` (default 300 s = Borg's sample window).
* An instance is **submitted** at `submit_tick`; the agent sees it in the pending
  queue until placed, deferred, or dropped.
* **Resources:** placing an instance on a node reserves its full `cpu_request`
  and `memory_request` for its whole lifetime. No overcommit (`reserve_ratio=1.0`).
* A node can host an instance iff both resource checks pass → encoded in the
  **action mask**.
* **Lifetime:** `finish_tick = start_tick + duration_ticks`.
* **Failure:** on placement the simulator draws `failed ~ Bernoulli(p_fail)`
  (see §5). A failed instance ends at `start + duration/2` and frees resources;
  failures are counted for metrics.
* **Deferral (HOLD):** the agent may defer the current instance; time advances to
  the next event. After `hold_max` deferrals, if any node fits, the instance is
  force-placed on the least-loaded feasible node.
* **Staleness:** a pending instance not placed within `max_wait_ticks` is dropped
  and counted as an SLA violation.
* **Episode end:** when all arrivals are consumed and the cluster drains, or when
  `steps >= max_steps_multiplier * n_arrivals` (truncation).

## 4. Observation (Box, length `5*n_nodes + 15`)

| Block | Size | Contents |
|---|---|---|
| Node | `4*n` | per node: cpu_util, mem_util, running_norm, recent_fail |
| Instance | 9 | cpu_norm, mem_norm, priority_norm, sched_class one-hot (4), duration_norm, service_norm |
| Affinity | `n` | replicas of the current instance's service already on each node |
| Global | 6 | queue_norm, cluster cpu_util, cluster mem_util, time sin, time cos, progress |

All values are normalised into `[0, 2]`.

## 5. Failure model

* Trained on real terminal events: `FAIL`/`EVICT`/`LOST` = unsuccessful,
  `FINISH` = successful; `KILL` and never-terminated excluded.
* Features (scheduling-time): `cpu_request`, `memory_request`, `priority`,
  `scheduling_class`. A standardised **logistic regression** (calibrated; a tree
  ensemble over-extrapolated on held-out windows).
* `p_fail` for a placement:

  ```
  base = P(fail | instance)                       # trace-learned
  load = max(cpu_after, mem_after) / capacity      # after reserving
  p_fail = clip( base * (1 + beta*load)
                 + recent_fail_weight * recent_fail * base, min_p, max_p )
  ```

  Defaults: `beta=0.5`, `recent_fail_weight=0.3`, `max_p=0.6`.
* `recent_fail` is a bounded EMA per node (alpha 0.3) of recent failure events.

## 6. Reward

Per decision, weighted sum (defaults in `configs/*.yaml`):

```
placement: r =  w_scheduled
             - w_balance  * imbalance_after          (mean CV of cpu/mem util)
             - w_failure  * (p_fail - base)           (controllable risk only)
             - w_affinity * same_service_replicas(node)
             - w_wait     * wait_ticks
             - w_frag     * fragmentation_after       (stranded resources)
hold:      r = - w_hold - w_wait * wait_ticks
drop:      r = - w_drop - w_wait * wait_ticks
```

Defaults: `w_scheduled=1`, `w_balance=2`, `w_failure=5`, `w_affinity=0.5`,
`w_wait=0.02`, `w_frag=0.3`, `w_hold=0.1`, `w_drop=1.0`.

Using `p_fail - base` removes the placement-independent part so the risk term
only rewards *actions that actually reduce risk*.

## 7. Policy interface (how Phase 2 plugs in)

```
obs, info = env.reset(seed=s)
while not done:
    mask   = env.action_masks()          # np.ndarray[bool], len n_nodes+1
    action = policy(obs, mask)           # 0..n_nodes-1 place, n_nodes = HOLD
    obs, reward, terminated, truncated, info = env.step(action)
```

* `action_space = Discrete(n_nodes + 1)`.
* `action_masks()` is exposed for `sb3-contrib` `MaskablePPO`; plain SB3 works
  with the masking wrapper / invalid-action penalty.
* Baselines (`cluster_rl.baselines`): random, round-robin, first-fit, best-fit,
  worst-fit, least-requested, risk-aware.

## 8. Metrics (`cluster_rl.utils.metrics`)

Jain's fairness; per-node utilisation mean; imbalance (CV); failure rate;
SLA-violation (dropped) rate; wait mean/p95; fragmentation; traffic imbalance
and M/M/1 mean/p95 latency.

## 9. Known limitations

* Absolute arrival density is downscaled to keep episodes tractable.
* Traffic and docker metadata are synthetic overlays (trace has none), calibrated
  to real resource statistics.
* The failure model is associational; node-load modulation is a documented proxy,
  not a counterfactual from the trace.
* Machine churn (add/remove) is ignored within an episode.
