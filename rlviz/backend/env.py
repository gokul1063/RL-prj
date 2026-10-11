"""Cluster environments for the ClusterRL UI.

The UI and `Session` depend ONLY on `BaseClusterEnv`. `RealClusterEnv` adapts the
real-data `cluster_rl.env.PlacementEnv` (Google Borg 2011 trace) to the UI's
frame schema:

* a physical cluster node  -> one UI "cluster"   (halo, label A/B/C...)
* a running container      -> one UI "container" (dot; red when failed)
* ingress traffic          -> UI "links" whose volume follows the traffic proxy
* an action                -> which placement policy drives the next window

Action = placement policy chosen by the agent per load band.
"""

from __future__ import annotations

import os
from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np

from cluster_rl.baselines import (
    BestFitPolicy,
    FirstFitPolicy,
    LeastRequestedPolicy,
    Policy,
    RiskAwarePolicy,
    RoundRobinPolicy,
    WorstFitPolicy,
)
from cluster_rl.env import EnvConfig, PlacementEnv
from cluster_rl.utils.metrics import imbalance_cpu_mem

REPO_ROOT = Path(__file__).resolve().parents[2]

LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"

# One action = one placement policy the agent may pick at a decision point.
ACTIONS = [
    {"name": "Round-robin", "short": "RR", "policy": "round_robin"},
    {"name": "First-fit", "short": "FF", "policy": "first_fit"},
    {"name": "Best-fit", "short": "BF", "policy": "best_fit"},
    {"name": "Worst-fit", "short": "WF", "policy": "worst_fit"},
    {"name": "Least-requested", "short": "LR", "policy": "least_requested"},
    {"name": "Risk-aware", "short": "RA", "policy": "risk_aware"},
]

POLICY_FACTORIES = {
    "round_robin": lambda: RoundRobinPolicy(),
    "first_fit": lambda: FirstFitPolicy(),
    "best_fit": lambda: BestFitPolicy(),
    "worst_fit": lambda: WorstFitPolicy(),
    "least_requested": lambda: LeastRequestedPolicy(),
    "risk_aware": lambda: RiskAwarePolicy(),
}


class BaseClusterEnv(ABC):
    """Contract between the environment, the agent and the UI."""

    actions: list[dict]
    n_states: int
    state_labels: list[str]
    t: int
    moved_frac: float

    @abstractmethod
    def reset(self, seed: int | None = None) -> None: ...

    @abstractmethod
    def observe(self) -> int: ...

    @abstractmethod
    def apply_action(self, action: int) -> None: ...

    @abstractmethod
    def tick(self) -> dict: ...

    @abstractmethod
    def frame(self) -> dict: ...

    @abstractmethod
    def inject_failure(self, cluster: int | None = None) -> list[int]: ...

    def fail_containers(self, ids: list[int]) -> None:
        pass


class RealClusterEnv(BaseClusterEnv):
    """Replays the real Google Borg 2011 workload through the UI.

    Each UI tick places exactly one incoming container using the active placement
    policy, so the canvas shows the real cluster evolving under live traffic.
    """

    actions = ACTIONS
    n_states = 4
    state_labels = ["Low traffic", "Moderate", "High", "Surge"]
    EDGES = (0.30, 0.55, 0.80)
    BASELINE_POLICY = "least_requested"

    def __init__(
        self,
        config_path: str | os.PathLike | None = None,
        config: EnvConfig | dict | None = None,
        seed: int = 7,
        instances=None,
        machines=None,
    ) -> None:
        if config is None:
            if config_path is None:
                config_path = os.getenv("CLUSTER_RL_CONFIG") or (
                    REPO_ROOT / "configs" / "viz.yaml"
                )
            config = EnvConfig.from_yaml(config_path)
        elif isinstance(config, dict):
            config = EnvConfig.from_dict(config)
        assert isinstance(config, EnvConfig)
        # The backend runs from rlviz/backend; resolve data paths against repo root.
        if not Path(config.processed_dir).is_absolute():
            config.processed_dir = str(REPO_ROOT / config.processed_dir)
        self.cfg = config
        self.ev = PlacementEnv(config, instances=instances, machines=machines)
        self._epoch = 0
        self._fails: dict[int, int] = {}
        self._failing: set[int] = set()
        self.seed = seed
        self.reset(seed)

    # --------------------------------------------------------------- life
    def reset(self, seed: int | None = None) -> None:
        if seed is not None:
            self.seed = seed
        self.ev.reset(seed=self.seed)
        self.policy: Policy = POLICY_FACTORIES[self.BASELINE_POLICY]()
        self.action = self.actions_index(self.BASELINE_POLICY)
        self.t = 0
        self.moved_frac = 0.0
        self._fails, self._failing = {}, set()
        self._warmup()

    def _roll(self) -> None:
        """Auto-roll to a fresh workload when an episode ends (timeline continues)."""
        self._epoch += 1
        self.ev.reset(seed=self.seed + 1000 * self._epoch)
        self._fails, self._failing = {}, set()
        self._warmup()

    def actions_index(self, policy: str) -> int:
        for i, a in enumerate(ACTIONS):
            if a["policy"] == policy:
                return i
        return 0

    def _warmup(self) -> None:
        ev = self.ev
        target = int(os.getenv("VIZ_WARMUP", "48"))
        steps = 0
        while not ev._done and steps < 300 and ev.cluster.running_count.sum() < target:
            if not ev.pending:
                ev.step(ev.HOLD)
            else:
                mask = ev.action_masks()
                act = (
                    self.policy.act(ev, ev._obs(), mask)
                    if mask[: ev.n_nodes].any()
                    else ev.HOLD
                )
                ev.step(act)
            steps += 1

    # --------------------------------------------------------------- agent
    def observe(self) -> int:
        u = float(self.ev.cluster.cpu_util().mean())
        return int(sum(u > e for e in self.EDGES))

    def apply_action(self, action: int) -> None:
        self.policy = POLICY_FACTORIES[ACTIONS[int(action)]["policy"]]()
        self.action = int(action)
        self.moved_frac = 0.0

    def tick(self) -> dict:
        if self.ev._done:
            self._roll()
            return self.frame()
        ev = self.ev
        if ev.pending:
            mask = ev.action_masks()
            act = (
                self.policy.act(ev, ev._obs(), mask)
                if mask[: ev.n_nodes].any()
                else ev.HOLD
            )
        else:
            act = ev.HOLD
        ev.step(act)
        self._sync_fail_flags()
        self.t += 1
        return self.frame()

    # --------------------------------------------------------------- failure
    def _sync_fail_flags(self) -> None:
        alive = set(self.ev.running)
        for iid in alive:
            if self.ev.running[iid].failed and iid not in self._failing:
                self._failing.add(iid)
                self._fails[iid] = self._fails.get(iid, 0) + 1
        for iid in list(self._failing):
            if iid not in alive:
                self._failing.discard(iid)

    def inject_failure(self, cluster: int | None = None) -> list[int]:
        ev = self.ev
        possible = [n for n in range(ev.n_nodes) if ev.cluster.running_count[n] > 0]
        if not possible:
            return []
        if cluster not in possible:
            cluster = int(np.random.choice(possible))
        hosted = [i for i in ev.cluster.hosted[cluster] if not ev.running[i].failed]
        if not hosted:
            return []
        count = min(max(2, len(hosted) // 4), len(hosted))
        picks = [int(p) for p in np.random.choice(hosted, size=count, replace=False)]
        ev.fail_instances(picks)
        self._sync_fail_flags()
        return picks

    def fail_containers(self, ids: list[int]) -> None:
        if ids:
            self.ev.fail_instances(ids)
            self._sync_fail_flags()

    # --------------------------------------------------------------- frame
    def frame(self) -> dict:
        ev = self.ev
        n_nodes = ev.n_nodes
        total_cpu = ev.cluster.total_cpu_capacity or 1e-9
        running = list(ev.running.values())
        running.sort(key=lambda ins: ins.start_tick, reverse=True)
        cap = int(os.getenv("VIZ_CONTAINERS", "140"))
        shown = running[:cap]
        failed_count = int(sum(1 for ins in running if ins.failed))

        clusters = []
        for n in range(n_nodes):
            members = [ins for ins in running if ins.node == n]
            clusters.append(
                {
                    "id": n,
                    "label": LETTERS[n % 26],
                    "size": int(ev.cluster.running_count[n]),
                    "failed": int(sum(1 for ins in members if ins.failed)),
                    "load": round(float(ev.cluster.cpu_util()[n]), 3),
                    "share": round(float(ev.cluster.cpu_used[n]) / total_cpu, 4),
                }
            )

        report = (
            ev.traffic_model.evaluate(ev.cluster, ev.service_of)
            if ev.traffic_cfg.get("enabled", True)
            else None
        )
        lam = report.node_lambda if report is not None else np.zeros(n_nodes)
        mean_lam = float(lam.mean()) or 1e-9

        nodes = []
        for ins in shown:
            iid = ins.idx
            nodes.append(
                {
                    "id": iid,
                    "name": ins.instance_id,
                    "image": ins.docker_image,
                    "cluster": int(ins.node),
                    "failed": bool(ins.failed),
                    "cpu": round(float(ins.cpu * 6.0), 3),
                    "mem": round(float(ins.mem * 6.0), 3),
                    "net": round(0.25 + 0.06 * (iid % 7), 3),
                    "demand": round(
                        float(min(1.6, max(0.25, (ins.cpu + ins.mem) * 9.0))), 3
                    ),
                    "fails": int(self._fails.get(iid, 0)),
                }
            )

        links = [
            {
                "source": "ingress",
                "target": int(c["id"]),
                "volume": round(float(lam[int(c["id"])]) / mean_lam, 4),
                "degraded": c["failed"] > 0,
            }
            for c in clusters
        ]

        util_mean = float(ev.cluster.cpu_util().mean())
        traffic = float(np.clip(0.35 + util_mean * 1.4, 0.2, 1.8))
        return {
            "t": self.t,
            "traffic": round(traffic, 4),
            "failure_rate": round(failed_count / max(len(running), 1), 4),
            "failed_count": failed_count,
            "method": ACTIONS[self.action]["name"],
            "k": n_nodes,
            "imbalance": round(imbalance_cpu_mem(ev.cluster), 4),
            "nodes": nodes,
            "clusters": clusters,
            "links": links,
        }