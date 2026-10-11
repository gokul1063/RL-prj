"""Observation vector builder (the frozen state contract).

Layout (length ``5 * n_nodes + 15``)::

    [ node features | instance features | affinity | globals ]
      n*4            9                    n          6

    node features (per node): cpu_util, mem_util, running_norm, recent_fail_norm
    instance:  cpu_norm, mem_norm, priority_norm, sched_class_onehot(4),
               duration_norm, service_norm
    affinity:  same-service replicas already on each node (normalised)
    globals:   queue_norm, cluster_cpu_util, cluster_mem_util,
               time_sin, time_cos, progress
"""

from __future__ import annotations

import numpy as np

EPS = 1e-9
NODE_FEATS = 4
INSTANCE_FEATS = 9
GLOBAL_FEATS = 6


class ObservationBuilder:
    def __init__(
        self,
        n_nodes: int,
        max_cpu_cap: float,
        max_mem_cap: float,
        episode_ticks: int,
        max_arrivals: int,
        n_services: int,
        max_running: float,
        tick_seconds: float,
    ) -> None:
        self.n_nodes = n_nodes
        self.max_cpu_cap = max(max_cpu_cap, EPS)
        self.max_mem_cap = max(max_mem_cap, EPS)
        self.episode_ticks = max(episode_ticks, 1)
        self.max_arrivals = max(max_arrivals, 1)
        self.n_services = max(n_services, 1)
        self.max_running = max(max_running, 1.0)
        self.tick_seconds = tick_seconds

    @property
    def dim(self) -> int:
        return NODE_FEATS * self.n_nodes + INSTANCE_FEATS + self.n_nodes + GLOBAL_FEATS

    def build(
        self,
        cluster,
        instance,
        pending_count: int,
        current_tick: int,
        service_of: list[str],
        service_index: dict[str, int],
    ) -> np.ndarray:
        n = self.n_nodes
        obs = np.zeros(self.dim, dtype=np.float32)

        cpu_util = cluster.cpu_util()
        mem_util = cluster.mem_util()
        running_norm = cluster.running_count / self.max_running
        fail_norm = np.clip(cluster.recent_fail, 0.0, 1.0)

        obs[0:n] = cpu_util
        obs[n:2 * n] = mem_util
        obs[2 * n:3 * n] = running_norm
        obs[3 * n:4 * n] = fail_norm

        base = 4 * n
        if instance is not None:
            obs[base + 0] = instance.cpu / self.max_cpu_cap
            obs[base + 1] = instance.mem / self.max_mem_cap
            obs[base + 2] = instance.priority / 11.0
            obs[base + 3 + min(instance.sched_class, 3)] = 1.0
            obs[base + 7] = min(1.0, instance.duration_ticks / self.episode_ticks)
            obs[base + 8] = service_index.get(instance.service, 0) / max(
                self.n_services - 1, 1
            )

        aff_base = base + INSTANCE_FEATS
        if instance is not None:
            counts = np.zeros(n, dtype=np.float32)
            for node in range(n):
                counts[node] = sum(
                    1 for iid in cluster.hosted[node]
                    if service_of[iid] == instance.service
                )
            obs[aff_base:aff_base + n] = np.clip(counts / 5.0, 0.0, 1.0)

        g = aff_base + n
        obs[g + 0] = min(1.0, pending_count / self.max_arrivals)
        obs[g + 1] = float(cpu_util.mean()) if n else 0.0
        obs[g + 2] = float(mem_util.mean()) if n else 0.0
        frac_day = (current_tick * self.tick_seconds) % 86400.0 / 86400.0
        obs[g + 3] = np.sin(2 * np.pi * frac_day)
        obs[g + 4] = np.cos(2 * np.pi * frac_day)
        obs[g + 5] = min(1.0, current_tick / self.episode_ticks)
        return obs
