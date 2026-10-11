"""Traffic proxy.

The Borg 2011 trace has no client request logs, so traffic is a calibrated
overlay: each synthetic *service* has a heavy-tailed request rate, routed across
the nodes currently hosting its replicas in proportion to spare CPU capacity.
Node latency is an M/M/1-style proxy. The resulting per-node arrival imbalance
is the "traffic balancing" signal.

This module contributes episode metrics; the per-step reward uses the cheaper
``service_colocation`` signal to avoid recomputing routing at every step.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

EPS = 1e-9


@dataclass
class TrafficReport:
    node_lambda: np.ndarray
    node_latency: np.ndarray
    imbalance: float
    mean_latency: float
    p95_latency: float


class TrafficModel:
    def __init__(
        self,
        service_rates: dict[str, float],
        mu_scale: float = 1.0,
        mu_per_cpu: float = 20.0,
        max_latency: float = 50.0,
    ) -> None:
        self.service_rates = dict(service_rates)
        self.mu_scale = mu_scale
        self.mu_per_cpu = mu_per_cpu
        self.max_latency = max_latency

    @staticmethod
    def service_colocation(cluster, node: int, service: str, service_of: Sequence[str]) -> int:
        """How many other replicas of ``service`` already run on ``node``."""
        return sum(
            1
            for iid in cluster.hosted[node]
            if 0 <= iid < len(service_of) and service_of[iid] == service
        )

    def evaluate(self, cluster, service_of: Sequence[str]) -> TrafficReport:
        n = cluster.n
        lam = np.zeros(n, dtype=float)
        spare = cluster.cpu_cap * (1.0 - cluster.cpu_util()) + EPS
        # group replicas by (service, node)
        by_service: dict[str, np.ndarray] = {}
        for node in range(n):
            for iid in cluster.hosted[node]:
                if iid < 0 or iid >= len(service_of):
                    continue
                svc = service_of[iid]
                by_service.setdefault(svc, np.zeros(n))[node] += 1.0

        for svc, counts in by_service.items():
            rate = self.service_rates.get(svc, 0.0)
            weights = np.where(counts > 0, spare, 0.0)
            total = weights.sum()
            if total <= 0:
                continue
            lam += rate * weights / total

        mu = np.maximum(cluster.cpu_cap * self.mu_per_cpu * self.mu_scale, EPS)
        headroom = mu - lam
        with np.errstate(divide="ignore", invalid="ignore"):
            latency = np.where(headroom > 0, 1.0 / headroom, self.max_latency)
        latency = np.clip(latency, 0.0, self.max_latency)

        mean = float(lam.mean())
        imbalance = float(lam.std() / (mean + EPS))
        return TrafficReport(
            node_lambda=lam,
            node_latency=latency,
            imbalance=imbalance,
            mean_latency=float(latency.mean()),
            p95_latency=float(np.percentile(latency, 95)),
        )
