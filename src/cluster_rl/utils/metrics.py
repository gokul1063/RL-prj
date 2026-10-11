"""Cluster-level metrics used for evaluation and reporting."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

import numpy as np

EPS = 1e-9


def jains_fairness(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    if x.size == 0 or np.allclose(x, 0):
        return 1.0
    return float((x.sum() ** 2) / (x.size * (x ** 2).sum() + EPS))


def shannon_entropy(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    total = x.sum()
    if total <= 0:
        return 0.0
    p = x[x > 0] / total
    return float(-(p * np.log(p)).sum())


def imbalance_cpu_mem(cluster) -> float:
    """Mean coefficient-of-variation of per-node CPU and memory utilisation."""
    cpu = cluster.cpu_util()
    mem = cluster.mem_util()
    return float(0.5 * (cpu.std() + mem.std()))


def fragmentation(cluster, ref_cpu: float, ref_mem: float) -> float:
    """Fraction of nodes with mismatched leftover capacity (stranded resources).

    A node is 'stranded' if it has spare CPU but not the reference memory, or
    vice versa -- leftover capacity that cannot host even a small container.
    """
    rem_cpu = cluster.cpu_cap * cluster.reserve_ratio - cluster.cpu_used
    rem_mem = cluster.mem_cap * cluster.reserve_ratio - cluster.mem_used
    cpu_free = rem_cpu > ref_cpu
    mem_free = rem_mem > ref_mem
    stranded = (cpu_free & ~mem_free) | (~cpu_free & mem_free)
    return float(stranded.mean())


@dataclass
class EpisodeMetrics:
    scheduled: int = 0
    failed: int = 0
    unscheduled: int = 0
    deferred: int = 0
    dropped_stale: int = 0
    wait_ticks: list[float] = field(default_factory=list)
    imbalance_sum: float = 0.0
    imbalance_n: int = 0
    cpu_util_sum: float = 0.0
    mem_util_sum: float = 0.0
    util_n: int = 0
    colocation: int = 0
    traffic_imbalance: float = float("nan")
    mean_latency: float = float("nan")
    p95_latency: float = float("nan")
    traffic_imbalance_sum: float = 0.0
    latency_sum: float = 0.0
    p95_latency_last: float = float("nan")
    traffic_n: int = 0

    def summary(self) -> dict:
        waits = np.asarray(self.wait_ticks, dtype=float) if self.wait_ticks else np.array([0.0])
        if self.traffic_n > 0:
            self.traffic_imbalance = self.traffic_imbalance_sum / self.traffic_n
            self.mean_latency = self.latency_sum / self.traffic_n
            self.p95_latency = self.p95_latency_last
        return {
            "scheduled": int(self.scheduled),
            "failed": int(self.failed),
            "unscheduled": int(self.unscheduled),
            "deferred": int(self.deferred),
            "dropped_stale": int(self.dropped_stale),
            "failure_rate": float(self.failed / max(self.scheduled, 1)),
            "sla_violation_rate": float(self.unscheduled / max(self.scheduled + self.unscheduled, 1)),
            "wait_mean_ticks": float(waits.mean()),
            "wait_p95_ticks": float(np.percentile(waits, 95)),
            "imbalance_mean": float(self.imbalance_sum / max(self.imbalance_n, 1)),
            "cpu_util_mean": float(self.cpu_util_sum / max(self.util_n, 1)),
            "mem_util_mean": float(self.mem_util_sum / max(self.util_n, 1)),
            "colocation_mean": float(self.colocation / max(self.scheduled, 1)),
            "traffic_imbalance": float(self.traffic_imbalance),
            "mean_latency": float(self.mean_latency),
            "p95_latency": float(self.p95_latency),
        }

    def as_dict(self) -> dict:
        return asdict(self)
