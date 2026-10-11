"""Cluster runtime state: nodes, capacity reservation and release."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

EPS = 1e-9


@dataclass
class Node:
    machine_id: int
    cpu_capacity: float
    memory_capacity: float
    platform_id: str = ""


class Cluster:
    """Vectorised node state with O(1) reserve/release.

    Resource accounting is a simple 1-D bin-packing: an instance occupies its
    full ``cpu_request``/``memory_request`` for its whole lifetime and is freed
    when it finishes or fails. No overcommit in Phase 1.
    """

    def __init__(
        self,
        nodes: list[Node],
        reserve_ratio: float = 1.0,
        failure_decay: float = 1.0,
        failure_alpha: float = 0.3,
    ) -> None:
        self.nodes = list(nodes)
        self.n = len(nodes)
        self.reserve_ratio = float(reserve_ratio)
        self.failure_decay = float(failure_decay)
        self.failure_alpha = float(failure_alpha)
        self.cpu_cap = np.array([nd.cpu_capacity for nd in nodes], dtype=np.float64)
        self.mem_cap = np.array([nd.memory_capacity for nd in nodes], dtype=np.float64)
        self.cpu_used = np.zeros(self.n, dtype=np.float64)
        self.mem_used = np.zeros(self.n, dtype=np.float64)
        self.running_count = np.zeros(self.n, dtype=np.int64)
        self.recent_fail = np.zeros(self.n, dtype=np.float64)
        self.hosted: list[list[int]] = [[] for _ in range(self.n)]
        self.reserved_cpu_committed = 0.0

    # -- capacity ------------------------------------------------------------
    def feasible_mask(self, cpu: float, mem: float) -> np.ndarray:
        cpu_ok = self.cpu_used + cpu <= self.cpu_cap * self.reserve_ratio + EPS
        mem_ok = self.mem_used + mem <= self.mem_cap * self.reserve_ratio + EPS
        return cpu_ok & mem_ok

    def can_fit(self, node: int, cpu: float, mem: float) -> bool:
        return bool(self.feasible_mask(cpu, mem)[node])

    def reserve(self, node: int, cpu: float, mem: float, iid: int) -> bool:
        if not self.can_fit(node, cpu, mem):
            return False
        self.cpu_used[node] += cpu
        self.mem_used[node] += mem
        self.running_count[node] += 1
        self.hosted[node].append(iid)
        self.reserved_cpu_committed += cpu
        return True

    def release(self, node: int, cpu: float, mem: float, iid: int, failed: bool = False) -> None:
        self.cpu_used[node] = max(0.0, self.cpu_used[node] - cpu)
        self.mem_used[node] = max(0.0, self.mem_used[node] - mem)
        self.running_count[node] = max(0, self.running_count[node] - 1)
        self.reserved_cpu_committed = max(0.0, self.reserved_cpu_committed - cpu)
        a = self.failure_alpha
        if failed:
            self.recent_fail[node] = self.recent_fail[node] * (1.0 - a) + a
        else:
            self.recent_fail[node] = self.recent_fail[node] * (1.0 - a)
        try:
            self.hosted[node].remove(iid)
        except ValueError:
            pass

    # -- observation helpers -------------------------------------------------
    def cpu_util(self) -> np.ndarray:
        return self.cpu_used / np.maximum(self.cpu_cap, EPS)

    def mem_util(self) -> np.ndarray:
        return self.mem_used / np.maximum(self.mem_cap, EPS)

    def service_counts(self) -> np.ndarray:
        return self.running_count.astype(np.float64)

    def decay_failures(self) -> None:
        self.recent_fail *= self.failure_decay

    @property
    def total_cpu_capacity(self) -> float:
        return float(self.cpu_cap.sum())

    @property
    def total_cpu_used(self) -> float:
        return float(self.cpu_used.sum())

    @property
    def total_mem_used(self) -> float:
        return float(self.mem_used.sum())

    @property
    def total_mem_capacity(self) -> float:
        return float(self.mem_cap.sum())

    def clone(self) -> "Cluster":
        new = Cluster(
            self.nodes,
            reserve_ratio=self.reserve_ratio,
            failure_decay=self.failure_decay,
            failure_alpha=self.failure_alpha,
        )
        new.cpu_used = self.cpu_used.copy()
        new.mem_used = self.mem_used.copy()
        new.running_count = self.running_count.copy()
        new.recent_fail = self.recent_fail.copy()
        new.hosted = [list(h) for h in self.hosted]
        new.reserved_cpu_committed = self.reserved_cpu_committed
        return new
