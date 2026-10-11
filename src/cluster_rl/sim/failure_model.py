"""Failure-risk model.

Real signal
-----------
Trained on the Borg trace's terminal events: ``FAIL``/``EVICT``/``LOST`` are
labelled unsuccessful, ``FINISH`` successful. ``KILL`` (user/scheduler initiated)
and never-terminated (``running``) instances are excluded from training.

The learned base probability is then modulated by the *placement decision* via
node utilisation and recent-failure history, so the agent is rewarded for making
choices that reduce failure likelihood. The modulation is a deliberate, clearly
documented proxy: the trace does not record per-placement counterfactuals.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from cluster_rl.sim.events import Instance

FEATURES = ["cpu_request", "memory_request", "priority", "scheduling_class"]
TRAIN_TERMINALS = {"finish", "evict", "fail", "lost"}


class FailureModel:
    def __init__(
        self,
        beta: float = 2.0,
        base_scale: float = 1.0,
        recent_fail_weight: float = 0.5,
        min_p: float = 1e-4,
        max_p: float = 0.6,
    ) -> None:
        self.beta = beta
        self.base_scale = base_scale
        self.recent_fail_weight = recent_fail_weight
        self.min_p = min_p
        self.max_p = max_p
        self.prior_ = 0.1
        self.clf = None
        self.machine_rate_: dict[int, float] = {}

    # -- training ------------------------------------------------------------
    def fit(self, instances, smoothing: float = 20.0) -> "FailureModel":
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler

        df = instances[instances["terminal_event"].isin(TRAIN_TERMINALS)].copy()
        if df.empty or df["failed"].nunique() < 2:
            self.prior_ = float(df["failed"].mean()) if len(df) else 0.1
            self.clf = None
            return self

        self.prior_ = float(df["failed"].mean())
        X = df[FEATURES].to_numpy(dtype=float)
        y = df["failed"].to_numpy(dtype=int)
        # A smooth linear model keeps probabilities calibrated and avoids the
        # extreme extrapolation a tree ensemble showed on held-out windows.
        self.clf = make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=1000, C=1.0, class_weight=None),
        )
        self.clf.fit(X, y)
        return self

    # -- inference -----------------------------------------------------------
    def base_probability(
        self,
        cpu: float,
        mem: float,
        priority: int,
        sched_class: int,
        machine_id: int = -1,
    ) -> float:
        if self.clf is None:
            return float(self.prior_)
        x = np.array([[cpu, mem, priority, sched_class]], dtype=float)
        return float(self.clf.predict_proba(x)[0, 1])

    def placement_probability(
        self,
        instance: Instance,
        cluster,
        node: int,
    ) -> float:
        """P(fail) after placing ``instance`` on ``node`` (load-aware)."""
        return self.risk_components(instance, cluster, node)[1]

    def risk_components(self, instance: Instance, cluster, node: int):
        """Return ``(base_p, p_fail_on_node)``.

        ``base_p`` is the trace-learned, placement-independent risk. The extra
        risk on a node comes from its post-placement utilisation and recent
        failure history -- the part the agent can actually influence.
        """
        base = self.base_probability(
            instance.cpu,
            instance.mem,
            instance.priority,
            instance.sched_class,
            cluster.nodes[node].machine_id,
        )
        cpu_after = (cluster.cpu_used[node] + instance.cpu) / max(cluster.cpu_cap[node], 1e-9)
        mem_after = (cluster.mem_used[node] + instance.mem) / max(cluster.mem_cap[node], 1e-9)
        load = max(cpu_after, mem_after)
        recent = cluster.recent_fail[node]
        p = self.base_scale * base * (1.0 + self.beta * load) + self.recent_fail_weight * recent * base
        return base, float(np.clip(p, self.min_p, self.max_p))

    # -- persistence ---------------------------------------------------------
    def save(self, path: str | Path) -> None:
        import joblib

        joblib.dump(self, path)

    @staticmethod
    def load(path: str | Path) -> "FailureModel":
        import joblib

        return joblib.load(path)
