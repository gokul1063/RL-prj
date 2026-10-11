"""Reward shaping for the container-placement MDP."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class RewardConfig:
    w_scheduled: float = 1.0
    w_balance: float = 2.0
    w_failure: float = 5.0
    w_affinity: float = 0.5
    w_wait: float = 0.02
    w_frag: float = 0.3
    w_hold: float = 0.1
    w_drop: float = 1.0

    @staticmethod
    def from_dict(d: dict | None) -> "RewardConfig":
        d = d or {}
        return RewardConfig(**{k: float(v) for k, v in d.items() if k in RewardConfig.__dataclass_fields__})


@dataclass
class RewardBreakdown:
    total: float = 0.0
    scheduled: float = 0.0
    balance: float = 0.0
    failure: float = 0.0
    affinity: float = 0.0
    wait: float = 0.0
    frag: float = 0.0
    hold: float = 0.0
    drop: float = 0.0

    def as_dict(self) -> dict:
        return {
            "r_scheduled": self.scheduled,
            "r_balance": self.balance,
            "r_failure": self.failure,
            "r_affinity": self.affinity,
            "r_wait": self.wait,
            "r_frag": self.frag,
            "r_hold": self.hold,
            "r_drop": self.drop,
            "r_total": self.total,
        }


class RewardShaper:
    """Turns one placement (or hold/drop) decision into a scalar reward.

    ``reward =  w_scheduled
              - w_balance  * imbalance_after
              - w_failure  * p_fail
              - w_affinity * same_service_replicas_on_node
              - w_wait     * wait_ticks
              - w_frag     * fragmentation_after
              - w_hold     (when deferring)
              - w_drop     (when a stale instance is dropped)
    """

    def __init__(self, config: RewardConfig | None = None) -> None:
        self.cfg = config or RewardConfig()

    def placement(
        self,
        imbalance_after: float,
        p_fail: float,
        colocation: int,
        wait_ticks: float,
        frag_after: float,
    ) -> RewardBreakdown:
        c = self.cfg
        b = RewardBreakdown()
        b.scheduled = c.w_scheduled
        b.balance = -c.w_balance * float(imbalance_after)
        b.failure = -c.w_failure * float(p_fail)
        b.affinity = -c.w_affinity * float(colocation)
        b.wait = -c.w_wait * float(wait_ticks)
        b.frag = -c.w_frag * float(frag_after)
        b.total = b.scheduled + b.balance + b.failure + b.affinity + b.wait + b.frag
        return b

    def hold(self, wait_ticks: float) -> RewardBreakdown:
        c = self.cfg
        b = RewardBreakdown()
        b.hold = -c.w_hold
        b.wait = -c.w_wait * float(wait_ticks)
        b.total = b.hold + b.wait
        return b

    def drop(self, wait_ticks: float) -> RewardBreakdown:
        c = self.cfg
        b = RewardBreakdown()
        b.drop = -c.w_drop
        b.wait = -c.w_wait * float(wait_ticks)
        b.total = b.drop + b.wait
        return b
