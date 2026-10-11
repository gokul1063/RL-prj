"""Environment configuration loaded from YAML."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml


@dataclass
class EnvConfig:
    seed: int = 0
    processed_dir: str = "data/processed/smoke"

    # cluster / episode
    max_nodes: int = 8
    tick_seconds: float = 300.0
    episode_ticks: int = 48
    max_arrivals: int = 1500
    reserve_ratio: float = 1.0
    max_wait_ticks: int = 12
    hold_max: int = 5
    max_steps_multiplier: int = 4

    # failure model
    failure: dict = field(
        default_factory=lambda: {
            "beta": 0.5,
            "base_scale": 1.0,
            "recent_fail_weight": 0.3,
        }
    )

    # reward
    reward: dict = field(default_factory=dict)

    # traffic proxy
    traffic: dict = field(
        default_factory=lambda: {
            "enabled": True,
            "mu_per_cpu": 20.0,
            "mu_scale": 1.0,
            "request_rate_scale": 4.0,
        }
    )

    @staticmethod
    def from_dict(d: dict | None) -> "EnvConfig":
        d = dict(d or {})
        known = {k: d[k] for k in d if k in EnvConfig.__dataclass_fields__}
        return EnvConfig(**known)

    @staticmethod
    def from_yaml(path: str | Path) -> "EnvConfig":
        data = yaml.safe_load(Path(path).read_text()) or {}
        return EnvConfig.from_dict(data)

    def to_dict(self) -> dict:
        return {
            k: getattr(self, k) for k in EnvConfig.__dataclass_fields__
        }
