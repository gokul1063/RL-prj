"""Shared pytest fixtures: small, offline, synthetic-but-schema-valid data."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from cluster_rl.env import EnvConfig, PlacementEnv


def make_machines(n: int = 6, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {
            "machine_id": np.arange(n, dtype=np.int64),
            "cpu_capacity": rng.choice([0.25, 0.5, 1.0], size=n),
            "memory_capacity": rng.choice([0.25, 0.5, 0.75], size=n),
            "platform_id": ["test"] * n,
            "add_time": np.zeros(n),
            "remove_time": np.full(n, np.nan),
        }
    )


def make_instances(n: int = 300, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    services = np.array(["frontend", "auth", "cart", "payments"])
    images = np.array(["nginx:1.25-alpine", "redis:7.2", "postgres:16", "python:3.12-slim"])
    coyote = rng.integers(0, len(services), size=n)
    priority = rng.choice([0, 1, 5, 9, 10], size=n)
    submit = np.sort(rng.uniform(0, 48 * 300e6, size=n))
    duration_s = rng.uniform(60, 900, size=n)
    # free/low priorities fail more often (mirrors real Borg evictions)
    fail_p = np.where(priority <= 1, 0.5, np.where(priority <= 8, 0.05, 0.1))
    failed = rng.random(n) < fail_p
    killed = (~failed) & (rng.random(n) < 0.08)
    finish = ~(failed | killed)

    terminal = np.where(failed, "evict", np.where(killed, "kill", "finish"))
    schedule = submit + rng.uniform(0, 60e6, size=n)
    end = schedule + duration_s * 1e6
    return pd.DataFrame(
        {
            "instance_id": [f"{1000+i}:0" for i in range(n)],
            "collection_id": 1000 + np.arange(n, dtype=np.int64),
            "instance_index": np.zeros(n, dtype=np.int64),
            "machine_id": rng.integers(0, 6, size=n).astype(np.int64),
            "submit_time": submit,
            "schedule_time": schedule,
            "end_time": end,
            "terminal_event": terminal,
            "failed": failed,
            "killed": killed,
            "cpu_request": rng.uniform(0.01, 0.1, size=n),
            "memory_request": rng.uniform(0.01, 0.1, size=n),
            "priority": priority,
            "scheduling_class": rng.integers(0, 4, size=n),
            "duration": duration_s * 1e6,
            "docker_image": images[coyote],
            "service": services[coyote],
        }
    )


@pytest.fixture
def machines() -> pd.DataFrame:
    return make_machines()


@pytest.fixture
def instances() -> pd.DataFrame:
    return make_instances()


@pytest.fixture
def env(instances, machines) -> PlacementEnv:
    cfg = EnvConfig.from_dict(
        {
            "max_nodes": 4,
            "episode_ticks": 24,
            "max_arrivals": 150,
            "max_wait_ticks": 6,
            "hold_max": 3,
            "processed_dir": "unused",
            "failure": {"beta": 0.5, "base_scale": 1.0, "recent_fail_weight": 0.3},
        }
    )
    return PlacementEnv(cfg, instances=instances, machines=machines)
