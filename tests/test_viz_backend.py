"""Sanity tests for the real-data backend adapter (rlviz/backend/env.py)."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np

os.environ.setdefault("VIZ_WARMUP", "6")  # small test cluster

BACKEND = Path(__file__).resolve().parents[1] / "rlviz" / "backend"
sys.path.insert(0, str(BACKEND))

from env import ACTIONS, RealClusterEnv  # noqa: E402


def _cfg():
    from cluster_rl.env import EnvConfig

    return EnvConfig.from_dict(
        {
            "processed_dir": "unused",
            "max_nodes": 4,
            "episode_ticks": 12,
            "max_arrivals": 240,
            "max_wait_ticks": 6,
            "hold_max": 3,
            "max_steps_multiplier": 4,
        }
    )


def _data():
    """Lively synthetic workload so several containers stay running."""
    import pandas as pd

    rng = np.random.default_rng(0)
    n = 240
    services = np.array(["frontend", "auth", "cart", "payments"])
    submit = np.sort(rng.uniform(0, 5 * 300e6, size=n))          # first 5 ticks
    dur = rng.uniform(2.0, 6.0, size=n) * 300e6                  # 2-6 ticks
    priority = rng.choice([0, 1, 5, 9, 10], size=n)
    fail_p = np.where(priority <= 1, 0.5, 0.1)
    failed = rng.random(n) < fail_p
    terminal = np.where(failed, "evict", "finish")
    schedule = submit + rng.uniform(0, 30e6, size=n)
    end = schedule + dur
    n_services = len(services)
    idx = rng.integers(0, n_services, size=n)
    images = np.array(["nginx:1.25-alpine", "redis:7.2", "postgres:16", "python:3.12-slim"])
    return pd.DataFrame(
        {
            "instance_id": [f"{1000 + i}:0" for i in range(n)],
            "collection_id": 1000 + np.arange(n, dtype=np.int64),
            "instance_index": np.zeros(n, dtype=np.int64),
            "machine_id": rng.integers(0, 6, size=n).astype(np.int64),
            "submit_time": submit,
            "schedule_time": schedule,
            "end_time": end,
            "terminal_event": terminal,
            "failed": failed,
            "killed": np.zeros(n, dtype=bool),
            "cpu_request": rng.uniform(0.02, 0.08, size=n),
            "memory_request": rng.uniform(0.02, 0.08, size=n),
            "priority": priority,
            "scheduling_class": rng.integers(0, 4, size=n),
            "duration": dur,
            "docker_image": images[idx],
            "service": services[idx],
        }
    )


def _machines():
    import pandas as pd

    return pd.DataFrame(
        {
            "machine_id": np.arange(6, dtype=np.int64),
            "cpu_capacity": np.full(6, 0.5),
            "memory_capacity": np.full(6, 0.5),
            "platform_id": ["test"] * 6,
            "add_time": np.zeros(6),
            "remove_time": np.full(6, np.nan),
        }
    )


def test_actions_are_placement_policies():
    assert len(ACTIONS) == 6
    assert {a["policy"] for a in ACTIONS} == {
        "round_robin",
        "first_fit",
        "best_fit",
        "worst_fit",
        "least_requested",
        "risk_aware",
    }


def test_frame_schema():
    rce = RealClusterEnv(config=_cfg(), seed=1, instances=_data(), machines=_machines())
    f = rce.frame()
    assert {
        "t", "traffic", "failure_rate", "failed_count", "method", "k",
        "nodes", "clusters", "links", "imbalance",
    } <= set(f)
    assert len(f["clusters"]) == rce.ev.n_nodes
    assert f["nodes"], "warm-up should leave running containers"
    assert {"id", "name", "image", "cluster", "failed", "cpu", "mem", "net", "demand", "fails"} == set(f["nodes"][0])
    assert {"id", "label", "size", "failed", "load", "share"} == set(f["clusters"][0])
    assert f["links"][0]["source"] == "ingress"


def test_observe_apply_and_tick():
    rce = RealClusterEnv(config=_cfg(), seed=2, instances=_data(), machines=_machines())
    assert 0 <= rce.observe() < rce.n_states
    rce.apply_action(0)
    assert rce.action == 0
    for _ in range(15):
        f = rce.tick()
        assert f["t"] == rce.t
        assert f["method"] == ACTIONS[0]["name"]
    assert rce.moved_frac == 0.0


def test_inject_failure():
    rce = RealClusterEnv(config=_cfg(), seed=3, instances=_data(), machines=_machines())
    ids = rce.inject_failure()
    assert isinstance(ids, list) and all(isinstance(i, int) for i in ids)
    if ids:
        rce.fail_containers(ids)
        assert rce.frame()["failed_count"] >= 0
    assert np.isfinite(rce.frame()["failure_rate"])