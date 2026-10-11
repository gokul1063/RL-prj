"""Instance specs and trace replay (arrivals, durations, outcomes)."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd


@dataclass
class Instance:
    idx: int
    instance_id: str
    collection_id: int
    instance_index: int
    service: str
    docker_image: str
    cpu: float
    mem: float
    priority: int
    sched_class: int
    submit_tick: int
    duration_ticks: int
    real_failed: bool
    real_killed: bool

    # runtime state
    node: int = -1
    start_tick: int = -1
    finish_tick: int = -1
    p_fail: float = 0.0
    p_base: float = 0.0
    failed: bool = False
    deferred: int = 0


@dataclass
class EpisodeSpec:
    """Everything the environment needs for one deterministic episode."""

    instances: list[Instance]
    arrivals_by_tick: dict[int, list[int]] = field(default_factory=dict)
    n_ticks: int = 0

    @property
    def n_instances(self) -> int:
        return len(self.instances)


def build_episode(
    instances: pd.DataFrame,
    tick_seconds: float = 300.0,
    episode_ticks: int = 48,
    max_arrivals: int = 1500,
    seed: int = 0,
) -> EpisodeSpec:
    """Subselect a contiguous tick window and subsample arrivals.

    The window is chosen to start at the first submit and span ``episode_ticks``
    ticks. If the window holds more than ``max_arrivals`` instances we draw an
    unweighted random subsample (documented limitation: absolute arrival density
    is downscaled, marginals are preserved).
    """
    df = instances.reset_index(drop=True)
    us_per_tick = tick_seconds * 1e6
    t0 = float(df["submit_time"].min())
    submit_tick_all = ((df["submit_time"] - t0) // us_per_tick).to_numpy(dtype=np.int64)

    dur_us_all = (df["end_time"] - df["schedule_time"]).to_numpy(dtype=float)
    valid = np.isfinite(dur_us_all) & (dur_us_all > 0)
    sample_pool = dur_us_all[valid]
    rng = np.random.default_rng(seed)
    if sample_pool.size == 0:
        sample_pool = np.array([tick_seconds * 1e6])
    fallback = rng.choice(sample_pool, size=len(df))
    dur_us_all = np.where(valid, dur_us_all, fallback)

    in_window = (submit_tick_all >= 0) & (submit_tick_all < episode_ticks)
    df = df[in_window].reset_index(drop=True)
    submit_tick = submit_tick_all[in_window]
    dur_us = dur_us_all[in_window]
    duration_ticks = np.clip(
        np.ceil(dur_us / us_per_tick), 1, episode_ticks
    ).astype(int)

    if len(df) > max_arrivals:
        keep = np.sort(rng.choice(len(df), size=max_arrivals, replace=False))
        df = df.iloc[keep].reset_index(drop=True)
        submit_tick = submit_tick[keep]
        duration_ticks = duration_ticks[keep]

    specs: list[Instance] = []
    arrivals: dict[int, list[int]] = {}
    for i in range(len(df)):
        row = df.iloc[i]
        t = int(submit_tick[i])
        inst = Instance(
            idx=i,
            instance_id=str(row["instance_id"]),
            collection_id=int(row["collection_id"]),
            instance_index=int(row["instance_index"]),
            service=str(row["service"]),
            docker_image=str(row["docker_image"]),
            cpu=float(row["cpu_request"]),
            mem=float(row["memory_request"]),
            priority=int(row["priority"]),
            sched_class=int(row["scheduling_class"]),
            submit_tick=t,
            duration_ticks=int(duration_ticks[i]),
            real_failed=bool(row["failed"]),
            real_killed=bool(row["killed"]),
        )
        specs.append(inst)
        arrivals.setdefault(t, []).append(i)

    return EpisodeSpec(instances=specs, arrivals_by_tick=arrivals, n_ticks=episode_ticks)


def select_nodes(machines: pd.DataFrame, max_nodes: int, seed: int = 0) -> pd.DataFrame:
    """Pick a reproducible subset of machines with the largest CPU capacity."""
    df = machines.dropna(subset=["cpu_capacity"]).copy()
    df["cpu_capacity"] = df["cpu_capacity"].clip(lower=1e-3)
    df["memory_capacity"] = df["memory_capacity"].clip(lower=1e-3)
    if len(df) > max_nodes:
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(df), size=max_nodes, replace=False)
        df = df.iloc[np.sort(idx)]
    return df.reset_index(drop=True)
