"""Google Borg 2011 trace adapter (``clusterdata-2011-2``).

The trace is a single 12.5k-machine cell from May 2011. The 500 ``task_events``
shards are sequential in time, so a contiguous ``parts`` range yields a coherent
replay window. Files download over plain HTTP with no authentication.

Reference: https://github.com/google/cluster-data/blob/master/ClusterData2011_2.md
"""

from __future__ import annotations

import os
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

from cluster_rl.data.base import TraceAdapter
from cluster_rl.data.schema import (
    FAILURE_EVENTS,
    MACHINE_COLUMNS,
    TASK_EVENT_TYPES,
    normalize_timestamps,
)

BUCKET = "https://storage.googleapis.com/clusterdata-2011-2"
TASK_EVENT_COLUMNS = [
    "timestamp",
    "missing_info",
    "job_id",
    "task_index",
    "machine_id",
    "event_type",
    "user",
    "scheduling_class",
    "priority",
    "cpu_request",
    "memory_request",
    "disk_request",
    "constraint",
]
MACHINE_EVENT_COLUMNS = [
    "timestamp",
    "machine_id",
    "event_type",
    "platform_id",
    "cpu_capacity",
    "memory_capacity",
]


def _download(url: str, dest: Path) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 0:
        return dest
    tmp = dest.with_suffix(dest.suffix + ".part")
    with urllib.request.urlopen(url, timeout=120) as resp, open(tmp, "wb") as fh:
        while True:
            chunk = resp.read(1 << 20)
            if not chunk:
                break
            fh.write(chunk)
    tmp.rename(dest)
    return dest


class Google2011Adapter(TraceAdapter):
    """Parse the Borg 2011 trace into the unified schema.

    Parameters
    ----------
    data_dir:
        Directory holding ``machine_events/`` and ``task_events/`` shards.
    parts:
        Contiguous ``task_events`` shard indices to load (sequential in time).
    download:
        If True, fetch missing shards from the public GCS bucket.
    max_instances:
        Optional cap on reconstructed instances (useful for smoke runs).
    """

    name = "google2011"

    def __init__(
        self,
        data_dir: str | os.PathLike = "data/raw/google2011",
        parts: list[int] | None = None,
        download: bool = True,
        max_instances: int | None = None,
    ) -> None:
        self.data_dir = Path(data_dir)
        self.parts = [0] if parts is None else list(parts)
        self.download = download
        self.max_instances = max_instances
        self._machines: pd.DataFrame | None = None
        self._instances: pd.DataFrame | None = None

    # -- files ---------------------------------------------------------------
    def _machine_file(self) -> Path:
        rel = "machine_events/part-00000-of-00001.csv.gz"
        dest = self.data_dir / rel
        return _download(f"{BUCKET}/{rel}", dest) if self.download else dest

    def _task_file(self, part: int) -> Path:
        rel = f"task_events/part-{part:05d}-of-00500.csv.gz"
        dest = self.data_dir / rel
        return _download(f"{BUCKET}/{rel}", dest) if self.download else dest

    # -- machines ------------------------------------------------------------
    def load_machines(self) -> pd.DataFrame:
        if self._machines is not None:
            return self._machines
        me = pd.read_csv(
            self._machine_file(), header=None, names=MACHINE_EVENT_COLUMNS
        )
        me["timestamp"] = pd.to_numeric(me["timestamp"])
        add = (
            me[me.event_type == 0]
            .groupby("machine_id", as_index=False)["timestamp"]
            .min()
            .rename(columns={"timestamp": "add_time"})
        )
        remove = (
            me[me.event_type == 1]
            .groupby("machine_id", as_index=False)["timestamp"]
            .min()
            .rename(columns={"timestamp": "remove_time"})
        )
        latest = (
            me.sort_values("timestamp")
            .groupby("machine_id", as_index=False)
            .last()[["machine_id", "platform_id", "cpu_capacity", "memory_capacity"]]
        )
        machines = latest.merge(add, on="machine_id", how="left").merge(
            remove, on="machine_id", how="left"
        )
        machines = machines[MACHINE_COLUMNS].reset_index(drop=True)
        machines, base = normalize_timestamps(machines, ["add_time", "remove_time"])
        self._machines = machines
        self._machine_base = base
        return machines

    # -- instances -----------------------------------------------------------
    def _task_events(self) -> pd.DataFrame:
        frames = []
        for part in self.parts:
            frames.append(
                pd.read_csv(
                    self._task_file(part), header=None, names=TASK_EVENT_COLUMNS
                )
            )
        te = pd.concat(frames, ignore_index=True)
        for c in (
            "timestamp",
            "job_id",
            "task_index",
            "machine_id",
            "event_type",
            "scheduling_class",
            "priority",
            "cpu_request",
            "memory_request",
        ):
            te[c] = pd.to_numeric(te[c], errors="coerce")
        return te

    def load_instances(self) -> pd.DataFrame:
        if self._instances is not None:
            return self._instances
        self.load_machines()  # ensures _machine_base is set
        te = self._task_events()
        keys = ["job_id", "task_index"]

        te_sorted = te.sort_values(keys + ["timestamp"])
        first = te_sorted.drop_duplicates(keys, keep="first")[
            keys + ["timestamp", "scheduling_class", "priority"]
        ].rename(columns={"timestamp": "submit_time"})

        sched = te[te.event_type == 1].sort_values("timestamp")
        sched = sched.drop_duplicates(keys, keep="first")[
            keys + ["timestamp", "machine_id"]
        ].rename(columns={"timestamp": "schedule_time"})

        req = te.dropna(subset=["cpu_request"]).sort_values("timestamp")
        req = req.drop_duplicates(keys, keep="first")[
            keys + ["cpu_request", "memory_request"]
        ]

        terminal_mask = te.event_type.isin([2, 3, 4, 5, 6])
        term = te[terminal_mask].sort_values("timestamp")
        term = term.drop_duplicates(keys, keep="last")[
            keys + ["timestamp", "event_type"]
        ].rename(columns={"timestamp": "end_time", "event_type": "terminal_code"})

        inst = first.merge(sched, on=keys, how="left")
        inst = inst.merge(req, on=keys, how="left")
        inst = inst.merge(term, on=keys, how="left")

        inst["cpu_request"] = inst["cpu_request"].fillna(inst["cpu_request"].median())
        inst["memory_request"] = inst["memory_request"].fillna(
            inst["memory_request"].median()
        )
        inst["scheduling_class"] = inst["scheduling_class"].fillna(0).astype(int)
        inst["priority"] = inst["priority"].fillna(0).astype(int)

        inst["terminal_event"] = inst["terminal_code"].map(TASK_EVENT_TYPES)
        inst["terminal_event"] = inst["terminal_event"].fillna("running")
        inst["failed"] = inst["terminal_event"].isin(FAILURE_EVENTS)
        inst["killed"] = inst["terminal_event"].eq("kill")
        inst["duration"] = inst["end_time"] - inst["schedule_time"]

        inst["instance_id"] = (
            inst["job_id"].astype("int64").astype(str)
            + ":"
            + inst["task_index"].astype("int64").astype(str)
        )
        inst["collection_id"] = inst["job_id"].astype("int64")
        inst["instance_index"] = inst["task_index"].astype("int64")

        inst, _ = normalize_timestamps(
            inst, ["submit_time", "schedule_time", "end_time"]
        )
        inst = inst.drop(columns=["terminal_code"])

        # synthetic docker/service overlay (see schema.py: trace has no images)
        from cluster_rl.data.synthetic import attach_docker_metadata

        inst = attach_docker_metadata(inst)

        if self.max_instances is not None and len(inst) > self.max_instances:
            inst = inst.sample(n=self.max_instances, random_state=0).reset_index(
                drop=True
            )
        self._instances = inst
        return inst
