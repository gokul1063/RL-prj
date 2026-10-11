"""Unified schema shared by every trace adapter.

Every adapter must emit two pandas DataFrames that conform to the column sets
below. This is the contract the simulator and the Gymnasium environment rely on;
adapters may add extra columns but must never drop or rename these.
"""

from __future__ import annotations

# --- machine table -----------------------------------------------------------
MACHINE_COLUMNS = [
    "machine_id",
    "cpu_capacity",   # normalized Borg units; largest machine in a cell == 1.0
    "memory_capacity",  # normalized
    "platform_id",
    "add_time",       # microseconds since trace start (float, NaN if unknown)
    "remove_time",    # microseconds since trace start (float, NaN if still up)
]

# --- instance (task/container) table ----------------------------------------
INSTANCE_COLUMNS = [
    "instance_id",       # str: f"{collection_id}:{instance_index}"
    "collection_id",     # job / service id
    "instance_index",    # task index inside the collection
    "machine_id",        # machine observed in the trace (reference only)
    "submit_time",       # us since trace start
    "schedule_time",     # us since trace start (NaN if never scheduled)
    "end_time",          # us since trace start (NaN if still running at cutoff)
    "terminal_event",    # 'submit'|'schedule'|'evict'|'fail'|'finish'|'kill'|'lost'|'running'
    "failed",            # bool: terminal_event in FAIL/EVICT/LOST
    "killed",            # bool: terminal_event == KILL (excluded from failure training)
    "cpu_request",       # normalized
    "memory_request",    # normalized
    "priority",          # 0..11 (higher == more important)
    "scheduling_class",  # 0..3 latency sensitivity
    "duration",          # us, schedule_time -> end_time
    "docker_image",      # synthetic overlay (trace has no images)
    "service",           # synthetic overlay derived from collection_id
]

# --- Borg 2011 event codes ---------------------------------------------------
TASK_EVENT_TYPES = {
    0: "submit",
    1: "schedule",
    2: "evict",
    3: "fail",
    4: "finish",
    5: "kill",
    6: "lost",
    7: "update_pending",
    8: "update_running",
}

MACHINE_EVENT_TYPES = {0: "add", 1: "remove", 2: "update"}

# Terminal states that count as an unsuccessful instance.
FAILURE_EVENTS = {"evict", "fail", "lost"}
# Terminal states that count as a clean completion.
SUCCESS_EVENTS = {"finish"}
# Terminal states excluded from failure modelling (user/scheduler initiated).
EXCLUDED_EVENTS = {"kill"}

# Priority tiers used by Borg 2011 and re-used in the synthetic docker overlay.
PRIORITY_TIERS = {
    "free": (0, 1),
    "best_effort": (2, 8),
    "production": (9, 11),
}


def priority_tier(priority: float) -> str:
    if priority <= 1:
        return "free"
    if priority <= 8:
        return "best_effort"
    return "production"


def normalize_timestamps(df, columns):
    """Shift microsecond timestamps so the table starts at 0."""
    import pandas as pd

    df = df.copy()
    start = None
    for c in columns:
        if c in df.columns and df[c].notna().any():
            candidate = df[c].min()
            start = candidate if start is None else min(start, candidate)
    if start is None:
        return df, 0.0
    for c in columns:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce") - start
    return df, float(start)
