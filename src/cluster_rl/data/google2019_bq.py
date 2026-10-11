"""Optional BigQuery adapter for the Google Borg 2019 trace.

The 2019 trace is only distributed through BigQuery (project
``google.com:google-cluster-data``, cells a..h) and requires a GCP project with
billing. This adapter is a thin wrapper that reuses the 2011 parsing logic: it
pulls an event window into pandas and maps it onto the unified schema.

It is intentionally optional: Phase 1 ships and runs on ``google2011``. Enable
this path once credentials exist (``pip install google-cloud-bigquery``).
"""

from __future__ import annotations

import os

import pandas as pd

from cluster_rl.data.base import TraceAdapter

PROJECT = "google.com:google-cluster-data"

INSTANCE_EVENT_QUERY = """
SELECT
  time, type, collection_id, instance_index, machine_id,
  scheduling_class, priority,
  resource_request.cpus AS cpu_request,
  resource_request.memory AS memory_request
FROM `{project}`.clusterdata_2019_{cell}.instance_events
WHERE time BETWEEN @t0 AND @t1
"""

MACHINE_EVENT_QUERY = """
SELECT
  time, machine_id, type,
  capacity.cpus AS cpu_capacity,
  capacity.memory AS memory_capacity,
  platform_id
FROM `{project}`.clusterdata_2019_{cell}.machine_events
WHERE time BETWEEN @t0 AND @t1
"""


class Google2019BigQueryAdapter(TraceAdapter):
    name = "google2019_bq"

    def __init__(
        self,
        cell: str = "a",
        t0_us: int = 0,
        t1_us: int = 24 * 3600 * 1_000_000,
        credentials_path: str | None = None,
    ) -> None:
        self.cell = cell
        self.t0_us = int(t0_us)
        self.t1_us = int(t1_us)
        self.credentials_path = credentials_path
        self._client = None

    def _bq(self):
        try:
            from google.cloud import bigquery
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise ImportError(
                "google-cloud-bigquery is required for the 2019 adapter; "
                "install it and provide GCP credentials."
            ) from exc
        if self.credentials_path:
            os.environ.setdefault(
                "GOOGLE_APPLICATION_CREDENTIALS", self.credentials_path
            )
        if self._client is None:
            self._client = bigquery.Client()
        return self._client

    def _query(self, sql: str) -> pd.DataFrame:
        from google.cloud import bigquery

        client = self._bq()
        job_config = bigquery.QueryJobConfig(
            query_parameters=[
                bigquery.ScalarQueryParameter("t0", "INT64", self.t0_us),
                bigquery.ScalarQueryParameter("t1", "INT64", self.t1_us),
            ]
        )
        return client.query(sql, job_config=job_config).to_dataframe()

    def load_machines(self) -> pd.DataFrame:
        me = self._query(MACHINE_EVENT_QUERY.format(project=PROJECT, cell=self.cell))
        latest = (
            me.sort_values("time")
            .groupby("machine_id", as_index=False)
            .last()[["machine_id", "platform_id", "cpu_capacity", "memory_capacity"]]
        )
        latest["add_time"] = self.t0_us
        latest["remove_time"] = float("nan")
        return latest

    def load_instances(self) -> pd.DataFrame:
        ie = self._query(INSTANCE_EVENT_QUERY.format(project=PROJECT, cell=self.cell))
        # 2019 event types: 0 submit, 3 schedule, 4 evict, 5 fail, 6 finish, 7 kill, 8 lost
        term_map = {4: "evict", 5: "fail", 6: "finish", 7: "kill", 8: "lost"}
        ie = ie.sort_values("time")
        first = ie.drop_duplicates(["collection_id", "instance_index"], keep="first")
        first = first.rename(columns={"time": "submit_time"})
        sched = ie[ie.type == 3].drop_duplicates(
            ["collection_id", "instance_index"], keep="first"
        )
        sched = sched.rename(columns={"time": "schedule_time"})
        term = ie[ie.type.isin(term_map)].drop_duplicates(
            ["collection_id", "instance_index"], keep="last"
        )
        term = term.rename(columns={"time": "end_time", "type": "terminal_code"})
        inst = first.merge(
            sched[["collection_id", "instance_index", "schedule_time", "machine_id"]],
            on=["collection_id", "instance_index"],
            how="left",
        ).merge(
            term[["collection_id", "instance_index", "end_time", "terminal_code"]],
            on=["collection_id", "instance_index"],
            how="left",
        )
        inst["terminal_event"] = inst["terminal_code"].map(term_map).fillna("running")
        inst["failed"] = inst["terminal_event"].isin(["evict", "fail", "lost"])
        inst["killed"] = inst["terminal_event"].eq("kill")
        inst["duration"] = inst["end_time"] - inst["schedule_time"]
        inst["instance_id"] = (
            inst["collection_id"].astype(str)
            + ":"
            + inst["instance_index"].astype(str)
        )
        from cluster_rl.data.synthetic import attach_docker_metadata

        return attach_docker_metadata(inst)
