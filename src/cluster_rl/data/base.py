"""Abstract trace adapter."""

from __future__ import annotations

from abc import ABC, abstractmethod

import pandas as pd


class TraceAdapter(ABC):
    """Produces unified machine and instance tables from a raw trace."""

    name: str = "abstract"

    @abstractmethod
    def load_machines(self) -> pd.DataFrame:
        """Return a DataFrame with columns from ``schema.MACHINE_COLUMNS``."""

    @abstractmethod
    def load_instances(self) -> pd.DataFrame:
        """Return a DataFrame with columns from ``schema.INSTANCE_COLUMNS``."""

    def describe(self) -> dict:
        machines = self.load_machines()
        instances = self.load_instances()
        return {
            "source": self.name,
            "n_machines": int(machines["machine_id"].nunique()),
            "n_instances": int(len(instances)),
            "failed_instances": int(instances["failed"].sum()),
            "span_us": float(instances["submit_time"].max() - instances["submit_time"].min()),
        }
