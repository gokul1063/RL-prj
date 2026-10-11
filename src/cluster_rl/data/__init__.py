"""Data adapters that turn real cluster traces into a unified tabular schema."""

from cluster_rl.data.base import TraceAdapter
from cluster_rl.data.google2011 import Google2011Adapter

__all__ = ["TraceAdapter", "Google2011Adapter", "get_adapter"]


def get_adapter(source: str, **kwargs) -> TraceAdapter:
    """Return a trace adapter by name.

    ``google2011`` is the zero-auth, directly downloadable Borg trace and the
    Phase 1 default. ``google2019_bq`` is an optional BigQuery-backed adapter
    (requires GCP credentials) exposed behind the same interface.
    """
    if source in ("google2011", "borg2011"):
        return Google2011Adapter(**kwargs)
    if source in ("google2019", "google2019_bq", "borg2019"):
        from cluster_rl.data.google2019_bq import Google2019BigQueryAdapter

        return Google2019BigQueryAdapter(**kwargs)
    raise ValueError(f"unknown trace source: {source!r}")
