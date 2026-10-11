"""Small, clearly-labelled synthetic overlays.

The Borg traces contain no docker image names and no client request traces. To
make the environment legible as a "docker on a cluster" problem we add:

* a deterministic ``collection_id -> (service, docker_image)`` mapping, and
* helpers to attach per-service request rates used by the traffic proxy.

These overlays do not alter the real resource-request or failure data.
"""

from __future__ import annotations

import hashlib

import numpy as np
import pandas as pd

IMAGE_VOCAB = [
    "nginx:1.25-alpine",
    "redis:7.2",
    "postgres:16",
    "python:3.12-slim",
    "node:20-alpine",
    "openjdk:21-slim",
    "envoyproxy/envoy:v1.29",
    "ghcr.io/acme/api:latest",
    "ghcr.io/acme/worker:latest",
    "grafana/grafana:10.4",
    "prom/prometheus:v2.52",
    "minio/minio:latest",
    "rabbitmq:3.13",
    "memcached:1.6",
    "traefik:v3.0",
]

SERVICE_VOCAB = [
    "frontend",
    "auth",
    "catalog",
    "cart",
    "checkout",
    "payments",
    "search",
    "recommendations",
    "notifications",
    "analytics",
    "media",
    "gateway",
]


def _stable_index(value: int, modulo: int) -> int:
    digest = hashlib.blake2b(str(int(value)).encode(), digest_size=8).digest()
    return int.from_bytes(digest, "big") % modulo


def attach_docker_metadata(instances: pd.DataFrame) -> pd.DataFrame:
    """Add ``service`` and ``docker_image`` columns derived from collection_id."""
    instances = instances.copy()
    collections = instances["collection_id"].astype("int64")
    instances["service"] = [
        SERVICE_VOCAB[_stable_index(c, len(SERVICE_VOCAB))] for c in collections
    ]
    instances["docker_image"] = [
        IMAGE_VOCAB[_stable_index(c, len(IMAGE_VOCAB))] for c in collections
    ]
    return instances


def service_request_rates(
    instances: pd.DataFrame, scale: float = 1.0, seed: int = 0
) -> dict[str, float]:
    """Heavy-tailed per-service request rate (requests/tick) for the traffic proxy.

    Calibrated so the *sum* of rates tracks the observed count of production
    instances, then skewed by a Zipf-like factor to create realistic hotspots.
    """
    if "service" not in instances.columns:
        instances = attach_docker_metadata(instances)
    prod = instances[instances["priority"] >= 9]
    weights = prod["service"].value_counts()
    if weights.empty:
        weights = instances["service"].value_counts()
    rng = np.random.default_rng(seed)
    zipf = rng.permutation(np.arange(1, len(weights) + 1))
    weights = pd.Series(zipf, index=weights.index[: len(zipf)])
    base = weights / weights.sum()
    return (base * float(scale) * len(instances)).to_dict()
