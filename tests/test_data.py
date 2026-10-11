import pandas as pd

from cluster_rl.data.preprocess import parse_parts
from cluster_rl.data.synthetic import attach_docker_metadata, service_request_rates


def test_parse_parts():
    assert parse_parts("0-3") == [0, 1, 2, 3]
    assert parse_parts("1,2,5") == [1, 2, 5]
    assert parse_parts("0-1,4") == [0, 1, 4]


def test_attach_docker_metadata(instances):
    out = attach_docker_metadata(instances)
    assert "service" in out.columns and "docker_image" in out.columns
    # deterministic
    again = attach_docker_metadata(instances)
    assert (out["service"] == again["service"]).all()


def test_service_request_rates(instances):
    rates = service_request_rates(instances, scale=1.0, seed=0)
    assert set(rates).issubset(set(instances["service"].unique()))
    assert all(v >= 0 for v in rates.values())


def test_processed_columns_if_present():
    from pathlib import Path

    from cluster_rl.data.schema import INSTANCE_COLUMNS, MACHINE_COLUMNS

    p = Path("data/processed/smoke")
    if (p / "instances.parquet").exists():
        inst = pd.read_parquet(p / "instances.parquet")
        assert set(INSTANCE_COLUMNS).issubset(inst.columns)
    if (p / "machines.parquet").exists():
        mach = pd.read_parquet(p / "machines.parquet")
        assert set(MACHINE_COLUMNS).issubset(mach.columns)
