"""CLI: turn a raw trace into compact Parquet artifacts.

Example
-------
    python -m cluster_rl.data.preprocess \
        --source google2011 --parts 0-3 \
        --out data/processed/smoke
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from cluster_rl.data import get_adapter


def parse_parts(spec: str) -> list[int]:
    parts: list[int] = []
    for chunk in str(spec).split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        if "-" in chunk:
            lo, hi = chunk.split("-")
            parts.extend(range(int(lo), int(hi) + 1))
        else:
            parts.append(int(chunk))
    return sorted(set(parts))


def preprocess(
    source: str,
    data_dir: str,
    parts: str,
    out: str,
    max_instances: int | None = None,
) -> Path:
    out_dir = Path(out)
    out_dir.mkdir(parents=True, exist_ok=True)
    adapter = get_adapter(
        source,
        data_dir=data_dir,
        parts=parse_parts(parts),
        max_instances=max_instances,
    )
    machines = adapter.load_machines()
    instances = adapter.load_instances()

    machines.to_parquet(out_dir / "machines.parquet", index=False)
    instances.to_parquet(out_dir / "instances.parquet", index=False)

    meta = {
        "source": source,
        "parts": parse_parts(parts),
        "n_machines": int(machines["machine_id"].nunique()),
        "n_instances": int(len(instances)),
        "n_failed": int(instances["failed"].sum()),
        "n_killed": int(instances["killed"].sum()),
        "span_us": float(
            instances["submit_time"].max() - instances["submit_time"].min()
        ),
        "submit_time_max_us": float(instances["submit_time"].max()),
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    print(
        f"[preprocess] {source} parts={parse_parts(parts)} -> {out_dir}\n"
        f"  machines={meta['n_machines']} instances={meta['n_instances']} "
        f"failed={meta['n_failed']} killed={meta['n_killed']} "
        f"span={meta['span_us'] / 3.6e9:.2f}h"
    )
    return out_dir


def main() -> None:
    ap = argparse.ArgumentParser(description="Preprocess a cluster trace to Parquet")
    ap.add_argument("--source", default="google2011")
    ap.add_argument("--data-dir", default="data/raw/google2011")
    ap.add_argument("--parts", default="0-3")
    ap.add_argument("--out", default="data/processed/default")
    ap.add_argument("--max-instances", type=int, default=None)
    args = ap.parse_args()
    preprocess(
        args.source, args.data_dir, args.parts, args.out, args.max_instances
    )


if __name__ == "__main__":
    main()
