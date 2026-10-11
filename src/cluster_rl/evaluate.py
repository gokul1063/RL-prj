"""Evaluate baseline policies on the placement environment.

    python -m cluster_rl.evaluate --config configs/smoke.yaml --episodes 5
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

import numpy as np

from cluster_rl.baselines import default_policies, run_episode
from cluster_rl.env import EnvConfig, PlacementEnv

AGG_KEYS = [
    "failure_rate",
    "sla_violation_rate",
    "cpu_util_mean",
    "mem_util_mean",
    "imbalance_mean",
    "wait_mean_ticks",
    "wait_p95_ticks",
    "colocation_mean",
    "traffic_imbalance",
    "mean_latency",
    "total_reward",
]


def evaluate(
    config: EnvConfig,
    episodes: int,
    seed: int,
    extra_policies: Sequence = (),
) -> dict:
    env = PlacementEnv(config)
    results: dict[str, dict] = {}
    policies = [*default_policies(seed=seed), *list(extra_policies)]
    for policy in policies:
        runs = [run_episode(env, policy, seed=seed + e) for e in range(episodes)]
        agg = {}
        for k in AGG_KEYS:
            vals = [r.get(k, float("nan")) for r in runs]
            agg[k] = float(np.nanmean(vals))
        agg["scheduled"] = float(np.mean([r["scheduled"] for r in runs]))
        agg["unscheduled"] = float(np.mean([r["unscheduled"] for r in runs]))
        results[policy.name] = agg
    return results


def print_table(results: dict) -> None:
    header = (
        f"{'policy':16s} {'sched':>7s} {'unsched':>7s} {'fail':>6s} {'sla':>6s} "
        f"{'cpu':>6s} {'imbal':>6s} {'wait95':>7s} {'traffic':>8s} {'reward':>9s}"
    )
    print(header)
    print("-" * len(header))
    for name, m in results.items():
        print(
            f"{name:16s} {m['scheduled']:7.1f} {m['unscheduled']:7.1f} "
            f"{m['failure_rate']:6.3f} {m['sla_violation_rate']:6.3f} "
            f"{m['cpu_util_mean']:6.3f} {m['imbalance_mean']:6.3f} "
            f"{m['wait_p95_ticks']:7.1f} {m['traffic_imbalance']:8.3f} "
            f"{m['total_reward']:9.1f}"
        )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/smoke.yaml")
    ap.add_argument("--episodes", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--output", default=None)
    ap.add_argument(
        "--rl-model",
        default=None,
        help="Path to a trained MaskablePPO .zip to compare against baselines",
    )
    args = ap.parse_args()

    config = EnvConfig.from_yaml(args.config)
    extra = []
    if args.rl_model:
        from sb3_contrib import MaskablePPO

        from cluster_rl.baselines.sb3_policy import SB3Policy

        model = MaskablePPO.load(args.rl_model)
        extra.append(SB3Policy(model, name=f"rl:{Path(args.rl_model).stem}"))
    results = evaluate(config, args.episodes, args.seed, extra_policies=extra)
    print_table(results)
    if args.output:
        Path(args.output).write_text(json.dumps(results, indent=2))
        print(f"\nwrote {args.output}")


if __name__ == "__main__":
    main()
