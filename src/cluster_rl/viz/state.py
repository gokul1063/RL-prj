"""Read-only adapter that turns live environment state into UI-friendly data.

Nothing here mutates the simulation; it is safe to call after any ``step``.
"""

from __future__ import annotations

import hashlib

import numpy as np

from cluster_rl.utils.metrics import imbalance_cpu_mem

SERVICE_PALETTE = [
    "#4C78A8", "#F58518", "#54A24B", "#E45756", "#72B7B2",
    "#B279A2", "#FF9DA6", "#9D755D", "#BAB0AC", "#EECA3B",
    "#8CD17D", "#F1CE63",
]


def service_color(service: str) -> str:
    idx = int.from_bytes(
        hashlib.blake2b(str(service).encode(), digest_size=4).digest(), "big"
    ) % len(SERVICE_PALETTE)
    return SERVICE_PALETTE[idx]


def heat_color(util: float) -> str:
    """Green (low) -> amber -> red (high/overloaded) utilization colour."""
    u = max(0.0, min(1.0, float(util)))
    if u < 0.5:
        # green to yellow
        t = u / 0.5
        r = int(84 + t * (241 - 84))
        g = int(162 + t * (206 - 162))
        b = int(74 + t * (99 - 74))
    else:
        t = (u - 0.5) / 0.5
        r = int(241 + t * (228 - 241))
        g = int(206 + t * (86 - 206))
        b = int(99 + t * (86 - 99))
    return f"rgb({r},{g},{b})"


def _container_cards(env, node: int) -> list[dict]:
    cards = []
    for iid in env.cluster.hosted[node]:
        inst = env.episode.instances[iid]
        cards.append(
            {
                "iid": iid,
                "docker_image": inst.docker_image,
                "service": inst.service,
                "cpu": inst.cpu,
                "mem": inst.mem,
                "failed": bool(inst.failed),
                "p_fail": float(inst.p_fail),
            }
        )
    return cards


def pending_card(env) -> dict | None:
    if not env.pending:
        return None
    inst = env.episode.instances[env.pending[0]]
    return {
        "iid": inst.idx,
        "instance_id": inst.instance_id,
        "docker_image": inst.docker_image,
        "service": inst.service,
        "cpu": inst.cpu,
        "mem": inst.mem,
        "priority": inst.priority,
        "sched_class": inst.sched_class,
        "duration_ticks": inst.duration_ticks,
        "wait_ticks": env.current_tick - inst.submit_tick,
        "deferred": inst.deferred,
        "base_risk": float(
            env.failure_model.base_probability(
                inst.cpu, inst.mem, inst.priority, inst.sched_class
            )
        ),
    }


def node_cards(env) -> list[dict]:
    cpu_util = env.cluster.cpu_util()
    mem_util = env.cluster.mem_util()
    mask = env.action_masks()
    cards = []
    for node in range(env.n_nodes):
        cards.append(
            {
                "index": node,
                "machine_id": env.cluster.nodes[node].machine_id,
                "cpu_util": float(cpu_util[node]),
                "mem_util": float(mem_util[node]),
                "cpu_used": float(env.cluster.cpu_used[node]),
                "cpu_cap": float(env.cluster.cpu_cap[node]),
                "mem_used": float(env.cluster.mem_used[node]),
                "mem_cap": float(env.cluster.mem_cap[node]),
                "running_count": int(env.cluster.running_count[node]),
                "recent_fail": float(env.cluster.recent_fail[node]),
                "feasible": bool(mask[node]),
                "containers": _container_cards(env, node),
            }
        )
    return cards


def kpis(env) -> dict:
    m = env.metrics
    sched = m.scheduled
    total = sched + m.unscheduled
    waits = np.asarray(m.wait_ticks) if m.wait_ticks else np.array([0.0])
    traffic = float("nan")
    if m.traffic_n > 0:
        traffic = m.traffic_imbalance_sum / m.traffic_n
    return {
        "tick": env.current_tick,
        "episode_ticks": env.cfg.episode_ticks,
        "steps": env._steps,
        "pending": len(env.pending),
        "running": len(env.running),
        "scheduled": sched,
        "failed": m.failed,
        "unscheduled": m.unscheduled,
        "deferred": m.deferred,
        "cpu_util_mean": float(env.cluster.cpu_util().mean()),
        "mem_util_mean": float(env.cluster.mem_util().mean()),
        "imbalance": imbalance_cpu_mem(env.cluster),
        "failure_rate": float(m.failed / max(sched, 1)),
        "sla_violation_rate": float(m.unscheduled / max(total, 1)),
        "wait_p95": float(np.percentile(waits, 95)),
        "traffic_imbalance": traffic,
        "done": bool(env._done),
    }


def snapshot(env) -> dict:
    """Full UI state: KPIs, pending container, node cards, mask, reward breakdown."""
    mask = env.action_masks()
    return {
        "kpis": kpis(env),
        "pending": pending_card(env),
        "nodes": node_cards(env),
        "n_nodes": env.n_nodes,
        "hold_index": env.HOLD,
        "mask": mask.tolist(),
        "reward_breakdown": env.last_reward_breakdown,
    }
