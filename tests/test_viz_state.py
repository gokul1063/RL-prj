import numpy as np

from cluster_rl.viz.state import (
    heat_color,
    kpis,
    node_cards,
    pending_card,
    service_color,
    snapshot,
)


def test_snapshot_keys(env):
    env.reset(seed=0)
    snap = snapshot(env)
    assert set(snap) == {
        "kpis",
        "pending",
        "nodes",
        "n_nodes",
        "hold_index",
        "mask",
        "reward_breakdown",
    }
    assert len(snap["nodes"]) == env.n_nodes
    assert len(snap["mask"]) == env.n_nodes + 1
    assert snap["hold_index"] == env.n_nodes


def test_node_cards_fields(env):
    env.reset(seed=1)
    for node in node_cards(env):
        assert 0.0 <= node["cpu_util"] <= 1.0 + 1e-6
        assert 0.0 <= node["mem_util"] <= 1.0 + 1e-6
        assert isinstance(node["feasible"], bool)
        assert isinstance(node["containers"], list)


def test_pending_and_kpis_after_step(env):
    env.reset(seed=2)
    p = pending_card(env)
    assert p is not None and "docker_image" in p and "base_risk" in p
    mask = env.action_masks()
    env.step(int(np.flatnonzero(mask)[0]))
    k = kpis(env)
    assert k["scheduled"] >= 1
    assert "imbalance" in k and "traffic_imbalance" in k


def test_reward_breakdown_recorded(env):
    env.reset(seed=3)
    mask = env.action_masks()
    env.step(int(np.flatnonzero(mask)[0]))
    assert env.last_reward_breakdown is not None
    assert "r_total" in env.last_reward_breakdown


def test_colors():
    assert service_color("frontend").startswith("#")
    assert service_color("frontend") == service_color("frontend")
    assert heat_color(0.0).startswith("rgb")
    assert heat_color(1.5).startswith("rgb")
