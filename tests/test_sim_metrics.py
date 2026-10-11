import numpy as np

from cluster_rl.sim.cluster import Cluster, Node
from cluster_rl.sim.events import build_episode, select_nodes
from cluster_rl.utils.metrics import (
    fragmentation,
    imbalance_cpu_mem,
    jains_fairness,
    shannon_entropy,
)


def test_jains_fairness():
    assert np.isclose(jains_fairness(np.ones(4)), 1.0)
    assert jains_fairness(np.array([1.0, 0.0, 0.0, 0.0])) < 0.4


def test_entropy():
    assert np.isclose(shannon_entropy(np.ones(4)), np.log(4))
    assert shannon_entropy(np.zeros(4)) == 0.0


def test_fragmentation_and_imbalance():
    c = Cluster([Node(0, 1.0, 1.0), Node(1, 1.0, 1.0)])
    assert np.isclose(fragmentation(c, 0.1, 0.1), 0.0)
    c.reserve(0, 1.0, 0.05, 0)
    c.reserve(1, 0.05, 1.0, 1)
    assert fragmentation(c, 0.1, 0.1) > 0.0
    assert imbalance_cpu_mem(c) > 0.0


def test_build_episode(instances):
    ep = build_episode(instances, episode_ticks=24, max_arrivals=100, seed=0)
    assert ep.n_instances <= 100
    assert all(0 <= i.submit_tick < 24 for i in ep.instances)
    total = sum(len(v) for v in ep.arrivals_by_tick.values())
    assert total == ep.n_instances


def test_select_nodes(instances):
    # reuse instances fixture only to get a frame; build machines-like frame
    import pandas as pd

    m = pd.DataFrame(
        {
            "machine_id": [1, 2, 3],
            "cpu_capacity": [0.5, 1.0, 0.25],
            "memory_capacity": [0.5, 1.0, 0.25],
            "platform_id": ["a", "b", "c"],
        }
    )
    sel = select_nodes(m, max_nodes=2, seed=0)
    assert len(sel) == 2
    assert sel["cpu_capacity"].max() > 0
