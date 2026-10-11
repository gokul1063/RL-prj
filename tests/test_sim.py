import numpy as np

from cluster_rl.sim.cluster import Cluster, Node


def test_reserve_release_roundtrip():
    nodes = [Node(0, 1.0, 1.0), Node(1, 0.5, 0.5)]
    c = Cluster(nodes)
    assert c.reserve(0, 0.4, 0.4, iid=0)
    assert not c.reserve(0, 0.7, 0.1, iid=1)  # cpu over capacity
    assert c.reserve(1, 0.4, 0.4, iid=2)
    mask = c.feasible_mask(0.3, 0.3)
    assert mask[0] and not mask[1]
    c.release(0, 0.4, 0.4, iid=0)
    assert np.isclose(c.cpu_used[0], 0.0)


def test_failure_ema_bounded():
    c = Cluster([Node(0, 1.0, 1.0)])
    c.reserve(0, 0.1, 0.1, 0)
    for k in range(50):
        c.reserve(0, 0.001, 0.001, 100 + k)
        c.release(0, 0.001, 0.001, 100 + k, failed=True)
    assert 0.0 <= c.recent_fail[0] <= 1.0
    assert c.recent_fail[0] > 0.0


def test_utilization():
    c = Cluster([Node(0, 1.0, 2.0)])
    assert c.reserve(0, 0.5, 1.0, 0)
    assert np.isclose(c.cpu_util()[0], 0.5)
    assert np.isclose(c.mem_util()[0], 0.5)
