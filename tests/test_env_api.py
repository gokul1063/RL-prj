import numpy as np
from gymnasium.utils.env_checker import check_env

from cluster_rl.baselines import default_policies, run_episode


def test_check_env(env):
    check_env(env, skip_render_check=True)


def test_reset_step_api(env):
    obs, info = env.reset(seed=0)
    assert env.observation_space.contains(obs)
    mask = env.action_masks()
    assert mask.shape == (env.n_nodes + 1,)
    assert mask.dtype == bool
    valid = np.flatnonzero(mask)
    obs, reward, terminated, truncated, info = env.step(int(valid[0]))
    assert env.observation_space.contains(obs)
    assert isinstance(reward, float)
    assert isinstance(terminated, bool) and isinstance(truncated, bool)


def test_hold_allowed(env):
    env.reset(seed=0)
    assert env.action_masks()[env.HOLD]


def test_baselines_run(env):
    for policy in default_policies(seed=0):
        m = run_episode(env, policy, seed=1)
        assert m["scheduled"] >= 0
        assert 0.0 <= m["failure_rate"] <= 1.0
        assert np.isfinite(m["total_reward"])
        assert "traffic_imbalance" in m


def test_episode_terminates(env):
    env.reset(seed=3)
    steps = 0
    done = False
    while not done and steps < 100_000:
        mask = env.action_masks()
        action = int(np.flatnonzero(mask)[0])
        _, _, term, trunc, _ = env.step(action)
        done = term or trunc
        steps += 1
    assert done
    assert steps > 0
