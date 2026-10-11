import numpy as np
import pytest

sb3 = pytest.importorskip("sb3_contrib")
from sb3_contrib import MaskablePPO  # noqa: E402
from sb3_contrib.common.wrappers import ActionMasker  # noqa: E402
from stable_baselines3.common.vec_env import DummyVecEnv  # noqa: E402

from cluster_rl.baselines import run_episode  # noqa: E402
from cluster_rl.baselines.sb3_policy import SB3Policy  # noqa: E402
from cluster_rl.env import EnvConfig, PlacementEnv  # noqa: E402


def _tiny_cfg():
    return EnvConfig.from_dict(
        {
            "processed_dir": "unused",
            "max_nodes": 4,
            "episode_ticks": 12,
            "max_arrivals": 60,
            "max_wait_ticks": 4,
            "hold_max": 3,
            "max_steps_multiplier": 3,
        }
    )


def _mask_fn(env):
    return env.action_masks()


def _build(instances, machines, seed):
    cfg = _tiny_cfg()

    def thunk():
        env = PlacementEnv(cfg, instances=instances, machines=machines)
        env._rng = np.random.default_rng(seed)
        return ActionMasker(env, _mask_fn)

    return thunk


def test_ppo_training_and_rollout(instances, machines):
    vec = DummyVecEnv([_build(instances, machines, 1000 * i) for i in range(2)])
    model = MaskablePPO(
        "MlpPolicy",
        vec,
        n_steps=64,
        batch_size=32,
        learning_rate=3e-4,
        seed=0,
        verbose=0,
    )
    model.learn(total_timesteps=128)
    vec.close()

    policy = SB3Policy(model, name="test_rl")
    env = PlacementEnv(_tiny_cfg(), instances=instances, machines=machines)
    metrics = run_episode(env, policy, seed=0, max_steps=200)
    assert metrics["scheduled"] >= 0
    assert np.isfinite(metrics["total_reward"])