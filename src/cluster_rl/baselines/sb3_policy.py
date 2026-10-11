"""Adapter that fits a trained SB3 policy into the rollout harness."""

from __future__ import annotations

from cluster_rl.baselines.policies import Policy


class SB3Policy(Policy):
    """Wraps a loaded ``MaskablePPO`` (or any model with ``predict``)."""

    def __init__(self, model, name: str = "rl_ppo") -> None:
        self.model = model
        self.name = name

    def act(self, env, obs, mask):
        action, _ = self.model.predict(
            obs, action_masks=mask, deterministic=True
        )
        return int(action)