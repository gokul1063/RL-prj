"""Gymnasium environment package."""

from cluster_rl.env.config import EnvConfig
from cluster_rl.env.placement_env import PlacementEnv

__all__ = ["PlacementEnv", "EnvConfig", "register"]


def register() -> None:
    """Register the environment with Gymnasium (idempotent)."""
    import gymnasium as gym

    from gymnasium.envs.registration import registry

    if "ContainerPlacement-v0" not in registry:
        gym.register(
            id="ContainerPlacement-v0",
            entry_point="cluster_rl.env.placement_env:PlacementEnv",
        )
