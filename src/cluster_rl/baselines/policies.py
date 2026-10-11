"""Heuristic baseline placement policies and a rollout helper."""

from __future__ import annotations

import numpy as np

from cluster_rl.sim.traffic import TrafficModel


class Policy:
    name = "policy"

    def act(self, env, obs, mask: np.ndarray) -> int:
        raise NotImplementedError


def _feasible_nodes(env, mask: np.ndarray) -> np.ndarray:
    return np.flatnonzero(mask[: env.n_nodes])


class RandomPolicy(Policy):
    name = "random"

    def __init__(self, seed: int = 0) -> None:
        self.rng = np.random.default_rng(seed)

    def act(self, env, obs, mask):
        nodes = _feasible_nodes(env, mask)
        if nodes.size == 0:
            return env.HOLD
        return int(self.rng.choice(nodes))


class RoundRobinPolicy(Policy):
    name = "round_robin"

    def __init__(self) -> None:
        self.cursor = 0

    def act(self, env, obs, mask):
        nodes = _feasible_nodes(env, mask)
        if nodes.size == 0:
            return env.HOLD
        self.cursor = (self.cursor + 1) % env.n_nodes
        for offset in range(env.n_nodes):
            node = (self.cursor + offset) % env.n_nodes
            if mask[node]:
                self.cursor = node
                return node
        return env.HOLD


class FirstFitPolicy(Policy):
    name = "first_fit"

    def act(self, env, obs, mask):
        nodes = _feasible_nodes(env, mask)
        return int(nodes[0]) if nodes.size else env.HOLD


class BestFitPolicy(Policy):
    """Tightest fit: minimise free capacity left after placement."""

    name = "best_fit"

    def act(self, env, obs, mask):
        inst = env.episode.instances[env.pending[0]]
        rem = (env.cluster.cpu_cap - env.cluster.cpu_used) + (
            env.cluster.mem_cap - env.cluster.mem_used
        )
        rem = rem - (inst.cpu + inst.mem)
        rem = np.where(mask[: env.n_nodes], rem, np.inf)
        return int(np.argmin(rem)) if np.isfinite(rem).any() else env.HOLD


class WorstFitPolicy(Policy):
    """Most free capacity."""

    name = "worst_fit"

    def act(self, env, obs, mask):
        free = (env.cluster.cpu_cap - env.cluster.cpu_used) + (
            env.cluster.mem_cap - env.cluster.mem_used
        )
        free = np.where(mask[: env.n_nodes], free, -np.inf)
        return int(np.argmax(free)) if np.isfinite(free).any() else env.HOLD


class LeastRequestedPolicy(Policy):
    """Kubernetes-style score: least requested resources."""

    name = "least_requested"

    def act(self, env, obs, mask):
        score = env.cluster.cpu_util() + env.cluster.mem_util()
        score = np.where(mask[: env.n_nodes], score, np.inf)
        return int(np.argmin(score)) if np.isfinite(score).any() else env.HOLD


class RiskAwarePolicy(Policy):
    """Greedy on failure risk + workload spread."""

    name = "risk_aware"

    def __init__(self, w_fail: float = 5.0, w_affinity: float = 0.5, w_balance: float = 2.0):
        self.w_fail = w_fail
        self.w_affinity = w_affinity
        self.w_balance = w_balance

    def act(self, env, obs, mask):
        nodes = _feasible_nodes(env, mask)
        if nodes.size == 0:
            return env.HOLD
        inst = env.episode.instances[env.pending[0]]
        best_node, best_score = nodes[0], np.inf
        for node in nodes:
            p_fail = env.failure_model.placement_probability(inst, env.cluster, node)
            coloc = TrafficModel.service_colocation(
                env.cluster, int(node), inst.service, env.service_of
            )
            load = env.cluster.cpu_util()[node] + env.cluster.mem_util()[node]
            score = self.w_fail * p_fail + self.w_affinity * coloc + self.w_balance * load
            if score < best_score:
                best_score, best_node = score, node
        return int(best_node)


def run_episode(env, policy: Policy, seed: int = 0, max_steps: int = 100_000) -> dict:
    obs, _ = env.reset(seed=seed)
    total_reward = 0.0
    steps = 0
    terminated = truncated = False
    while not (terminated or truncated) and steps < max_steps:
        if not env.pending:
            obs, reward, terminated, truncated, _ = env.step(env.HOLD)
            total_reward += reward
            steps += 1
            continue
        mask = env.action_masks()
        action = policy.act(env, obs, mask)
        obs, reward, terminated, truncated, _ = env.step(action)
        total_reward += reward
        steps += 1
    metrics = env.episode_metrics()
    metrics["total_reward"] = total_reward
    metrics["steps"] = steps
    metrics["policy"] = policy.name
    return metrics


def default_policies(seed: int = 0) -> list[Policy]:
    return [
        RandomPolicy(seed=seed),
        RoundRobinPolicy(),
        FirstFitPolicy(),
        BestFitPolicy(),
        WorstFitPolicy(),
        LeastRequestedPolicy(),
        RiskAwarePolicy(),
    ]
