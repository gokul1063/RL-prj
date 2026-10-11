"""Gymnasium environment for online container placement on a cluster.

See ``docs/ENV_SPEC.md`` for the full rules, reward and policy contract.
"""

from __future__ import annotations

import heapq
from collections import deque
from pathlib import Path

import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium import spaces

from cluster_rl.data.synthetic import service_request_rates
from cluster_rl.env.config import EnvConfig
from cluster_rl.env.observation import ObservationBuilder
from cluster_rl.env.reward import RewardConfig, RewardShaper
from cluster_rl.sim.cluster import Cluster, Node
from cluster_rl.sim.events import Instance, build_episode, select_nodes
from cluster_rl.sim.failure_model import FailureModel
from cluster_rl.sim.traffic import TrafficModel
from cluster_rl.utils.metrics import (
    EpisodeMetrics,
    fragmentation,
    imbalance_cpu_mem,
)


class PlacementEnv(gym.Env):
    """Assign each pending docker container to one node, or defer it.

    Action space is ``Discrete(n_nodes + 1)``: indices ``0..n_nodes-1`` place the
    current instance on that node; index ``n_nodes`` defers (HOLD). Infeasible
    placements are masked via :meth:`action_masks`.
    """

    metadata = {"render_modes": ["human", "ansi"]}

    def __init__(
        self,
        config: EnvConfig | dict | None = None,
        instances: pd.DataFrame | None = None,
        machines: pd.DataFrame | None = None,
        failure_model: FailureModel | None = None,
        render_mode: str | None = None,
    ) -> None:
        super().__init__()
        if config is None:
            self.cfg = EnvConfig()
        elif isinstance(config, EnvConfig):
            self.cfg = config
        else:
            self.cfg = EnvConfig.from_dict(config)
        self.render_mode = render_mode

        self.instances_df = instances if instances is not None else self._load("instances.parquet")
        self.machines_df = machines if machines is not None else self._load("machines.parquet")

        self.node_df = select_nodes(self.machines_df, self.cfg.max_nodes, seed=self.cfg.seed)
        self.nodes = [
            Node(
                machine_id=int(r.machine_id),
                cpu_capacity=float(r.cpu_capacity),
                memory_capacity=float(r.memory_capacity),
                platform_id=str(getattr(r, "platform_id", "")),
            )
            for r in self.node_df.itertuples()
        ]
        self.n_nodes = len(self.nodes)

        if failure_model is None:
            fm = FailureModel(**self.cfg.failure)
            fm.fit(self.instances_df)
            self.failure_model = fm
        else:
            fm = failure_model
            fm.beta = self.cfg.failure.get("beta", fm.beta)
            fm.base_scale = self.cfg.failure.get("base_scale", fm.base_scale)
            fm.recent_fail_weight = self.cfg.failure.get(
                "recent_fail_weight", fm.recent_fail_weight
            )
            self.failure_model = fm

        self.services = sorted(self.instances_df["service"].unique().tolist())
        self.service_index = {s: i for i, s in enumerate(self.services)}

        self.max_cpu_cap = float(self.node_df["cpu_capacity"].max())
        self.max_mem_cap = float(self.node_df["memory_capacity"].max())
        self.max_running = float(max(1.0, self.cfg.max_arrivals * self.cfg.tick_seconds / 300.0))
        self.ref_cpu = float(np.percentile(self.instances_df["cpu_request"], 5)) or 1e-3
        self.ref_mem = float(np.percentile(self.instances_df["memory_request"], 5)) or 1e-3

        self.obs_builder = ObservationBuilder(
            n_nodes=self.n_nodes,
            max_cpu_cap=self.max_cpu_cap,
            max_mem_cap=self.max_mem_cap,
            episode_ticks=self.cfg.episode_ticks,
            max_arrivals=self.cfg.max_arrivals,
            n_services=len(self.services),
            max_running=self.max_running,
            tick_seconds=self.cfg.tick_seconds,
        )
        self.observation_space = spaces.Box(
            low=0.0, high=2.0, shape=(self.obs_builder.dim,), dtype=np.float32
        )
        self.HOLD = self.n_nodes
        self.action_space = spaces.Discrete(self.n_nodes + 1)

        self.reward_shaper = RewardShaper(RewardConfig.from_dict(self.cfg.reward))
        self.traffic_cfg = self.cfg.traffic
        self.traffic_sample_every = int(self.traffic_cfg.get("sample_every", 25))
        rates = service_request_rates(
            self.instances_df,
            scale=self.traffic_cfg.get("request_rate_scale", 4.0),
            seed=self.cfg.seed,
        )
        self.traffic_model = TrafficModel(
            rates,
            mu_per_cpu=self.traffic_cfg.get("mu_per_cpu", 20.0),
            mu_scale=self.traffic_cfg.get("mu_scale", 1.0),
        )

        self._rng = np.random.default_rng(self.cfg.seed)
        self.cluster = Cluster(self.nodes, reserve_ratio=self.cfg.reserve_ratio)

    # -- setup helpers -------------------------------------------------------
    def _load(self, name: str) -> pd.DataFrame:
        path = Path(self.cfg.processed_dir) / name
        if not path.exists():
            raise FileNotFoundError(
                f"{path} not found. Run: python -m cluster_rl.data.preprocess "
                f"--parts 0-3 --out {self.cfg.processed_dir}"
            )
        return pd.read_parquet(path)

    def _reset_runtime(self) -> None:
        self.cluster = Cluster(self.nodes, reserve_ratio=self.cfg.reserve_ratio)
        self.current_tick = 0
        self.pending: deque[int] = deque()
        self.running: dict[int, Instance] = {}
        self.heap: list[tuple[int, int]] = []
        self.service_of: list[str] = [inst.service for inst in self.episode.instances]
        self._arrival_ticks = sorted(self.episode.arrivals_by_tick.keys())
        self._arr_ptr = 0
        self.metrics = EpisodeMetrics()
        self._steps = 0
        self._done = False
        self._last_ticks_processed = 0
        self.last_reward_breakdown: dict | None = None

    # -- gym API -------------------------------------------------------------
    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self.episode = build_episode(
            self.instances_df,
            tick_seconds=self.cfg.tick_seconds,
            episode_ticks=self.cfg.episode_ticks,
            max_arrivals=self.cfg.max_arrivals,
            seed=int(self._rng.integers(2**31)),
        )
        if options and options.get("resample_nodes"):
            self.node_df = select_nodes(
                self.machines_df, self.cfg.max_nodes, seed=int(self._rng.integers(2**31))
            )
            self.nodes = [
                Node(
                    machine_id=int(r.machine_id),
                    cpu_capacity=float(r.cpu_capacity),
                    memory_capacity=float(r.memory_capacity),
                    platform_id=str(getattr(r, "platform_id", "")),
                )
                for r in self.node_df.itertuples()
            ]
            self.max_cpu_cap = float(self.node_df["cpu_capacity"].max())
            self.max_mem_cap = float(self.node_df["memory_capacity"].max())
        self._reset_runtime()
        self._fill_arrivals(0)
        self._auto_advance_if_needed()
        return self._obs(), self._info()

    def step(self, action: int):
        self._steps += 1
        self._auto_advance_if_needed()
        if self._done:
            terminated = True
            return self._obs(), 0.0, terminated, False, self._info()

        drop_reward = self._drop_stale()

        if not self.pending:
            self._auto_advance_if_needed()
            if self._done:
                return self._obs(), drop_reward, True, False, self._info()
            if not self.pending:
                # nothing placeable yet; treat as a no-op step
                br = self.reward_shaper.hold(0.0)
                self.last_reward_breakdown = br.as_dict()
                reward = drop_reward + br.total
                truncated = self._steps >= self._max_steps()
                return self._obs(), reward, False, truncated, self._info()

        inst = self.episode.instances[self.pending[0]]
        action = int(action)
        if action == self.HOLD:
            inst.deferred += 1
            self.metrics.deferred += 1
            br = self.reward_shaper.hold(self.current_tick - inst.submit_tick)
            self._advance_to_next_event()
            if inst.deferred >= self.cfg.hold_max:
                mask = self.cluster.feasible_mask(inst.cpu, inst.mem)
                if mask.any():
                    node = self._least_loaded(mask)
                    br = self._place(inst, node, forced=True)
        else:
            mask = self.cluster.feasible_mask(inst.cpu, inst.mem)
            if 0 <= action < self.n_nodes and mask[action]:
                br = self._place(inst, action)
            else:
                # invalid/unmasked action -> count as a hold with penalty
                inst.deferred += 1
                self.metrics.deferred += 1
                br = self.reward_shaper.hold(self.current_tick - inst.submit_tick)
                self._advance_to_next_event()

        reward = drop_reward + br.total
        self.last_reward_breakdown = br.as_dict()
        terminated = self._done
        truncated = self._steps >= self._max_steps()
        return self._obs(), float(reward), terminated, truncated, self._info()

    def action_masks(self) -> np.ndarray:
        mask = np.ones(self.n_nodes + 1, dtype=bool)
        inst = self.episode.instances[self.pending[0]] if self.pending else None
        if inst is not None:
            mask[: self.n_nodes] = self.cluster.feasible_mask(inst.cpu, inst.mem)
        mask[self.HOLD] = True
        return mask

    def render(self):
        util = self.cluster.cpu_util()
        lines = [
            f"tick={self.current_tick}/{self.cfg.episode_ticks} "
            f"steps={self._steps} pending={len(self.pending)} running={len(self.running)}",
            f"cpu_util={np.round(util, 3).tolist()}",
            f"scheduled={self.metrics.scheduled} failed={self.metrics.failed} "
            f"unscheduled={self.metrics.unscheduled}",
        ]
        out = "\n".join(lines)
        if self.render_mode == "human":
            print(out)
        return out

    # -- simulation internals ------------------------------------------------
    def _max_steps(self) -> int:
        return max(10, self.cfg.max_steps_multiplier * self.episode.n_instances)

    def _fill_arrivals(self, upto_tick: int) -> None:
        while (
            self._arr_ptr < len(self._arrival_ticks)
            and self._arrival_ticks[self._arr_ptr] <= upto_tick
        ):
            t = self._arrival_ticks[self._arr_ptr]
            for iid in self.episode.arrivals_by_tick[t]:
                self.pending.append(iid)
            self._arr_ptr += 1

    def _release_until(self, upto_tick: int) -> None:
        while self.heap and self.heap[0][0] <= upto_tick:
            finish_tick, iid = heapq.heappop(self.heap)
            if iid not in self.running:
                continue
            inst = self.running.pop(iid)
            self.cluster.release(inst.node, inst.cpu, inst.mem, iid, failed=inst.failed)
            if inst.failed:
                self.metrics.failed += 1
        delta = max(0, upto_tick - self._last_ticks_processed)
        if delta > 0:
            self.cluster.recent_fail *= self.cluster.failure_decay ** min(delta, 100)
            self._last_ticks_processed = upto_tick

    def _advance_to(self, tick: int) -> None:
        tick = max(tick, self.current_tick)
        self._release_until(tick)
        self.current_tick = tick
        self._fill_arrivals(tick)

    def _next_event_tick(self) -> int:
        candidates = []
        if self._arr_ptr < len(self._arrival_ticks):
            candidates.append(self._arrival_ticks[self._arr_ptr])
        if self.heap:
            candidates.append(self.heap[0][0])
        candidates = [c for c in candidates if c > self.current_tick]
        if not candidates:
            return self.current_tick + 1
        return min(candidates)

    def _advance_to_next_event(self) -> None:
        self._advance_to(self._next_event_tick())

    def _auto_advance_if_needed(self) -> None:
        """Advance time until there is a pending instance or the episode ends."""
        while not self.pending and not self._done:
            if self._arr_ptr < len(self._arrival_ticks):
                self._advance_to(self._arrival_ticks[self._arr_ptr])
                continue
            # no future arrivals: drain running instances and finish
            if self.running:
                last_finish = max(self.heap)[0] if self.heap else self.current_tick
                self._advance_to(max(last_finish, self.current_tick))
            self._done = True

    def _drop_stale(self) -> float:
        reward = 0.0
        while self.pending:
            inst = self.episode.instances[self.pending[0]]
            if self.current_tick - inst.submit_tick <= self.cfg.max_wait_ticks:
                break
            self.pending.popleft()
            self.metrics.unscheduled += 1
            self.metrics.dropped_stale += 1
            reward += self.reward_shaper.drop(self.current_tick - inst.submit_tick).total
        return reward

    def _least_loaded(self, mask: np.ndarray) -> int:
        score = self.cluster.cpu_util() + self.cluster.mem_util()
        score = np.where(mask, score, np.inf)
        return int(np.argmin(score))

    def _place(self, inst: Instance, node: int, forced: bool = False):
        colocation = TrafficModel.service_colocation(
            self.cluster, node, inst.service, self.service_of
        )
        self.cluster.reserve(node, inst.cpu, inst.mem, inst.idx)
        inst.node = node
        inst.start_tick = self.current_tick
        base_p, p_fail = self.failure_model.risk_components(inst, self.cluster, node)
        inst.p_fail = p_fail
        inst.p_base = base_p
        failed = bool(self._rng.random() < p_fail)
        inst.failed = failed
        if failed:
            finish = self.current_tick + max(1, inst.duration_ticks // 2)
        else:
            finish = self.current_tick + inst.duration_ticks
        inst.finish_tick = finish
        heapq.heappush(self.heap, (finish, inst.idx))
        self.running[inst.idx] = inst
        self.pending.popleft()

        wait = self.current_tick - inst.submit_tick
        self.metrics.scheduled += 1
        self.metrics.wait_ticks.append(wait)
        self.metrics.colocation += colocation
        imbalance = imbalance_cpu_mem(self.cluster)
        frag = fragmentation(self.cluster, self.ref_cpu, self.ref_mem)
        self.metrics.imbalance_sum += imbalance
        self.metrics.imbalance_n += 1
        self.metrics.cpu_util_sum += float(self.cluster.cpu_util().mean())
        self.metrics.mem_util_sum += float(self.cluster.mem_util().mean())
        self.metrics.util_n += 1
        if (
            self.traffic_cfg.get("enabled", True)
            and self.traffic_sample_every > 0
            and self.metrics.scheduled % self.traffic_sample_every == 0
        ):
            report = self.traffic_model.evaluate(self.cluster, self.service_of)
            self.metrics.traffic_imbalance_sum += report.imbalance
            self.metrics.latency_sum += report.mean_latency
            self.metrics.p95_latency_last = report.p95_latency
            self.metrics.traffic_n += 1
        controllable_risk = max(0.0, p_fail - base_p)
        return self.reward_shaper.placement(
            imbalance, controllable_risk, colocation, wait, frag
        )

    def _obs(self) -> np.ndarray:
        inst = self.episode.instances[self.pending[0]] if self.pending else None
        return self.obs_builder.build(
            self.cluster,
            inst,
            len(self.pending),
            self.current_tick,
            self.service_of,
            self.service_index,
        )

    def fail_instances(self, iids: list[int]) -> int:
        """Force-fail running containers (live-demo injection). Returns count."""
        n = 0
        for iid in iids:
            inst = self.running.get(int(iid))
            if inst is None or inst.failed:
                continue
            inst.failed = True
            inst.finish_tick = self.current_tick + 1
            heapq.heappush(self.heap, (inst.finish_tick, inst.idx))
            n += 1
        return n

    def _info(self) -> dict:
        return {
            "tick": self.current_tick,
            "steps": self._steps,
            "pending": len(self.pending),
            "running": len(self.running),
            "cpu_util_mean": float(self.cluster.cpu_util().mean()),
            "mem_util_mean": float(self.cluster.mem_util().mean()),
            "scheduled": self.metrics.scheduled,
            "failed": self.metrics.failed,
            "unscheduled": self.metrics.unscheduled,
            "done": self._done,
        }

    # -- evaluation ----------------------------------------------------------
    def episode_metrics(self) -> dict:
        return self.metrics.summary()
