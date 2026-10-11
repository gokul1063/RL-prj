"""One demo session = RL-driven env + static-baseline env on identical real workload."""
from __future__ import annotations

import asyncio
import os

import numpy as np

from agent import QAgent
from env import ACTIONS, RealClusterEnv

WINDOW = 10            # ticks between agent decisions
BASELINE_ACTION = 4    # static policy: Least-requested
BASE_INTERVAL = 0.45   # seconds per tick at 1x

W_FAIL, W_IMB, W_MOVE = 8.0, 2.5, 0.5


def make_env(seed: int):
    """Real-data environment (Google Borg 2011 via cluster_rl.env.PlacementEnv)."""
    return RealClusterEnv(seed=seed)


class Session:
    def __init__(self, seed: int = 7):
        self.playing = True
        self.speed = 1.0
        self.run_id = 0
        self.dirty = True
        self.seed = seed
        self.reset(seed)

    def reset(self, seed: int | None = None):
        if seed is not None:
            self.seed = seed
        self.run_id += 1
        self.env = make_env(self.seed)
        self.base = make_env(self.seed)
        self.agent = QAgent(self.env.n_states, len(self.env.actions), seed=self.seed)
        self.tick_in_window = 0
        self.window_fr: list[float] = []
        self.window_im: list[float] = []
        self.cur_s, self.cur_a, self.explored = 0, 1, False
        self.last_reward = 0.0
        self.cum_reward = 0.0
        self.decisions = 0
        self.rewards: list[float] = []
        self.step()
        self.dirty = True

    def handle(self, msg: dict):
        cmd = msg.get("cmd")
        if cmd == "play":
            self.playing = True
        elif cmd == "pause":
            self.playing = False
        elif cmd == "speed":
            self.speed = float(min(max(msg.get("value", 1), 0.25), 20))
        elif cmd == "reset":
            self.reset(msg.get("seed"))
        elif cmd == "inject":
            ids = self.env.inject_failure(msg.get("cluster"))
            self.base.fail_containers(ids)   # same containers hit in the baseline
        self.dirty = True

    def step(self):
        if self.tick_in_window == 0:
            s = self.env.observe()
            a, explored = self.agent.act(s)
            self.cur_s, self.cur_a, self.explored = s, a, explored
            self.env.apply_action(a)
            self.base.apply_action(BASELINE_ACTION)
            self.window_fr, self.window_im = [], []
        rl = self.env.tick()
        self.base.tick()
        self.window_fr.append(rl["failure_rate"])
        self.window_im.append(rl.get("imbalance", 0.0))
        self.tick_in_window += 1
        if self.tick_in_window >= WINDOW:
            r = -W_FAIL * float(np.mean(self.window_fr)) \
                - W_IMB * float(np.mean(self.window_im)) \
                - W_MOVE * self.env.moved_frac
            self.agent.update(self.cur_s, self.cur_a, r, self.env.observe())
            self.last_reward = r
            self.cum_reward += r
            self.decisions += 1
            self.rewards.append(round(r, 3))
            self.tick_in_window = 0

    def build(self) -> dict:
        ag = self.agent
        s = self.env.observe()
        return {
            "type": "frame", "run": self.run_id, "t": self.env.t,
            "playing": self.playing, "speed": self.speed,
            "rl": self.env.frame(), "baseline": self.base.frame(),
            "baseline_name": ACTIONS[BASELINE_ACTION]["name"] + " (static)",
            "agent": {
                "action_id": self.cur_a, "action_name": ACTIONS[self.cur_a]["name"],
                "explored": bool(self.explored), "epsilon": float(ag.eps),
                "state": s, "state_label": self.env.state_labels[s],
                "reward": float(self.last_reward), "cum_reward": float(self.cum_reward),
                "decisions": self.decisions, "window": WINDOW,
                "next_in": WINDOW - self.tick_in_window,
                "q": ag.q.round(3).tolist(), "visits": ag.visits.tolist(),
                "actions": [a["short"] for a in ACTIONS],
                "action_names": [a["name"] for a in ACTIONS],
                "state_labels": self.env.state_labels,
                "rewards": self.rewards[-80:],
            },
        }

    async def run(self, ws):
        try:
            while True:
                if self.playing:
                    self.step()
                    await ws.send_json(self.build())
                    self.dirty = False
                    await asyncio.sleep(BASE_INTERVAL / self.speed)
                elif self.dirty:
                    await ws.send_json(self.build())
                    self.dirty = False
                else:
                    await asyncio.sleep(0.1)
                if self.playing and self.dirty:
                    self.dirty = False
        except asyncio.CancelledError:
            raise
        except Exception:
            return