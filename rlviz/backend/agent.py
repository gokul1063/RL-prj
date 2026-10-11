"""Tabular Q-learning agent. Swap for DQN / PPO later; the Session only needs
`act(state) -> (action, explored)` and `update(s, a, r, s2)`."""
from __future__ import annotations

import numpy as np


class QAgent:
    def __init__(self, n_states, n_actions, gamma=0.1, eps_start=1.0, eps_min=0.05,
                 eps_decay=0.975, seed=0):
        self.q = np.zeros((n_states, n_actions))
        self.visits = np.zeros((n_states, n_actions), dtype=int)
        self.gamma = gamma
        self.eps = eps_start
        self.eps_min = eps_min
        self.eps_decay = eps_decay
        self.rng = np.random.default_rng(seed)

    def act(self, s: int):
        if self.rng.random() < self.eps:
            return int(self.rng.integers(self.q.shape[1])), True
        row = self.q[s]
        best = np.flatnonzero(row == row.max())
        return int(self.rng.choice(best)), False

    def update(self, s, a, r, s2):
        self.visits[s, a] += 1
        alpha = max(0.1, 1.0 / (1.0 + 0.3 * self.visits[s, a]))
        target = r + self.gamma * self.q[s2].max()
        self.q[s, a] += alpha * (target - self.q[s, a])
        self.eps = max(self.eps_min, self.eps * self.eps_decay)
