#!/usr/bin/env python3
"""
elite_buffer.py — Elite replay buffer for storing successful (goal-reaching) episodes.

Stores transitions from episodes that reached the goal. During training, a fraction
of each batch is sampled from this buffer so the network always has access to optimal
goal-reaching experience — self-imitation / GA-style elitism.

Supports discrete actions (stored as integers).
"""

import random
from collections import deque
import numpy as np

from rl_training import config as C


class EliteBuffer:
    """
    Separate replay buffer for elite episodes.

    Stores complete episode trajectories from episodes that reached the goal,
    so successful behavior is always available to sample during training.
    """

    def __init__(self, buffer_size: int = C.ELITE_BUFFER_SIZE, seed: int = 123):
        self.buffer_size = int(buffer_size)
        self.buffer = deque(maxlen=self.buffer_size)
        self._rng = random.Random(seed)

    def should_store(self, success: bool) -> bool:
        """An episode qualifies as elite iff it reached the goal."""
        return bool(success)

    def store_episode(self, transitions: list):
        """
        Store all transitions from a successful episode.

        Args:
            transitions: list of (state, action, reward, done, next_state) tuples
                         action is an integer for DQN
        """
        for t in transitions:
            self.buffer.append(t)

    def sample(self, batch_size: int):
        """Sample a random mini-batch from the elite buffer."""
        if len(self.buffer) == 0:
            return None
        n = min(len(self.buffer), batch_size)
        batch = self._rng.sample(list(self.buffer), n)

        states      = np.array([b[0] for b in batch])
        actions     = np.array([b[1] for b in batch], dtype=np.int64).reshape(-1, 1)
        rewards     = np.array([b[2] for b in batch]).reshape(-1, 1)
        dones       = np.array([b[3] for b in batch]).reshape(-1, 1)
        next_states = np.array([b[4] for b in batch])

        return states, actions, rewards, dones, next_states

    def __len__(self):
        return len(self.buffer)
