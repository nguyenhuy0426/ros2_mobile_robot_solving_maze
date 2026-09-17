#!/usr/bin/env python3
"""
replay_buffer.py — Experience replay buffer for off-policy RL.

Adapted from DRL-Robot-Navigation-ROS2/src/drl_navigation_ros2/replay_buffer.py
with numpy-optimized batch sampling.

Supports discrete actions (stored as integers).
"""

import random
from collections import deque
import numpy as np


class ReplayBuffer:
    """
    Fixed-size FIFO replay buffer storing (s, a, r, done, s') tuples.

    The buffer uses a deque for O(1) append/pop and samples random
    mini-batches as numpy arrays for efficient training.

    For DQN, actions are stored as integers (action indices).
    """

    def __init__(self, buffer_size: int = 100_000, seed: int = 42):
        self.buffer_size = int(buffer_size)
        self.buffer = deque(maxlen=self.buffer_size)
        self._rng = random.Random(seed)

    def add(self, state, action, reward, done, next_state):
        """Add a single experience to the buffer.

        Args:
            state: observation array
            action: integer action index (for DQN)
            reward: float reward
            done: bool/float terminal flag
            next_state: next observation array
        """
        self.buffer.append((
            np.array(state, dtype=np.float32),
            int(action),
            float(reward),
            float(done),
            np.array(next_state, dtype=np.float32),
        ))

    def sample(self, batch_size: int):
        """
        Sample a random mini-batch.

        Returns:
            states:      (batch_size, state_dim)
            actions:     (batch_size, 1) — integer action indices
            rewards:     (batch_size, 1)
            dones:       (batch_size, 1)
            next_states: (batch_size, state_dim)
        """
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

    @property
    def size(self):
        return len(self.buffer)
