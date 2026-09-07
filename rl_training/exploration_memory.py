"""Episode visitation memory from normalized odometry, without a maze oracle."""

import gymnasium as gym
import numpy as np

from rl_training import config as C


class VisitMemory(gym.ObservationWrapper):
    """Append a spatial visitation bitmap; keep the legacy observation intact.

    Bins cover the normalized maze bounding box, not registry zone labels.
    This records where the robot has been, without revealing unseen walls.
    It is coarse memory, not a connectivity map or a path planner.
    """

    def __init__(self, env):
        super().__init__(env)
        self._grid = np.zeros((C.EXPL_MEMORY_GRID, C.EXPL_MEMORY_GRID),
                              dtype=np.float32)
        self.observation_space = gym.spaces.Box(
            low=np.concatenate((env.observation_space.low,
                                np.zeros(self._grid.size, dtype=np.float32))),
            high=np.concatenate((env.observation_space.high,
                                 np.ones(self._grid.size, dtype=np.float32))),
            dtype=np.float32)

    def reset(self, **kwargs):
        self._grid.fill(0)
        return super().reset(**kwargs)

    def observation(self, observation):
        # The last six legacy features are x, y, cos(yaw), sin(yaw), cov, phase.
        xy = np.asarray(observation[-6:-4])
        ix, iy = np.clip(((xy + 1.0) * 0.5 * C.EXPL_MEMORY_GRID).astype(int),
                         0, C.EXPL_MEMORY_GRID - 1)
        self._grid[iy, ix] = 1.0
        return np.concatenate((observation, self._grid.ravel()))
