"""Vec-env that issues every robot's wheel command before waiting on sensors.

``DummyVecEnv`` steps its envs one after another, and each Gazebo env's
``step()`` blocks ~0.6 s of wall clock waiting for its own next LaserScan.
With N robots sharing one world that serialises into an N x longer gap
between successive commands to any given robot -- and since a wheel-velocity
command stays latched in Gazebo until it is overwritten, each robot then
integrates its command for N control periods instead of one. Measured with
N=13: 29-135 cm of travel per "0.1 s" control step versus 3.6 cm for a single
env, so robots crossed whole corridors per step and every episode ended in a
1-3 step collision.

Splitting the step into send-all-then-receive-all fixes both halves: every
robot's command covers one control period regardless of N, and the sensor
waits overlap instead of stacking (one vec step costs ~one scan period, not N).
"""

from __future__ import annotations

from copy import deepcopy

import numpy as np

from stable_baselines3.common.vec_env import DummyVecEnv


class ConcurrentVecEnv(DummyVecEnv):
    """DummyVecEnv that publishes all actions up front, then collects results."""

    def step_async(self, actions) -> None:
        for env, action in zip(self.envs, actions):
            env.unwrapped.pre_send(action)
        super().step_async(actions)

    def step_wait(self):
        """Collect every env's result first, then reset the finished ones.

        ``reset()`` teleports and settles one env at a time, so a batch of
        resets stretches the vec step to tens of seconds. Robots that are
        still mid-episode would spend all of it driving on their latched
        command, so the fleet is held still across the reset window.
        """
        obs_list = [None] * self.num_envs
        try:
            for i in range(self.num_envs):
                obs, self.buf_rews[i], terminated, truncated, self.buf_infos[i] = (
                    self.envs[i].step(self.actions[i]))
                # Stop THIS robot the moment its own control period closes.
                # Holding the whole fleet only after the last env returned
                # leaves every early finisher driving through its slower
                # peers' sensor waits, so per-step travel still grew with N:
                # measured 1.6 cm at N=1, 2.1 cm at N=4, 4.9 cm at N=8.
                self.envs[i].unwrapped.hold()
                self.buf_dones[i] = terminated or truncated
                self.buf_infos[i]["TimeLimit.truncated"] = truncated and not terminated
                obs_list[i] = obs
        finally:
            # SAC updates, callbacks and resets all run outside the control
            # interval. Their variable latency must not extend wheel commands.
            # Also stop the fleet immediately if any sensor wait fails.
            for env in self.envs:
                env.unwrapped.hold()

        if self.buf_dones.any():
            for i in range(self.num_envs):
                if self.buf_dones[i]:
                    self.buf_infos[i]["terminal_observation"] = obs_list[i]
                    obs_list[i], self.reset_infos[i] = self.envs[i].reset()

        for i in range(self.num_envs):
            self._save_obs(i, obs_list[i])
        return (self._obs_from_buf(), np.copy(self.buf_rews),
                np.copy(self.buf_dones), deepcopy(self.buf_infos))
