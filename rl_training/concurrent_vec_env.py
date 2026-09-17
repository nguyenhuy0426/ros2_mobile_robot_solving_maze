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

from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
import os

import numpy as np

from stable_baselines3.common.vec_env import DummyVecEnv


class ConcurrentVecEnv(DummyVecEnv):
    """DummyVecEnv that publishes all actions up front, then collects results."""

    def reset(self):
        """Reset robot envs in bounded parallel batches.

        ``DummyVecEnv.reset`` resets every Gazebo environment serially.  A
        reset includes a set-pose service call, settle delay and a fresh sensor
        wait, so the 13-robot trainer can appear frozen at startup or after a
        terminal burst even though Gazebo and all bridges are healthy.  Four
        concurrent resets keep service pressure bounded while removing the
        N-times sensor wait from the control path.
        """
        reset_args = [
            (self._seeds[i], self._options[i])
            for i in range(self.num_envs)
        ]
        try:
            requested_workers = int(os.environ.get("RL_RESET_WORKERS", "4"))
            workers = max(1, min(requested_workers, self.num_envs))
            with ThreadPoolExecutor(max_workers=workers) as pool:
                futures = [
                    pool.submit(
                        self.envs[i].reset,
                        seed=seed,
                        **({"options": options} if options else {}),
                    )
                    for i, (seed, options) in enumerate(reset_args)
                ]
                results = [future.result() for future in futures]
        except Exception:
            for env in self.envs:
                try:
                    env.unwrapped.hold()
                except Exception:
                    pass
            raise

        for i, (obs, info) in enumerate(results):
            self.reset_infos[i] = info
            self._save_obs(i, obs)
        self._reset_seeds()
        self._reset_options()
        return self._obs_from_buf()

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
        # One fleet-wide sensor wait instead of N sequential ones: all robots'
        # lidar renders complete inside the same scan window (one render
        # thread), so waiting together collapses the vec step to ~one scan
        # period instead of N staggered waits (measured 1.8-2 s -> ~0.3 s at
        # N=13, 5 Hz lidar). The old for-loop made this only conceptually
        # parallel: one stalled robot could block the other 12 for the full
        # sensor timeout, leaving their wheel commands latched or held and
        # making the fleet look frozen. Submit every barrier concurrently.
        barriers = [getattr(env.unwrapped, "wait_fresh_barrier", None)
                    for env in self.envs]
        barriers = [barrier for barrier in barriers if barrier is not None]
        try:
            if barriers:
                with ThreadPoolExecutor(max_workers=len(barriers)) as pool:
                    futures = [pool.submit(barrier) for barrier in barriers]
                    # Consume in submission order so every barrier has
                    # finished before step() consumes its latched baseline.
                    for future in futures:
                        future.result()
        except Exception:
            # A barrier failure happens before the existing step_wait try /
            # finally block. Stop every robot here as well; otherwise a
            # successful pre_send on a later robot can remain latched while
            # the trainer unwinds or the supervisor restarts it.
            for env in self.envs:
                try:
                    env.unwrapped.hold()
                except Exception:
                    pass
            raise
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
            done_idx = [i for i in range(self.num_envs) if self.buf_dones[i]]
            for i in done_idx:
                self.buf_infos[i]["terminal_observation"] = obs_list[i]
            # Resets are I/O-bound (gz service call + settle sleep), one env's
            # subprocess never touches another's node, so they run concurrently
            # in threads: a 5-robot death burst costs one reset (~1-2 s), not
            # five (10-25 s sequential, measured).
            with ThreadPoolExecutor(max_workers=min(4, len(done_idx))) as pool:
                futs = {i: pool.submit(self.envs[i].reset) for i in done_idx}
                for i, fut in futs.items():
                    obs_list[i], self.reset_infos[i] = fut.result()

        for i in range(self.num_envs):
            self._save_obs(i, obs_list[i])
        return (self._obs_from_buf(), np.copy(self.buf_rews),
                np.copy(self.buf_dones), deepcopy(self.buf_infos))
