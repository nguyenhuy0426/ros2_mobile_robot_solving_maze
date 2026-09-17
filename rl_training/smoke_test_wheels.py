#!/usr/bin/env python3
"""
smoke_test_wheels.py — End-to-end sanity check of the wheel-control stack.

Requires Gazebo running with worlds/nhom8_maze.sdf and spawn_robot_wheel.sh.
Checks:
  1. env reset: teleport + fresh LiDAR observation (shape, range, walls seen)
  2. all-forward wheel command actually translates the robot (+x) — proves
     the JointControllers drive the chassis (VelocityControl stripped)
  3. env.step() reward/termination plumbing and geodesic progress sign
"""

import sys

import numpy as np

from rl_training import config as C
from rl_training.wheel_env import make_wheel_env


def main() -> None:
    env = make_wheel_env(robot_id=1)
    failures = []
    try:
        obs, info = env.reset()
        print(f"[1] reset OK — obs shape={obs.shape}, "
              f"min={obs.min():.3f}, max={obs.max():.3f}, "
              f"geo_dist={info['geo_dist']:.2f} m")
        if obs.shape != (C.WHEEL_STATE_DIM,):
            failures.append(f"obs shape {obs.shape} != {(C.WHEEL_STATE_DIM,)}")
        if not (0.0 <= obs.min() and obs.max() <= 1.0):
            failures.append("obs not normalized to [0,1]")
        if obs.min() > 0.9:
            failures.append("lidar sees no walls — robot outside maze?")
        if not (1.0 < info["geo_dist"] < 15.0):
            failures.append(f"implausible start geo_dist {info['geo_dist']}")

        start_xy = env._snapshot()[1]
        forward = np.array([0.5, 0.5, 0.5, 0.5], dtype=np.float32)
        total_reward, ended = 0.0, None
        for i in range(20):  # 2 s of forward driving
            obs, r, term, trunc, sinfo = env.step(forward)
            total_reward += r
            if term or trunc:
                ended = sinfo
                break
        end_xy = env._snapshot()[1]
        dx = end_xy[0] - start_xy[0]
        dy = end_xy[1] - start_xy[1]
        print(f"[2] 20 forward steps: dx={dx:+.3f} m, dy={dy:+.3f} m, "
              f"reward_sum={total_reward:+.2f}, ended={ended}")
        if abs(dx) < 0.05 and ended is None:
            failures.append(
                "robot did not translate — wheel commands not reaching "
                "physics (check bridge / VelocityControl removal)")
        if dx > 0.05 and total_reward <= 20 * C.WR_TIME:
            failures.append("moved toward goal but no positive progress reward")

        obs2, info2 = env.reset()
        back_xy = env._snapshot()[1]
        print(f"[3] second reset OK — back at "
              f"({back_xy[0]:.2f}, {back_xy[1]:.2f}), "
              f"geo_dist={info2['geo_dist']:.2f} m")
        if abs(back_xy[0] - C.START_XY[0]) > 0.15 or \
           abs(back_xy[1] - C.START_XY[1]) > 0.15:
            failures.append(f"reset teleport failed: robot at {back_xy}")
    finally:
        env.close()

    if failures:
        print("\nSMOKE TEST FAILED:")
        for f in failures:
            print(f"  ✗ {f}")
        sys.exit(1)
    print("\nSMOKE TEST PASSED ✓")


if __name__ == "__main__":
    main()
