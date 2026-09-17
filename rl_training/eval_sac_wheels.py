#!/usr/bin/env python3
"""
eval_sac_wheels.py — Deterministic evaluation of a trained wheel-SAC policy.

Loads a Stable-Baselines3 SAC model and runs N episodes in Gazebo with the
deterministic (mean) policy. Reports success rate, collision rate, average
episode length and average reward.

Example:
  source /opt/ros/jazzy/setup.bash
  ./rl_venv/bin/python -m rl_training.eval_sac_wheels \\
      --model sac_wheel_checkpoints/sac_wheels_final --episodes 20
"""

import argparse
import sys
from pathlib import Path

try:
    import rclpy
except ImportError:
    sys.exit("rclpy not found — run: source /opt/ros/jazzy/setup.bash")

import numpy as np
from stable_baselines3 import SAC

from rl_training.wheel_env import make_wheel_env


def main() -> None:
    ap = argparse.ArgumentParser(description="Evaluate wheel-SAC policy")
    ap.add_argument("--model", type=str, required=True,
                    help="path to SB3 model (.zip, extension optional)")
    ap.add_argument("--episodes", type=int, default=20)
    ap.add_argument("--robot", type=int, default=1)
    args = ap.parse_args()

    model_path = Path(args.model)
    if not model_path.exists() and not model_path.with_suffix(".zip").exists():
        sys.exit(f"model not found: {args.model}")

    model = SAC.load(str(model_path))
    env = make_wheel_env(robot_id=args.robot)

    successes, collisions, lengths, rewards = 0, 0, [], []
    try:
        for ep in range(1, args.episodes + 1):
            obs, _ = env.reset()
            done = False
            ep_reward, ep_len = 0.0, 0
            info = {}
            while not done:
                action, _ = model.predict(obs, deterministic=True)
                obs, r, terminated, truncated, info = env.step(action)
                ep_reward += r
                ep_len += 1
                done = terminated or truncated

            outcome = ("SUCCESS" if info.get("success")
                       else "COLLISION" if info.get("collision")
                       else "TIMEOUT")
            successes += int(info.get("success", False))
            collisions += int(info.get("collision", False))
            lengths.append(ep_len)
            rewards.append(ep_reward)
            print(f"ep {ep:3d}/{args.episodes}: {outcome:9s} "
                  f"steps={ep_len:4d} reward={ep_reward:8.2f} "
                  f"geo_dist={info.get('geo_dist', float('nan')):.2f}")
    finally:
        env.close()
        if rclpy.ok():
            rclpy.shutdown()

    n = len(lengths)
    if n == 0:
        sys.exit("no episodes completed")
    print("\n══════════ Evaluation summary ══════════")
    print(f"episodes        : {n}")
    print(f"success rate    : {successes / n:.1%}")
    print(f"collision rate  : {collisions / n:.1%}")
    print(f"timeout rate    : {(n - successes - collisions) / n:.1%}")
    print(f"avg episode len : {np.mean(lengths):.1f} steps")
    print(f"avg reward      : {np.mean(rewards):.2f}")


if __name__ == "__main__":
    main()
