#!/usr/bin/env python3
"""
train_sac_wheels.py — SAC training for LiDAR-only 4-wheel velocity control.

Uses Stable-Baselines3 SAC (off-policy: essential because Gazebo runs in
real time at ~10 env steps/s, so every transition must be reused via the
replay buffer; PPO's on-policy sample cost would be prohibitive here).

Prerequisites (3 terminals):
  1. gz sim -r worlds/nhom8_maze.sdf
  2. source /opt/ros/jazzy/setup.bash && ./spawn_robot_wheel.sh 1
  3. this script (see README/guide for the exact command)

Example:
  source /opt/ros/jazzy/setup.bash
  ./rl_venv/bin/python -m rl_training.train_sac_wheels --steps 300000
"""

import argparse
import sys
from pathlib import Path

try:
    import rclpy  # noqa: F401
except ImportError:
    sys.exit("rclpy not found — run: source /opt/ros/jazzy/setup.bash")

from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from stable_baselines3.common.monitor import Monitor

from rl_training import config as C
from rl_training.wheel_env import make_wheel_env


class EpisodeOutcomeCallback(BaseCallback):
    """Log success/collision/timeout rates to TensorBoard at episode end."""

    def __init__(self):
        super().__init__()
        self.successes = 0
        self.collisions = 0
        self.timeouts = 0
        self.episodes = 0

    def _on_step(self) -> bool:
        for done, info in zip(self.locals["dones"], self.locals["infos"]):
            if not done:
                continue
            self.episodes += 1
            self.successes += int(info.get("success", False))
            self.collisions += int(info.get("collision", False))
            self.timeouts += int(info.get("timeout", False))
            self.logger.record("episode/success_rate",
                               self.successes / self.episodes)
            self.logger.record("episode/collision_rate",
                               self.collisions / self.episodes)
            self.logger.record("episode/timeout_rate",
                               self.timeouts / self.episodes)
            self.logger.record("episode/geo_dist_final",
                               info.get("geo_dist", float("nan")))
        return True


def main() -> None:
    ap = argparse.ArgumentParser(
        description="SAC wheel-velocity training (LiDAR-only policy)")
    ap.add_argument("--steps", type=int, default=C.SACW_TOTAL_STEPS,
                    help="total environment steps")
    ap.add_argument("--robot", type=int, default=1, help="robot id")
    ap.add_argument("--load", type=str, default=None,
                    help="path to a .zip model to resume from")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    C.SACW_CKPT_DIR.mkdir(parents=True, exist_ok=True)
    C.SACW_LOG_DIR.mkdir(parents=True, exist_ok=True)

    env = Monitor(make_wheel_env(robot_id=args.robot),
                  filename=str(C.SACW_LOG_DIR / "train_monitor.csv"))

    if args.load:
        if not Path(args.load).exists():
            sys.exit(f"model not found: {args.load}")
        model = SAC.load(args.load, env=env,
                         tensorboard_log=str(C.SACW_TB_DIR))
        print(f"Resumed from {args.load}")
    else:
        model = SAC(
            "MlpPolicy", env,
            learning_rate=C.SACW_LR,
            buffer_size=C.SACW_BUFFER_SIZE,
            batch_size=C.SACW_BATCH_SIZE,
            gamma=C.SACW_GAMMA,
            tau=C.SACW_TAU,
            learning_starts=C.SACW_LEARN_STARTS,
            train_freq=C.SACW_TRAIN_FREQ,
            gradient_steps=1,
            policy_kwargs={"net_arch": C.SACW_NET_ARCH},
            tensorboard_log=str(C.SACW_TB_DIR),
            seed=args.seed,
            verbose=1,
        )

    callbacks = [
        CheckpointCallback(
            save_freq=C.SACW_CKPT_EVERY,
            save_path=str(C.SACW_CKPT_DIR),
            name_prefix="sac_wheels"),
        EpisodeOutcomeCallback(),
    ]

    final_path = C.SACW_CKPT_DIR / "sac_wheels_final"
    try:
        model.learn(total_timesteps=args.steps, callback=callbacks,
                    reset_num_timesteps=args.load is None,
                    progress_bar=False)
    except KeyboardInterrupt:
        print("\nInterrupted — saving model before exit.")
    finally:
        model.save(final_path)
        print(f"Model saved to {final_path}.zip")
        env.close()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
