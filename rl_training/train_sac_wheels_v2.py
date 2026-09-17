#!/usr/bin/env python3
"""
train_sac_wheels_v2.py — SAC with reward-shaping v2 + start-distance curriculum.

Same env, algorithm, and LiDAR-only observation as the v1 baseline, but with the
evidence-driven fixes from rl_training/reports/last_train_analysis.md enabled:
softened collision, anti-stall + exploration-coverage reward, action smoothing,
and a start-distance curriculum so the agent can actually reach the +200 goal.

Outputs go to the separate ``sac_wheel_v2_*`` folders; the v1 baseline artifacts
are untouched.

Prerequisites (3 terminals):
  1. gz sim -r worlds/nhom8_maze.sdf
  2. source /opt/ros/jazzy/setup.bash && ./spawn_robot_wheel.sh 1
  3. this script

Example:
  source /opt/ros/jazzy/setup.bash
  ./rl_venv/bin/python -m rl_training.train_sac_wheels_v2 --steps 300000
"""

import argparse
import sys
from pathlib import Path

try:
    import rclpy  # noqa: F401
except ImportError:
    sys.exit("rclpy not found — run: source /opt/ros/jazzy/setup.bash")

from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.monitor import Monitor

from rl_training import config as C
from rl_training.callbacks import EpisodeMetricsCallback
from rl_training.wheel_env import make_wheel_env


def main() -> None:
    ap = argparse.ArgumentParser(
        description="SAC v2 (shaping + curriculum) wheel training")
    ap.add_argument("--steps", type=int, default=C.SACW_TOTAL_STEPS)
    ap.add_argument("--robot", type=int, default=1)
    ap.add_argument("--load", type=str, default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-curriculum", action="store_true",
                    help="disable the start-distance curriculum")
    args = ap.parse_args()

    C.SACW_V2_CKPT_DIR.mkdir(parents=True, exist_ok=True)
    C.SACW_V2_LOG_DIR.mkdir(parents=True, exist_ok=True)

    env = Monitor(
        make_wheel_env(robot_id=args.robot, shaping=True,
                       curriculum=not args.no_curriculum, seed=args.seed),
        filename=str(C.SACW_V2_LOG_DIR / "train_monitor.csv"))

    if args.load:
        if not Path(args.load).exists():
            sys.exit(f"model not found: {args.load}")
        model = SAC.load(args.load, env=env,
                         tensorboard_log=str(C.SACW_V2_TB_DIR))
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
            tensorboard_log=str(C.SACW_V2_TB_DIR),
            seed=args.seed,
            verbose=1,
        )

    callbacks = [
        CheckpointCallback(save_freq=C.SACW_CKPT_EVERY,
                           save_path=str(C.SACW_V2_CKPT_DIR),
                           name_prefix="sac_wheels_v2"),
        EpisodeMetricsCallback(),
    ]

    final_path = C.SACW_V2_CKPT_DIR / "sac_wheels_v2_final"
    try:
        model.learn(total_timesteps=args.steps, callback=callbacks,
                    reset_num_timesteps=args.load is None, progress_bar=False)
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
