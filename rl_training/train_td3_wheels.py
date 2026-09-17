#!/usr/bin/env python3
"""
train_td3_wheels.py — TD3 baseline on the SAME GazeboWheelEnv.

TD3 uses a *fixed* Gaussian exploration noise that does not auto-collapse the
way SAC's entropy coefficient did in the v1 run (ent_coef 0.40 → 0.004), so
this is a direct test of the diagnosed exploration-collapse failure mode. It
reuses the identical env, reward shaping, curriculum, callbacks, and
LiDAR-only observation — no environment logic is duplicated.

Outputs go to ``td3_wheel_*`` folders.

Prerequisites (3 terminals): same as train_sac_wheels_v2.py.

Example:
  source /opt/ros/jazzy/setup.bash
  ./rl_venv/bin/python -m rl_training.train_td3_wheels --steps 300000
"""

import argparse
import sys
from pathlib import Path

try:
    import rclpy  # noqa: F401
except ImportError:
    sys.exit("rclpy not found — run: source /opt/ros/jazzy/setup.bash")

import numpy as np
from stable_baselines3 import TD3
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.noise import NormalActionNoise

from rl_training import config as C
from rl_training.callbacks import EpisodeMetricsCallback
from rl_training.wheel_env import make_wheel_env


def main() -> None:
    ap = argparse.ArgumentParser(
        description="TD3 baseline (shaping + curriculum) wheel training")
    ap.add_argument("--steps", type=int, default=C.SACW_TOTAL_STEPS)
    ap.add_argument("--robot", type=int, default=1)
    ap.add_argument("--load", type=str, default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-shaping", action="store_true")
    ap.add_argument("--no-curriculum", action="store_true")
    args = ap.parse_args()

    C.TD3W_CKPT_DIR.mkdir(parents=True, exist_ok=True)
    C.TD3W_LOG_DIR.mkdir(parents=True, exist_ok=True)

    env = Monitor(
        make_wheel_env(robot_id=args.robot, shaping=not args.no_shaping,
                       curriculum=not args.no_curriculum, seed=args.seed),
        filename=str(C.TD3W_LOG_DIR / "train_monitor.csv"))

    action_noise = NormalActionNoise(
        mean=np.zeros(C.WHEEL_ACTION_DIM),
        sigma=C.TD3W_ACTION_NOISE * np.ones(C.WHEEL_ACTION_DIM))

    if args.load:
        if not Path(args.load).exists():
            sys.exit(f"model not found: {args.load}")
        model = TD3.load(args.load, env=env,
                         tensorboard_log=str(C.TD3W_TB_DIR))
        print(f"Resumed from {args.load}")
    else:
        model = TD3(
            "MlpPolicy", env,
            learning_rate=C.SACW_LR,
            buffer_size=C.SACW_BUFFER_SIZE,
            batch_size=C.SACW_BATCH_SIZE,
            gamma=C.SACW_GAMMA,
            tau=C.SACW_TAU,
            learning_starts=C.SACW_LEARN_STARTS,
            train_freq=C.SACW_TRAIN_FREQ,
            gradient_steps=1,
            action_noise=action_noise,
            policy_kwargs={"net_arch": C.SACW_NET_ARCH},
            tensorboard_log=str(C.TD3W_TB_DIR),
            seed=args.seed,
            verbose=1,
        )

    callbacks = [
        CheckpointCallback(save_freq=C.SACW_CKPT_EVERY,
                           save_path=str(C.TD3W_CKPT_DIR),
                           name_prefix="td3_wheels"),
        EpisodeMetricsCallback(),
    ]

    final_path = C.TD3W_CKPT_DIR / "td3_wheels_final"
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
