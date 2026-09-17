#!/usr/bin/env python3
"""
train_gazebo_rl.py — Pure-RL training for the FIXED 5×5 maze in Gazebo.

No classical solver is involved anywhere in this pipeline: the robot learns
solely from reward feedback via a neural policy (SAC or TD3). Every episode
spawns the robot at the true START (config.START_XY, cell row 2 / col 0) and
the task is to reach the GOAL (config.GOAL_XY, cell row 2 / col 4).

Compared with the v1/v2 baselines (see rl_training/reports/last_train_analysis.md),
this script enables the evidence-driven fixes:

  1. 2-DOF differential-drive action  — matches the physical wheels (no dead
     lateral DOF to explore).
  2. Odometry in the observation      — the maze is fixed, so (x, y, yaw)
     makes the task nearly Markov; no more perceptual-aliasing guessing.
  3. Balanced terminal penalties      — collision (-25) with the SB3 timeout
     bootstrap removes both the "stall is safe" and "suicide is cheap"
     degenerate optima of v1/v2.

The geodesic distance field (maze_field.py, Dijkstra over the known map) is
used ONLY inside the reward as potential-based shaping — the policy itself is
a plain MLP acting on LiDAR + odometry, and the maze is solved by that policy,
not by a planner.

Start-distance curriculum is OPT-IN (--curriculum): by default every episode
spawns at the true START so the spawn position in Gazebo is always the same,
correct cell. Enable --curriculum only if you accept near-goal spawn cells
during the early levels in exchange for faster value bootstrapping.

Prerequisites (3 terminals):
  1. gz sim -r worlds/nhom8_maze.sdf
  2. source /opt/ros/jazzy/setup.bash && ./spawn_robot_wheel.sh 1
  3. this script

Examples:
  source /opt/ros/jazzy/setup.bash
  ./rl_venv/bin/python -m rl_training.train_gazebo_rl --algo sac --steps 200000
  ./rl_venv/bin/python -m rl_training.train_gazebo_rl --algo td3 --steps 200000
  ./rl_venv/bin/python -m rl_training.train_gazebo_rl --load sac_wheel_v3_checkpoints/sac_v3_final
"""

import argparse
import sys
from pathlib import Path

import numpy as np

try:
    import rclpy  # noqa: F401
except ImportError:
    sys.exit("rclpy not found — run: source /opt/ros/jazzy/setup.bash")

from stable_baselines3 import SAC, TD3
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.noise import NormalActionNoise

from rl_training import config as C
from rl_training.callbacks import EpisodeMetricsCallback
from rl_training.wheel_env import make_wheel_env

_ALGOS = {"sac": SAC, "td3": TD3}


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Pure-RL (SAC/TD3) training on the fixed 5x5 maze in Gazebo")
    ap.add_argument("--algo", choices=sorted(_ALGOS), default="sac")
    ap.add_argument("--steps", type=int, default=C.SAC_V3_TOTAL_STEPS)
    ap.add_argument("--robot", type=int, default=1)
    ap.add_argument("--load", type=str, default=None,
                    help="path to a .zip model to resume from")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--curriculum", action="store_true",
                    help="opt-in start-distance curriculum: early episodes "
                         "spawn on cells NEAR THE GOAL and the start moves "
                         "farther as success rises. Default OFF — every "
                         "episode spawns at the true START (2.25, -3.25).")
    args = ap.parse_args()

    algo_cls = _ALGOS[args.algo]
    if args.algo == "sac":
        ckpt_dir, log_dir, tb_dir = (C.SAC_V3_CKPT_DIR, C.SAC_V3_LOG_DIR,
                                     C.SAC_V3_TB_DIR)
        prefix = "sac_v3"
        final_path = ckpt_dir / "sac_v3_final"
    else:
        ckpt_dir, log_dir, tb_dir = (C.TD3_V3_CKPT_DIR, C.TD3_V3_LOG_DIR,
                                     C.TD3_V3_TB_DIR)
        prefix = "td3_v3"
        final_path = ckpt_dir / "td3_v3_final"

    ckpt_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    env = Monitor(
        make_wheel_env(
            robot_id=args.robot,
            shaping=True,
            curriculum=args.curriculum,
            seed=args.seed,
            diff_drive=True,
            use_odom=True,
            collision_penalty=C.WR_COLLISION_V3,
            stuck_penalty=C.WR_STUCK_PENALTY_V3,
            goal_radius=C.WHEEL_GOAL_RADIUS,
        ),
        filename=str(log_dir / "train_monitor.csv"))

    common_kwargs = dict(
        learning_rate=C.SACW_LR,
        buffer_size=C.SACW_BUFFER_SIZE,
        batch_size=C.SACW_BATCH_SIZE,
        gamma=C.SACW_GAMMA,
        tau=C.SACW_TAU,
        learning_starts=C.SACW_LEARN_STARTS,
        train_freq=C.SACW_TRAIN_FREQ,
        gradient_steps=1,
        policy_kwargs={"net_arch": C.SACW_NET_ARCH},
        tensorboard_log=str(tb_dir),
        seed=args.seed,
        verbose=1,
    )

    if args.load:
        if not Path(args.load).exists() and \
                not Path(args.load).with_suffix(".zip").exists():
            sys.exit(f"model not found: {args.load}")
        model = algo_cls.load(str(args.load), env=env,
                              tensorboard_log=str(tb_dir))
        print(f"Resumed from {args.load}")
    else:
        if args.algo == "td3":
            # Fixed exploration noise — unlike SAC's learned entropy it cannot
            # collapse to a deterministic policy before the goal is discovered
            # (v1 failure mode A).
            common_kwargs["action_noise"] = NormalActionNoise(
                np.zeros(env.action_space.shape[0]),
                C.TD3W_ACTION_NOISE * np.ones(env.action_space.shape[0]))
        model = algo_cls("MlpPolicy", env, **common_kwargs)

    callbacks = [
        CheckpointCallback(save_freq=C.SACW_CKPT_EVERY,
                           save_path=str(ckpt_dir),
                           name_prefix=prefix),
        EpisodeMetricsCallback(),
    ]

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
