#!/usr/bin/env python3
"""
train_explore.py — Pure-RL training for the explore-then-exit task.

The robot must roam the whole 0.75 m 5×5 maze (visit all 25 cells, scan the
walls into a 2D occupancy map) without touching any wall, and only then take
the single exit gap in the east border. No classical planner is involved:
the policy is a plain MLP (SAC) acting on stacked LiDAR + odometry +
coverage state; the geodesic exit field (maze_field.py) is used ONLY inside
the reward as potential-based shaping during the EXIT phase.

Prerequisites (3 terminals):
  1. gz sim -r worlds/nhom8_maze75.sdf
  2. source /opt/ros/jazzy/setup.bash && ./spawn_robot_explore.sh 1
  3. this script

Examples:
  source /opt/ros/jazzy/setup.bash
  ./rl_venv/bin/python -m rl_training.train_explore --steps 400000
  ./rl_venv/bin/python -m rl_training.train_explore --load sac_explore_checkpoints/sac_explore_final
  ./rl_venv/bin/python -m rl_training.train_explore --eval --episodes 10
"""

import argparse
import csv
import sys
import time
from pathlib import Path

import numpy as np

try:
    import rclpy  # noqa: F401
except ImportError:
    sys.exit("rclpy not found — run: source /opt/ros/jazzy/setup.bash")

from stable_baselines3 import SAC, TD3
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.noise import NormalActionNoise

from rl_training import config as C
from rl_training.explore_env import make_explore_env

_ALGOS = {"sac": SAC, "td3": TD3}


class ExploreMetricsCallback(BaseCallback):
    """Log coverage / map / phase metrics per episode (TensorBoard + CSV)."""

    def __init__(self, csv_path: Path) -> None:
        super().__init__()
        self._csv_path = csv_path
        self.episodes = 0
        self.successes = 0
        self.collisions = 0
        self.early_exits = 0
        self.timeouts = 0
        self.stalls = 0
        # Schema change guard: rotate a CSV whose header predates the current
        # row layout so appended rows never misalign with old ones.
        if csv_path.exists():
            header = csv_path.read_text().splitlines()[:1]
            if header and "stall" not in header[0]:
                csv_path.rename(csv_path.with_suffix(".csv.pre_stall"))

    def _on_step(self) -> bool:
        for done, info in zip(self.locals["dones"], self.locals["infos"]):
            if not done:
                continue
            self.episodes += 1
            cov = info.get("coverage_cells", 0)
            self.successes += int(info.get("success", False))
            self.collisions += int(bool(info.get("collision", False)))
            self.early_exits += int(info.get("early_exit", False))
            self.timeouts += int(info.get("timeout", False))
            self.stalls += int(info.get("stall", False))
            self.logger.record("episode/coverage_cells", cov)
            self.logger.record("episode/coverage_frac", cov / 25.0)
            self.logger.record("episode/map_pct", info.get("map_pct", 0.0))
            self.logger.record("episode/success_rate",
                               self.successes / self.episodes)
            self.logger.record("episode/collision_rate",
                               self.collisions / self.episodes)
            self.logger.record("episode/early_exit_rate",
                               self.early_exits / self.episodes)
            self.logger.record("episode/timeout_rate",
                               self.timeouts / self.episodes)
            self.logger.record("episode/stall_rate",
                               self.stalls / self.episodes)
            row = {
                "episode": self.episodes,
                "timesteps": self.num_timesteps,
                "coverage_cells": cov,
                "map_pct": round(info.get("map_pct", 0.0), 4),
                "success": int(info.get("success", False)),
                "collision": int(bool(info.get("collision", False))),
                "early_exit": int(info.get("early_exit", False)),
                "timeout": int(info.get("timeout", False)),
                "stall": int(info.get("stall", False)),
            }
            new_file = not self._csv_path.exists()
            with self._csv_path.open("a", newline="") as f:
                w = csv.DictWriter(f, fieldnames=list(row))
                if new_file:
                    w.writeheader()
                w.writerow(row)
            print(f"[ep {self.episodes:4d}] cells {cov:2d}/25 "
                  f"map {row['map_pct'] * 100:5.1f}% "
                  f"{'SUCCESS' if row['success'] else ''}"
                  f"{' COLLISION' if row['collision'] else ''}"
                  f"{' EARLY-EXIT' if row['early_exit'] else ''}"
                  f"{' STALL' if row['stall'] else ''}"
                  f"{' TIMEOUT' if row['timeout'] else ''}",
                  flush=True)
        return True


def evaluate(model, env, episodes: int) -> None:
    """Greedy rollout (no exploration noise) printing per-episode stats."""
    maps_dir = Path("explore_maps_eval")
    maps_dir.mkdir(exist_ok=True)
    for ep in range(episodes):
        obs, _ = env.reset()
        done = truncated = False
        total = 0.0
        while not (done or truncated):
            action, _ = model.predict(obs, deterministic=True)
            obs, r, done, truncated, info = env.step(action)
            total += r
        tag = ("SUCCESS" if info.get("success")
               else "COLLISION" if info.get("collision")
               else "EARLY-EXIT" if info.get("early_exit") else "TIMEOUT")
        print(f"[eval ep {ep + 1:2d}] cells {info.get('coverage_cells', 0):2d}/25 "
              f"map {info.get('map_pct', 0) * 100:5.1f}%  R={total:8.2f}  {tag}")
        env._mapper.save_png(maps_dir / f"eval_ep_{ep + 1:02d}.png")


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Pure-RL (SAC) explore-then-exit training in Gazebo")
    ap.add_argument("--algo", choices=sorted(_ALGOS), default="sac")
    ap.add_argument("--steps", type=int, default=C.SAC_V4_TOTAL_STEPS)
    ap.add_argument("--robot", type=int, default=1)
    ap.add_argument("--load", type=str, default=None,
                    help="path to a .zip model to resume from")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--eval", action="store_true",
                    help="greedy evaluation instead of training")
    ap.add_argument("--episodes", type=int, default=10,
                    help="episodes for --eval mode")
    args = ap.parse_args()

    algo_cls = _ALGOS[args.algo]
    prefix = f"{args.algo}_explore"
    ckpt_dir, log_dir, tb_dir = (C.SAC_V4_CKPT_DIR, C.SAC_V4_LOG_DIR,
                                 C.SAC_V4_TB_DIR)
    final_path = ckpt_dir / f"{prefix}_final"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    env = Monitor(
        make_explore_env(robot_id=args.robot, seed=args.seed),
        filename=str(log_dir / "train_monitor.csv"))

    if args.eval:
        if args.load is None:
            sys.exit("--eval requires --load <model.zip>")
        model = algo_cls.load(args.load, env=env)
        evaluate(model, env, args.episodes)
        env.close()
        return

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
            common_kwargs["action_noise"] = NormalActionNoise(
                np.zeros(env.action_space.shape[0]),
                C.TD3W_ACTION_NOISE * np.ones(env.action_space.shape[0]))
        model = algo_cls("MlpPolicy", env, **common_kwargs)

    callbacks = [
        CheckpointCallback(save_freq=C.SACW_CKPT_EVERY,
                           save_path=str(ckpt_dir),
                           name_prefix=prefix),
        ExploreMetricsCallback(log_dir / "explore_monitor.csv"),
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
