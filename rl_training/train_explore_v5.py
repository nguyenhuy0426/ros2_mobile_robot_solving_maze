#!/usr/bin/env python3
"""
train_explore_v5.py — Pure-RL training for the multi-maze explore-then-exit
task (v5).

Trains SAC on the 13 decoded mazes inside the combined Gazebo world
``nhom8_maze_multi`` (config.MAZE_MULTI_WORLD_NAME): all mazes are hosted
side by side in one world and the maze is switched PER EPISODE by
teleporting the robot to the selected maze's start pose (gz cannot hot-swap
worlds). Each episode the robot must roam the whole current maze (visit
every coverage zone, scan the walls into a 2D occupancy map) without
touching any wall, and only then take the maze's single exit opening. No
classical planner is involved: the policy is a plain MLP (SAC/TD3) acting
on stacked LiDAR + odometry + coverage state; the geodesic exit field
(maze_registry) is used ONLY inside the reward as potential-based shaping
during the EXIT phase.

Prerequisites (3 terminals):
  1. gz sim -r worlds/nhom8_maze_multi.sdf
  2. source /opt/ros/jazzy/setup.bash && ./spawn_robot_maze.sh $(./rl_venv/bin/python -m rl_training.maze_registry --spawn sigma_1 --multi) 1
  3. this script

Examples:
  source /opt/ros/jazzy/setup.bash
  ./rl_venv/bin/python -m rl_training.train_explore_v5 --steps 400000
  ./rl_venv/bin/python -m rl_training.train_explore_v5 --mazes sigma_1,delta_1
  ./rl_venv/bin/python -m rl_training.train_explore_v5 --load sac_explore_v5_checkpoints/sac_explore_v5_final
  ./rl_venv/bin/python -m rl_training.train_explore_v5 --eval --episodes 10
"""

import argparse
import csv
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

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
import rl_training.maze_registry as MR

_ALGOS = {"sac": SAC, "td3": TD3}


class ExploreMetricsCallback(BaseCallback):
    """Log per-maze coverage / map / return metrics per episode (SB3 + CSV)."""

    CSV_COLUMNS = ["episode", "timesteps", "maze", "coverage_cells",
                   "coverage_frac", "map_pct", "episode_return", "success",
                   "collision", "early_exit", "timeout", "stall",
                   "out_of_bounds"]

    def __init__(self, csv_path: Path) -> None:
        super().__init__()
        self._csv_path = csv_path
        self.episodes = 0
        self.successes = 0
        self.collisions = 0
        self.early_exits = 0
        self.timeouts = 0
        self.stalls = 0
        self._episode_return = 0.0
        self._maze_stats: Dict[str, Dict[str, int]] = {}
        # Schema change guard: rotate a CSV whose header predates the v5
        # row layout so appended rows never misalign with old ones.
        if csv_path.exists():
            header = csv_path.read_text().splitlines()[:1]
            if header and "maze" not in header[0].split(","):
                csv_path.rename(csv_path.with_suffix(".csv.pre_v5"))

    def _on_step(self) -> bool:
        for reward, done, info in zip(self.locals["rewards"],
                                      self.locals["dones"],
                                      self.locals["infos"]):
            self._episode_return += float(reward)
            if not done:
                continue
            maze = info["maze"]
            cov_cells = info.get("coverage_cells", 0)
            cov_frac = info.get("coverage_frac", 0.0)
            stats = self._maze_stats.setdefault(
                maze, {"episodes": 0, "successes": 0, "collisions": 0})
            self.episodes += 1
            stats["episodes"] += 1
            success = int(info.get("success", False))
            collision = int(bool(info.get("collision", False)))
            self.successes += success
            self.collisions += collision
            stats["successes"] += success
            stats["collisions"] += collision
            self.early_exits += int(info.get("early_exit", False))
            self.timeouts += int(info.get("timeout", False))
            self.stalls += int(info.get("stall", False))
            self.logger.record("episode/coverage_cells", cov_cells)
            self.logger.record("episode/coverage_frac", cov_frac)
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
            self.logger.record("episode/episode_return",
                               self._episode_return)
            self.logger.record(f"maze/{maze}/success_rate",
                               stats["successes"] / stats["episodes"])
            self.logger.record(f"maze/{maze}/episodes", stats["episodes"])
            row = {
                "episode": self.episodes,
                "timesteps": self.num_timesteps,
                "maze": maze,
                "coverage_cells": cov_cells,
                "coverage_frac": round(cov_frac, 4),
                "map_pct": round(info.get("map_pct", 0.0), 4),
                "episode_return": round(self._episode_return, 4),
                "success": success,
                "collision": collision,
                "early_exit": int(info.get("early_exit", False)),
                "timeout": int(info.get("timeout", False)),
                "stall": int(info.get("stall", False)),
                "out_of_bounds": int(info.get("out_of_bounds", False)),
            }
            new_file = not self._csv_path.exists()
            with self._csv_path.open("a", newline="") as f:
                w = csv.DictWriter(f, fieldnames=self.CSV_COLUMNS)
                if new_file:
                    w.writeheader()
                w.writerow(row)
            print(f"[ep {self.episodes:4d}][{maze}] cov {cov_frac:4.2f} "
                  f"map {row['map_pct'] * 100:5.1f}% "
                  f"R={row['episode_return']:8.2f} "
                  f"{'SUCCESS' if row['success'] else ''}"
                  f"{' COLLISION' if row['collision'] else ''}"
                  f"{' EARLY-EXIT' if row['early_exit'] else ''}"
                  f"{' OUT-OF-BOUNDS' if row['out_of_bounds'] else ''}"
                  f"{' STALL' if row['stall'] else ''}"
                  f"{' TIMEOUT' if row['timeout'] else ''}",
                  flush=True)
            self._episode_return = 0.0
            if self.episodes % 26 == 0:
                print(f"-- maze success table after {self.episodes} episodes",
                      flush=True)
                for name in sorted(self._maze_stats):
                    s = self._maze_stats[name]
                    rate = 100.0 * s["successes"] / s["episodes"]
                    print(f"{name} eps={s['episodes']} "
                          f"succ={s['successes']} coll={s['collisions']} "
                          f"rate={rate:.0f}%", flush=True)
        return True


def evaluate(model, env, episodes: int) -> None:
    """Greedy rollout (no exploration noise) printing per-episode stats."""
    maps_dir = Path("explore_maps_eval_v5")
    maps_dir.mkdir(exist_ok=True)
    for ep in range(episodes):
        obs, info = env.reset()
        maze = info["maze"]
        done = truncated = False
        total = 0.0
        while not (done or truncated):
            action, _ = model.predict(obs, deterministic=True)
            obs, r, done, truncated, info = env.step(action)
            total += r
        tag = ("SUCCESS" if info.get("success")
               else "COLLISION" if info.get("collision")
               else "EARLY-EXIT" if info.get("early_exit")
               else "OUT-OF-BOUNDS" if info.get("out_of_bounds")
               else "TIMEOUT")
        print(f"[eval ep {ep + 1:2d}][{maze}] "
              f"cov {info.get('coverage_frac', 0.0):4.2f} "
              f"map {info.get('map_pct', 0) * 100:5.1f}%  "
              f"R={total:8.2f}  {tag}")
        env._mapper.save_png(maps_dir / f"eval_ep_{ep + 1:02d}_{info['maze']}.png")


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Pure-RL (SAC) multi-maze explore-then-exit training "
                    "in Gazebo")
    ap.add_argument("--algo", choices=sorted(_ALGOS), default="sac")
    ap.add_argument("--mazes", type=str, default=None,
                    help="comma-separated maze subset, e.g. "
                         "'sigma_1,delta_1' (default: all registry mazes)")
    ap.add_argument("--selection", choices=("round_robin", "random"),
                    default=None,
                    help="per-episode maze selection "
                         "(default: config.EXPL_V5_SELECTION)")
    ap.add_argument("--steps", type=int, default=C.SAC_V4_TOTAL_STEPS)
    ap.add_argument("--robot", type=int, default=1)
    ap.add_argument("--load", type=str, default=None,
                    help="path to a .zip model to resume from")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--ckpt-every", type=int, default=C.SACW_CKPT_EVERY,
                    help="env steps between checkpoints")
    ap.add_argument("--eval", action="store_true",
                    help="greedy evaluation instead of training")
    ap.add_argument("--episodes", type=int, default=10,
                    help="episodes for --eval mode")
    args = ap.parse_args()

    maze_names: Optional[List[str]] = None
    if args.mazes:
        maze_names = [n.strip() for n in args.mazes.split(",") if n.strip()]
        valid = MR.sorted_registry_names()
        unknown = [n for n in maze_names if n not in valid]
        if unknown:
            sys.exit(f"unknown maze(s) {unknown} — valid mazes: "
                     f"{', '.join(valid)}")

    algo_cls = _ALGOS[args.algo]
    prefix = "sac_explore_v5"
    ckpt_dir, log_dir, tb_dir = (C.EXPL_V5_CKPT_DIR, C.EXPL_V5_LOG_DIR,
                                 C.EXPL_V5_TB_DIR)
    final_path = ckpt_dir / f"{prefix}_final"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    env = Monitor(
        make_explore_env(robot_id=args.robot, seed=args.seed,
                         maze_names=maze_names, selection=args.selection,
                         prebuild=True),
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
        CheckpointCallback(save_freq=args.ckpt_every,
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
