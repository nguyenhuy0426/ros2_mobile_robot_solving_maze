#!/usr/bin/env python3
"""
train_explore_multi.py — Multi-robot shared-policy training for the
explore-then-exit task in the combined Gazebo world ``nhom8_maze_multi``
(config.MAZE_MULTI_WORLD_NAME).

N robots (robot_1 .. robot_N) are simulated SIMULTANEOUSLY in one Gazebo
instance; robot k is permanently locked to maze k: every reset() teleports
that robot back to ITS OWN maze's start pose — there is no teleporting
between mazes. A ConcurrentVecEnv stacks the N GazeboExploreEnv instances so a
single replay buffer and ONE shared SAC policy learn from all robots in
parallel: every env step from any robot feeds the same gradient updates.

NOTE: SB3's num_timesteps counts steps across ALL robots (one vec-env step
advances all N robots at once), so --steps is a wall-clock budget shared by
the whole fleet, not per robot.

Prerequisites:
  1. gz sim -r worlds/nhom8_maze_multi.sdf
  2. robots robot_1..robot_N must ALREADY exist in the world, each spawned
     into its maze via spawn_robot_maze.sh — this script NEVER spawns or
     kills entities, it only subscribes to the existing robot topics
     (/scan{k}, /wheel_*_{k}, /model/robot_{k}/pose).
  3. this script

Examples:
  source /opt/ros/jazzy/setup.bash
  ./rl_venv/bin/python -m rl_training.train_explore_multi --steps 400000
  ./rl_venv/bin/python -m rl_training.train_explore_multi --mazes sigma_1,delta_1

GA-style generations: every time a robot beats its maze's best episode return,
the shared policy is snapshotted to ``elite_robot_XX_<maze>.zip`` and the hall
of fame is recorded in ``sac_explore_multi_logs/elites.json`` (one elite per
maze plus a global best). Later runs warm-start from an elite with
``--load sac_explore_multi_checkpoints/sac_explore_multi_elite.zip`` (or any
per-maze elite) to reinforce the best learned behavior — GA elitism, where
each training generation inherits the fittest individuals of the previous one.
"""

import argparse
import csv
import functools
import json
import math
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

try:
    import rclpy  # noqa: F401
except ImportError:
    sys.exit("rclpy not found — run: source /opt/ros/jazzy/setup.bash")

from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from stable_baselines3.common.monitor import Monitor

from rl_training import config as C
from rl_training.concurrent_vec_env import ConcurrentVecEnv
from rl_training.explore_env import make_explore_env
import rl_training.maze_registry as MR

# Script-local artifact paths (kept separate from v5 sac_explore_* outputs).
CKPT_DIR = Path("sac_explore_multi_checkpoints")
LOG_DIR = Path("sac_explore_multi_logs")
TB_DIR = Path("sac_explore_multi_tensorboard")


def _make_env(robot_id: int, maze: str, seed: int,
              stall_limit: Optional[int] = None,
              roam_bonus: Optional[float] = None):
    return Monitor(
        make_explore_env(robot_id=robot_id, maze_names=[maze],
                         selection="round_robin", prebuild=True, seed=seed,
                         stall_limit=stall_limit, roam_bonus=roam_bonus))


class MultiRobotMetricsCallback(BaseCallback):
    """Per-robot / per-maze episode metrics for the multi-robot trainer.

    One line of CSV per terminal event plus console tags; every 50 terminal
    events a per-maze summary table is printed and rates are recorded to the
    SB3 logger. Missing info keys degrade to defaults instead of crashing.
    """

    CSV_COLUMNS = ["timesteps", "robot", "maze", "episode_return",
                   "coverage_frac", "success", "collision", "early_exit",
                   "timeout", "stall"]

    def __init__(self, mazes: List[str], csv_path: Path) -> None:
        super().__init__()
        self._mazes = mazes
        self._csv_path = csv_path
        self._ret: Optional[np.ndarray] = None  # lazy: sized on first step
        self._term = 0
        self._maze_stats: Dict[str, Dict[str, float]] = {
            m: {"episodes": 0, "successes": 0, "collisions": 0, "timeouts": 0,
                "stalls": 0, "ret_sum": 0.0}
            for m in mazes}

    def _on_step(self) -> bool:
        if self._ret is None:
            self._ret = np.zeros(len(self.locals["dones"]), dtype=float)
        self._ret += self.locals["rewards"]

        for i, done in enumerate(self.locals["dones"]):
            if not done:
                continue
            info = self.locals["infos"][i]
            robot = i + 1
            maze = info.get("maze", "?")
            ep_ret = float(self._ret[i])
            self._ret[i] = 0.0

            success = int(bool(info.get("success", False)))
            collision = int(bool(info.get("collision", False)))
            early_exit = int(bool(info.get("early_exit", False)))
            timeout = int(bool(info.get("timeout", False)))
            stall = int(bool(info.get("stall", False)))
            cov_frac = info.get("coverage_frac", 0.0)

            stats = self._maze_stats.setdefault(
                maze, {"episodes": 0, "successes": 0, "collisions": 0,
                       "timeouts": 0, "stalls": 0, "ret_sum": 0.0})
            stats["episodes"] += 1
            stats["successes"] += success
            stats["collisions"] += collision
            stats["timeouts"] += timeout
            stats["stalls"] += stall
            stats["ret_sum"] += ep_ret

            new_file = not self._csv_path.exists()
            with self._csv_path.open("a", newline="") as f:
                w = csv.DictWriter(f, fieldnames=self.CSV_COLUMNS)
                if new_file:
                    w.writeheader()
                w.writerow({
                    "timesteps": self.num_timesteps,
                    "robot": robot,
                    "maze": maze,
                    "episode_return": round(ep_ret, 2),
                    "coverage_frac": round(cov_frac, 4),
                    "success": success,
                    "collision": collision,
                    "early_exit": early_exit,
                    "timeout": timeout,
                    "stall": stall,
                })

            tag = (f"{' SUCCESS' if success else ''}"
                   f"{' COLLISION' if collision else ''}"
                   f"{' EARLY-EXIT' if early_exit else ''}"
                   f"{' TIMEOUT' if timeout else ''}"
                   f"{' STALL' if stall else ''}"
                   f"{' OUT-OF-BOUNDS' if info.get('out_of_bounds', False) else ''}")
            print(f"[robot {robot:2d} | {maze}] cov {cov_frac:4.2f} "
                  f"R={ep_ret:8.2f}{tag}", flush=True)

            self._term += 1
            if self._term % 50 == 0:
                self._print_summary()
        return True

    def _print_summary(self) -> None:
        total_eps = sum(s["episodes"] for s in self._maze_stats.values())
        print(f"-- multi-robot maze summary after {self._term} episodes "
              f"({self.num_timesteps} timesteps)", flush=True)
        succ_sum = 0
        coll_sum = 0
        ret_sum = 0.0
        for name in sorted(self._maze_stats):
            s = self._maze_stats[name]
            if s["episodes"] == 0:
                continue
            succ_sum += s["successes"]
            coll_sum += s["collisions"]
            ret_sum += s["ret_sum"]
            mean_ret = s["ret_sum"] / s["episodes"]
            print(f"{name} eps={s['episodes']} succ={s['successes']} "
                  f"coll={s['collisions']} timeout={s['timeouts']} "
                  f"mean_ret={mean_ret:.2f}", flush=True)
            self.logger.record(f"maze/{name}/success_rate",
                               s["successes"] / s["episodes"])
        if total_eps == 0:
            return
        self.logger.record("multi/success_rate", succ_sum / total_eps)
        self.logger.record("multi/collision_rate", coll_sum / total_eps)
        self.logger.record("multi/mean_return", ret_sum / total_eps)


class EliteCheckpointsCallback(BaseCallback):
    """GA-style per-maze elite snapshots ("hall of fame"), one per maze.

    On every terminal episode the callback compares the episode return with
    the maze's current best (merged from ``elites.json`` across runs unless
    ``reset`` was requested). A new best:

      * saves the shared policy to ``elite_robot_XX_<maze>.zip``
      * atomically rewrites ``elites.json`` (tmp file + os.replace) so the
        next training generation inherits the hall of fame
      * when it is also the global best, saves ``sac_explore_multi_elite.zip``

    Save failures are logged and skipped — checkpointing must never crash
    training. ``model.save(path)`` writes ``path + ".zip"``; the JSON stores
    file names including the ``.zip`` suffix.
    """

    def __init__(self, ckpt_dir: Path, json_path: Path,
                 reset: bool = False) -> None:
        super().__init__()
        self._ckpt_dir = ckpt_dir
        self._json_path = json_path
        self._ret: Optional[np.ndarray] = None  # lazy: sized on first step
        self._best: Dict[str, float] = {}       # maze -> best episode return
        self._mazes: Dict[str, Dict[str, Any]] = {}  # elites.json "mazes"
        self._global: Optional[Dict[str, Any]] = None
        if not reset and json_path.exists():
            self._load(json_path)

    def _load(self, json_path: Path) -> None:
        try:
            data = json.loads(json_path.read_text())
        except Exception as exc:
            print(f"[elite] WARNING: cannot read {json_path}: {exc}",
                  flush=True)
            return
        for maze, entry in data.get("mazes", {}).items():
            try:
                self._best[maze] = float(entry["score"])
                self._mazes[maze] = dict(entry)
            except (KeyError, TypeError, ValueError):
                continue
        g = data.get("global") or {}
        try:
            self._global = {"maze": str(g["maze"]),
                            "score": float(g["score"]),
                            "timesteps": int(g.get("timesteps", 0)),
                            "file": "sac_explore_multi_elite.zip"}
        except (KeyError, TypeError, ValueError):
            self._global = None

    def _write_json(self) -> None:
        payload = {
            "meta": {"updated_timesteps": self.num_timesteps,
                     "updated": datetime.now().isoformat()},
            "global": self._global if self._global is not None else {},
            "mazes": self._mazes,
        }
        try:
            tmp = self._json_path.with_suffix(".tmp")
            tmp.write_text(json.dumps(payload, indent=2))
            os.replace(tmp, self._json_path)
        except Exception as exc:
            print(f"[elite] WARNING: could not write {self._json_path}: "
                  f"{exc}", flush=True)

    def _on_step(self) -> bool:
        if self._ret is None:
            self._ret = np.zeros(len(self.locals["dones"]), dtype=float)
        self._ret += self.locals["rewards"]

        for i, done in enumerate(self.locals["dones"]):
            if not done:
                continue
            info = self.locals["infos"][i]
            maze = info.get("maze", "?")
            ep_ret = float(self._ret[i])
            self._ret[i] = 0.0
            prev = self._best.get(maze, -math.inf)
            if ep_ret <= prev:
                continue

            robot = i + 1
            target = self._ckpt_dir / f"elite_robot_{robot:02d}_{maze}"
            fname = target.name + ".zip"
            try:
                self.model.save(target)
            except Exception as exc:
                print(f"[elite] WARNING: {fname} save failed: {exc}",
                      flush=True)
                continue

            self._best[maze] = ep_ret
            self._mazes[maze] = {
                "robot": robot,
                "score": ep_ret,
                "coverage_frac": float(info.get("coverage_frac", 0.0)),
                "timesteps": self.num_timesteps,
                "file": fname,
            }
            print(f"[elite] robot {robot:2d} {maze} new best R={ep_ret:.2f} "
                  f"(prev {prev:.2f}) -> {fname}", flush=True)
            self.logger.record(f"maze/{maze}/elite_score", ep_ret)

            if self._global is None or ep_ret > self._global["score"]:
                self._global = {"maze": maze, "score": ep_ret,
                                "timesteps": self.num_timesteps,
                                "file": "sac_explore_multi_elite.zip"}
                try:
                    self.model.save(self._ckpt_dir
                                    / "sac_explore_multi_elite")
                except Exception as exc:
                    print("[elite] WARNING: global elite save failed: "
                          f"{exc}", flush=True)
                self.logger.record("multi/elite_score", ep_ret)
            self._write_json()
        return True


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Multi-robot shared-policy (SAC) explore-then-exit "
                    "training in Gazebo (robot k locked to maze k)")
    ap.add_argument("--n-robots", type=int, default=13,
                    help="number of robots/mazes when --mazes is omitted")
    ap.add_argument("--mazes", type=str, default=None,
                    help="comma-separated maze subset, e.g. "
                         "'sigma_1,delta_1' (robot count = number of mazes)")
    ap.add_argument("--steps", type=int, default=C.SAC_V4_TOTAL_STEPS,
                    help="total timesteps across ALL robots")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--ckpt-every", type=int, default=C.SACW_CKPT_EVERY,
                    help="env steps between checkpoints")
    ap.add_argument("--load", type=str, default=None,
                    help="warm-start from a saved SAC .zip (e.g. an elite "
                         "snapshot); total_timesteps resume the loaded "
                         "model's counter")
    ap.add_argument("--stall-limit", type=int, default=None,
                    help="stall-limit episodes before hard termination "
                         "(None = config default: "
                         f"{C.EXPL_STALL_LIMIT})")
    ap.add_argument("--roam-bonus", type=float, default=None,
                    help="bold-roaming reward scale r += bonus * speed "
                         "(None = config default: "
                         f"{C.EXPL_ROAM_BONUS})")
    ap.add_argument("--reset-elites", action="store_true",
                    help="ignore/overwrite elites.json instead of merging "
                         "with the existing hall of fame")
    args = ap.parse_args()

    mazes: List[str]
    if args.mazes:
        mazes = [n.strip() for n in args.mazes.split(",") if n.strip()]
        valid = MR.sorted_registry_names()
        unknown = [n for n in mazes if n not in valid]
        if unknown:
            sys.exit(f"unknown maze(s) {unknown} — valid mazes: "
                     f"{', '.join(valid)}")
    else:
        mazes = MR.sorted_registry_names()[:args.n_robots]

    print("=== multi-robot shared-policy assignment ===")
    for k, m in enumerate(mazes, start=1):
        print(f"robot {k} -> maze {m}")

    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    TB_DIR.mkdir(parents=True, exist_ok=True)

    env = ConcurrentVecEnv([
        functools.partial(_make_env, robot_id=k, maze=m, seed=args.seed + k,
                          stall_limit=args.stall_limit,
                          roam_bonus=args.roam_bonus)
        for k, m in enumerate(mazes, start=1)])

    if args.load:
        model = SAC.load(args.load, env=env)
        print(f"Loaded warm-start model from {args.load} "
              f"(num_timesteps={model.num_timesteps})", flush=True)
    else:
        model = SAC("MlpPolicy", env,
                    learning_rate=C.SACW_LR,
                    buffer_size=C.SACW_BUFFER_SIZE,
                    batch_size=C.SACW_BATCH_SIZE,
                    gamma=C.SACW_GAMMA,
                    tau=C.SACW_TAU,
                    learning_starts=C.SACW_LEARN_STARTS,
                    train_freq=C.SACW_TRAIN_FREQ,
                    gradient_steps=1,
                    policy_kwargs={"net_arch": C.SACW_NET_ARCH},
                    tensorboard_log=str(TB_DIR),
                    seed=args.seed,
                    verbose=1)

    callbacks = [
        CheckpointCallback(save_freq=args.ckpt_every,
                           save_path=str(CKPT_DIR),
                           name_prefix="sac_explore_multi"),
        MultiRobotMetricsCallback(mazes, LOG_DIR / "explore_multi.csv"),
        EliteCheckpointsCallback(ckpt_dir=CKPT_DIR,
                                 json_path=LOG_DIR / "elites.json",
                                 reset=args.reset_elites),
    ]

    try:
        total = (model.num_timesteps + args.steps) if args.load else args.steps
        model.learn(total_timesteps=total, callback=callbacks,
                    progress_bar=False)
    except KeyboardInterrupt:
        print("\nInterrupted — saving model before exit.")
    finally:
        final_path = CKPT_DIR / "sac_explore_multi_final"
        model.save(final_path)
        print(f"Model saved to {final_path}.zip")
        env.close()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
