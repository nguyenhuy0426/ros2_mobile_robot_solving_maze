#!/usr/bin/env python3
"""
train_explore_multi.py — Multi-robot shared-policy training for the
explore-then-exit task in the combined Gazebo world ``nhom8_maze_multi``
(config.MAZE_MULTI_WORLD_NAME).

N robots (robot_1 .. robot_N) are simulated SIMULTANEOUSLY in one Gazebo
instance. The maze pool is dealt round-robin into N DISJOINT groups
(``_partition_mazes``) and robot k cycles through group k: each reset()
teleports that robot to the NEXT maze of its own group. Groups never
overlap, so no two robots can be teleported onto the same start pose, and
the union of the groups is the whole pool, so no maze goes untrained.
A ConcurrentVecEnv stacks the N GazeboExploreEnv instances so a
single replay buffer and ONE shared SAC policy learn from all robots in
parallel: every env step from any robot feeds the same gradient updates.

NOTE: SB3's num_timesteps counts steps across ALL robots (one vec-env step
advances all N robots at once), so --steps is a transition budget shared by
the whole fleet, not per robot or a wall-clock duration.

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

Each run has an isolated artifact directory (printed at startup), including
logs/elites.json and checkpoints/elite_robot_XX_<maze>.zip. Elites are ranked
by coverage then return within the run. Later runs can warm-start weights
with --load <checkpoint.zip>; replay data is not carried by that zip.
Only one instance of this trainer may own the shared robot topics at a time.
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
from rl_training.exploration_memory import VisitMemory
from rl_training.training_session import training_lock
import rl_training.maze_registry as MR


def _partition_mazes(mazes: List[str], n_robots: int) -> List[List[str]]:
    """Deal the maze pool into ``n_robots`` disjoint, near-equal groups.

    Striding (``mazes[k::n]``) instead of cutting contiguous blocks: the
    registry is sorted by family (delta_*, ortho_*, sigma_*), so contiguous
    blocks would give one robot every ortho maze and another every sigma
    maze — a robot that keeps dying early would then starve a whole family
    out of the shared replay buffer.
    """
    if n_robots > len(mazes):
        raise ValueError(f"{n_robots} robots cannot partition "
                         f"{len(mazes)} mazes disjointly")
    return [list(mazes[k::n_robots]) for k in range(n_robots)]


def _make_env(robot_id: int, mazes: List[str], seed: int,
              stall_limit: Optional[int] = None,
              roam_bonus: Optional[float] = None, visit_memory: bool = False):
    env = make_explore_env(robot_id=robot_id, maze_names=list(mazes),
                         selection="round_robin", prebuild=True, seed=seed,
                         stall_limit=stall_limit, roam_bonus=roam_bonus)
    return Monitor(VisitMemory(env) if visit_memory else env)


class MultiRobotMetricsCallback(BaseCallback):
    """Per-robot / per-maze episode metrics for the multi-robot trainer.

    One line of CSV per terminal event plus console tags; every 50 terminal
    events a per-maze summary table is printed and rates are recorded to the
    SB3 logger. Missing info keys degrade to defaults instead of crashing.
    """

    CSV_COLUMNS = ["timesteps", "robot", "maze", "episode_return",
                   "coverage_frac", "success", "collision", "early_exit",
                   "timeout", "stall", "out_of_bounds", "ep_len",
                   "min_slack", "danger_ray", "map_pct", "pose_x", "pose_y",
                   "pose_yaw", "path_length", "action_scale", "safety_override"]

    def __init__(self, mazes: List[str], csv_path: Path) -> None:
        super().__init__()
        self._mazes = mazes
        self._csv_path = csv_path
        self._rotate_stale_csv()
        self._ret: Optional[np.ndarray] = None  # lazy: sized on first step
        self._prev_pos: Optional[List[Optional[tuple]]] = None
        self._path_len: Optional[np.ndarray] = None
        self._term = 0
        self._maze_stats: Dict[str, Dict[str, float]] = {
            m: {"episodes": 0, "successes": 0, "collisions": 0, "timeouts": 0,
                "stalls": 0, "ret_sum": 0.0}
            for m in mazes}

    def _rotate_stale_csv(self) -> None:
        """Archive a pre-existing CSV whose header lacks the current columns.

        Appending wider rows under a narrower header silently misaligns every
        column, so an older log is renamed instead of being written into.
        """
        if not self._csv_path.exists():
            return
        with self._csv_path.open(newline="") as f:
            header = next(csv.reader(f), [])
        if header == self.CSV_COLUMNS:
            return
        stamp = time.strftime("%Y%m%d-%H%M%S")
        archived = self._csv_path.with_name(
            f"{self._csv_path.stem}.{stamp}{self._csv_path.suffix}")
        self._csv_path.rename(archived)
        print(f"[metrics] header cu -> luu {archived.name}", flush=True)

    def _on_step(self) -> bool:
        if self._ret is None:
            self._ret = np.zeros(len(self.locals["dones"]), dtype=float)
            self._prev_pos = [None] * len(self.locals["dones"])
            self._path_len = np.zeros(len(self.locals["dones"]), dtype=float)
        self._ret += self.locals["rewards"]

        for i, info in enumerate(self.locals["infos"]):
            pos = info.get("pos")
            if pos is None:
                continue
            pos = (float(pos[0]), float(pos[1]))
            if self._prev_pos[i] is not None:
                self._path_len[i] += float(np.hypot(
                    pos[0] - self._prev_pos[i][0], pos[1] - self._prev_pos[i][1]))
            self._prev_pos[i] = pos

        for i, done in enumerate(self.locals["dones"]):
            if not done:
                continue
            info = self.locals["infos"][i]
            robot = i + 1
            maze = info.get("maze", "?")
            ep_ret = float(self._ret[i])
            self._ret[i] = 0.0
            pos = info.get("pos") or (float("nan"), float("nan"))
            yaw = float(info.get("yaw", float("nan")))

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
                    "out_of_bounds": int(bool(info.get("out_of_bounds", False))),
                    "ep_len": int(info.get("ep_len", 0)),
                    "min_slack": round(float(info.get("min_slack", 0.0)), 4),
                    "danger_ray": int(info.get("danger_ray", -1)),
                    "map_pct": round(float(info.get("map_pct", 0.0)), 4),
                    "pose_x": round(float(pos[0]), 4),
                    "pose_y": round(float(pos[1]), 4),
                    "pose_yaw": round(yaw, 4),
                    "path_length": round(float(self._path_len[i]), 4),
                    "action_scale": round(float(info.get("action_scale", 1.0)), 4),
                    "safety_override": int(bool(info.get("safety_override", False))),
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
            self._path_len[i] = 0.0
            self._prev_pos[i] = None
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
        # v6: elites are ranked lexicographically by (coverage_frac, ep_ret),
        # not by ep_ret alone. Under v4 the top-3 episodes were 1500-step
        # timeouts at coverage 0.08 -- pure return breeds roam-bonus/timeout
        # farmers, which is the opposite of what an explorer elite should be.
        self._best: Dict[str, tuple] = {}       # maze -> (coverage, return)
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
                self._best[maze] = (float(entry.get("coverage_frac", 0.0)),
                                    float(entry["score"]))
                self._mazes[maze] = dict(entry)
            except (KeyError, TypeError, ValueError):
                continue
        g = data.get("global") or {}
        try:
            self._global = {"maze": str(g["maze"]),
                            "score": float(g["score"]),
                            "coverage_frac": float(g.get("coverage_frac",
                                                         0.0)),
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
            cov = float(info.get("coverage_frac", 0.0))
            key = (cov, ep_ret)
            prev = self._best.get(maze, (-math.inf, -math.inf))
            if key <= prev:
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

            self._best[maze] = key
            self._mazes[maze] = {
                "robot": robot,
                "score": ep_ret,
                "coverage_frac": cov,
                "timesteps": self.num_timesteps,
                "file": fname,
            }
            print(f"[elite] robot {robot:2d} {maze} new best "
                  f"cov={cov:.2f} R={ep_ret:.2f} "
                  f"(prev cov={prev[0]:.2f} R={prev[1]:.2f}) -> {fname}",
                  flush=True)
            self.logger.record(f"maze/{maze}/elite_score", ep_ret)

            g_key = (float(self._global.get("coverage_frac", 0.0)),
                     float(self._global["score"])) \
                if self._global is not None else (-math.inf, -math.inf)
            if key > g_key:
                self._global = {"maze": maze, "score": ep_ret,
                                "coverage_frac": cov,
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
        "training in Gazebo (the maze pool is split into n disjoint "
        "round-robin groups, one per robot)")
    # 4, not 13: all gpu_lidar sensors render in ONE thread, so a 13-robot
    # fleet holds /scanN at 6.5 Hz against a configured 10 Hz. The control
    # loop waits on a fresh scan, so that alone stretches the control period
    # and lets each latched wheel command integrate for longer than EXPL_DT.
    # Measured end-to-end: N=4 -> 16.0 env-steps/s at 2.1 cm travel/step,
    # N=8 -> 9.8 at 4.9 cm (33 cm worst case, half a maze cell), N=13 ->
    # _wait_fresh_step trips its 5 s timeout and the run dies. N=4 is both
    # ~2.8x faster and ~2.3x higher fidelity than N=13.
    ap.add_argument("--n-robots", type=int, default=4,
                    help="fleet size; the maze pool is split into this many "
                         "disjoint round-robin groups")
    ap.add_argument("--bootstrap-scripted", type=int, default=0,
                    metavar="N",
                    help="before training, roll the geodesic demonstrator for "
                         "N vec-env steps and store the transitions in the "
                         "replay buffer (SACfD warm start). 0 = pure RL from "
                         "a random policy (default)")
    ap.add_argument("--mazes", type=str, default=None,
                    help="comma-separated maze pool to split between the "
                         "robots, e.g. 'sigma_1,delta_1' (default: all "
                         "registry mazes)")
    ap.add_argument("--steps", type=int, default=C.SAC_V4_TOTAL_STEPS,
                    help="total timesteps across ALL robots")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--ckpt-every", type=int, default=C.SACW_CKPT_EVERY,
                    help="env steps between checkpoints")
    ap.add_argument("--load", type=str, default=None,
                    help="warm-start from a saved SAC .zip (e.g. an elite "
                         "snapshot); total_timesteps resume the loaded "
                         "model's counter")
    ap.add_argument("--load-buffer", type=str, default=None,
                    help="replay_buffer.pkl saved by a previous attempt; "
                         "restores its transitions so the resumed run "
                         "updates from step one instead of re-running a "
                         "random-action warmup")
    ap.add_argument("--stall-limit", type=int, default=None,
                    help="consecutive stalled steps before hard termination "
                         "(None = config default: "
                         f"{C.EXPL_STALL_LIMIT})")
    ap.add_argument("--roam-bonus", type=float, default=None,
                    help="bold-roaming reward scale r += bonus * speed "
                         "(None = config default: "
                         f"{C.EXPL_ROAM_BONUS})")
    ap.add_argument("--reset-elites", action="store_true",
                    help="start a fresh elite table (each run already has "
                         "an isolated hall of fame)")
    ap.add_argument("--visit-memory", action="store_true",
                    help="append 25 odometry visitation bits; requires a "
                         "new policy or a checkpoint with the same input")
    ap.add_argument("--gradient-steps", type=int, default=1,
                    help="updates per vector step; -1 matches collected "
                         "transitions (13 updates for 13 robots)")
    ap.add_argument("--run-dir", type=Path, default=None,
                    help="new directory for this run's logs/checkpoints; "
                         "default: sac_explore_multi_runs/<timestamp-pid>")
    args = ap.parse_args()
    if args.steps <= 0 or args.ckpt_every <= 0 or args.n_robots <= 0:
        ap.error("steps, ckpt-every and n-robots must be positive")
    if args.gradient_steps != -1 and args.gradient_steps <= 0:
        ap.error("gradient-steps must be -1 or positive")

    mazes: List[str]
    if args.mazes:
        mazes = [n.strip() for n in args.mazes.split(",") if n.strip()]
        valid = MR.sorted_registry_names()
        unknown = [n for n in mazes if n not in valid]
        if unknown:
            sys.exit(f"unknown maze(s) {unknown} — valid mazes: "
                     f"{', '.join(valid)}")
    else:
        # v8: the whole registry, not the first n_robots names. The fleet is
        # capped at 4 by single-threaded gpu_lidar rendering, but the shared
        # policy still has to generalise over every maze.
        mazes = MR.sorted_registry_names()

    if not mazes or len(set(mazes)) != len(mazes):
        ap.error("choose a non-empty list of distinct mazes")
    if args.n_robots > len(mazes):
        print(f"only {len(mazes)} maze(s) in the pool — running "
              f"{len(mazes)} robots instead of {args.n_robots}", flush=True)
        args.n_robots = len(mazes)
    with training_lock(C.WS_ROOT / ".train_explore_multi.lock"):
        _train(args, mazes)


def _bootstrap_scripted(model, env, steps: int) -> int:
    """Seed the replay buffer with geodesic-demonstrator rollouts (SACfD).

    SAC starts from a uniform-random policy, and a uniform-random policy on
    this task dies against a wall in a handful of steps: the buffer it fills
    during ``learning_starts`` contains almost nothing but crashes, so the
    critic learns "everything is -30" long before it ever sees a zone bonus.
    Demonstrations From Data fixes exactly that — prefill the SAME buffer
    with successful trajectories, then let SAC improve on them off-policy.
    No planner runs during learning; this returns before ``model.learn``.

    ``env_method("scripted_action")`` reaches through the Monitor/VisitMemory
    wrappers (SB3 2.8 resolves it with ``get_wrapper_attr``), so each robot
    answers from its OWN maze and its own zone-visit set.

    Returns the number of vec-env steps taken (each writes ``num_envs``
    transitions).
    """
    model._last_obs = env.reset()
    for _ in range(steps):
        actions = np.asarray(env.env_method("scripted_action"),
                             dtype=np.float32)
        new_obs, rewards, dones, infos = env.step(actions)
        # _store_transition owns the terminal_observation substitution and
        # the _last_obs bookkeeping, so the buffer ends up bit-identical to
        # one SAC filled itself.
        model._store_transition(model.replay_buffer, actions, new_obs,
                                rewards, dones, infos)
    # The seeded transitions are only worth having if SAC updates on them:
    # after --load, learning_starts is set relative to num_timesteps and
    # would make the run refill the buffer from scratch before its first
    # gradient step.
    model.learning_starts = 0
    # Start learn() on fresh episodes. The training callback accumulates
    # episode_return and path_length itself, but ep_len / coverage_frac /
    # min_slack come from the env and span the WHOLE episode, so an episode
    # still in flight here is logged with the demonstrator's step count and
    # coverage beside a return covering only the tail. That is how the
    # ortho_5 row at ts 243052 came out as "608 steps, 25/25 zones, 0.198 m
    # of path" — an incoherent row that reads like a policy solve. The
    # transitions are already in the buffer, so the reset costs nothing.
    model._last_obs = env.reset()
    return steps


def _train(args, mazes) -> None:
    groups = _partition_mazes(mazes, args.n_robots)
    run_dir = args.run_dir or (C.WS_ROOT / "sac_explore_multi_runs" /
                              f"{datetime.now():%Y%m%d-%H%M%S}-{os.getpid()}")
    # Reject reuse to prevent mixing counters and overwriting old experiments.
    run_dir.mkdir(parents=True, exist_ok=False)
    ckpt_dir, log_dir, tb_dir = (run_dir / n for n in
                                ("checkpoints", "logs", "tensorboard"))
    for directory in (ckpt_dir, log_dir, tb_dir):
        directory.mkdir()
    (run_dir / "run.json").write_text(json.dumps({
        "started": datetime.now().isoformat(), "pid": os.getpid(),
        "args": vars(args), "mazes": mazes,
        "assignment": {f"robot_{k}": g for k, g in enumerate(groups, start=1)},
        "observation_dim": C.EXPL_STATE_DIM +
            (C.EXPL_MEMORY_GRID ** 2 if args.visit_memory else 0),
        "gamma": C.EXPL_GAMMA, "target_entropy": C.EXPL_TARGET_ENTROPY,
    }, indent=2, default=str))
    print(f"Run artifacts: {run_dir}", flush=True)

    print("=== multi-robot shared-policy assignment ===")
    for k, group in enumerate(groups, start=1):
        print(f"robot {k} -> mazes {', '.join(group)}")

    env = ConcurrentVecEnv([
        functools.partial(_make_env, robot_id=k, mazes=group,
                          seed=args.seed + k,
                          stall_limit=args.stall_limit,
                          roam_bonus=args.roam_bonus,
                          visit_memory=args.visit_memory)
        for k, group in enumerate(groups, start=1)])

    try:
        if args.load:
            model = SAC.load(args.load, env=env)
            # v6: a checkpoint carries the hyperparameters it was saved with, so a
            # warm start would silently resurrect gamma=0.99 / target_entropy=-2.
            # Both are the v4 failure modes (10 s horizon, entropy collapse), so
            # override them on load.
            model.gamma = C.EXPL_GAMMA
            model.target_entropy = float(C.EXPL_TARGET_ENTROPY)
            model.gradient_steps = args.gradient_steps
            model.tensorboard_log = str(tb_dir)
            # A .zip carries no replay buffer. Without one the resumed run
            # must refill before it can update -- but that warmup is counted
            # from the loaded counter, so an attempt that dies before
            # reaching it performs ZERO gradient updates and acts uniformly
            # at random for its entire life. Restoring the previous
            # attempt's buffer removes the reason to wait at all.
            model.learning_starts = model.num_timesteps + C.SACW_LEARN_STARTS
            if args.load_buffer:
                model.load_replay_buffer(args.load_buffer)
                model.learning_starts = model.num_timesteps
                print(f"Restored replay buffer from {args.load_buffer} "
                      f"({model.replay_buffer.size()} transitions); "
                      f"updates resume immediately", flush=True)
            print(f"Loaded warm-start model from {args.load} "
                  f"(num_timesteps={model.num_timesteps}); "
                  f"gamma={model.gamma} target_entropy={model.target_entropy}",
                  flush=True)
        else:
            model = SAC("MlpPolicy", env,
                        learning_rate=C.SACW_LR,
                        buffer_size=C.SACW_BUFFER_SIZE,
                        batch_size=C.SACW_BATCH_SIZE,
                        gamma=C.EXPL_GAMMA,
                        tau=C.SACW_TAU,
                        learning_starts=C.SACW_LEARN_STARTS,
                        train_freq=C.SACW_TRAIN_FREQ,
                        gradient_steps=args.gradient_steps,
                        policy_kwargs={"net_arch": C.SACW_NET_ARCH},
                        tensorboard_log=str(tb_dir),
                        target_entropy=C.EXPL_TARGET_ENTROPY,
                        seed=args.seed,
                        verbose=1)

        if args.bootstrap_scripted:
            print(f"Bootstrapping the replay buffer with "
                  f"{args.bootstrap_scripted} scripted steps "
                  f"x {env.num_envs} robots...", flush=True)
            _bootstrap_scripted(model, env, args.bootstrap_scripted)
            print(f"Replay buffer seeded "
                  f"({model.replay_buffer.size() * env.num_envs} "
                  f"transitions); learning_starts=0", flush=True)

        callbacks = [
            CheckpointCallback(save_freq=max(args.ckpt_every // len(groups), 1),
                               save_path=str(ckpt_dir),
                               name_prefix="sac_explore_multi"),
            MultiRobotMetricsCallback(mazes, log_dir / "explore_multi.csv"),
            EliteCheckpointsCallback(ckpt_dir=ckpt_dir,
                                     json_path=log_dir / "elites.json",
                                     reset=args.reset_elites),
        ]

        try:
            model.learn(total_timesteps=args.steps, callback=callbacks,
                        reset_num_timesteps=not bool(args.load), progress_bar=False)
        except KeyboardInterrupt:
            print("\nInterrupted — saving model before exit.")
        finally:
            final_path = ckpt_dir / "sac_explore_multi_final"
            model.save(final_path)
            # Saved on crashes too (this is a finally): the supervisor hands
            # it to the next attempt so a restart costs no learning progress.
            model.save_replay_buffer(ckpt_dir / "replay_buffer.pkl")
            print(f"Model saved to {final_path}.zip")
    finally:
        env.close()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
