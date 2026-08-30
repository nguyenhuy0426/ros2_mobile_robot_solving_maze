#!/usr/bin/env python3
"""
explore_env.py — Gymnasium env: explore-then-exit task (v5, multi-maze).

v5: ALL 13 decoded mazes are hosted side by side in the combined Gazebo world
``nhom8_maze_multi`` (config.MAZE_MULTI_WORLD_NAME). Since gz cannot hot-swap
worlds, the maze is switched PER EPISODE by teleporting the robot to the
selected maze's start pose. Maze geometry, start poses, exit openings and
coverage zones all come from the maze registry (rl_training.maze_registry):
each maze spec is re-placed at its combined-world grid slot
(``MazeSpec.place_at(multi_placement()[name])``). No v4 hardcoded geometry is
used for maze shape.

Task (mirrors the free-roaming behaviour of the two reference repos, but the
exploration policy itself is learned by RL — no BFS/DFS planner):

  Phase EXPLORE — roam every nook of the CURRENT maze, scan the walls with
    the lidar and build a 2D occupancy map. Reward = one-time bonus per newly
    visited zone; the episode REQUIRES visiting all 25 zones.
  Phase EXIT — once all zones are visited the map is complete: the robot must
    find and take the maze's single exit opening (south border for all 13
    mazes) and leave.

Hard rules:
  * Touching a wall terminates the episode (collision → terminal penalty).
  * Leaving through the exit before the map is complete fails the episode.
  * Approaching a wall ahead is only acceptable while slowing down: a
    clearance-scaled speed penalty r ∝ −v·(1 − clear/thresh) makes fast
    motion near the front wall expensive, slow creep cheap.

Observation (150-dim): 4×36 stacked lidar frames + odometry
[x_norm, y_norm, cos yaw, sin yaw] (x/y normalized to the CURRENT maze's
world-frame bbox → [−1, 1]) + [coverage fraction, exit-phase flag].
Action (2-dim): differential-drive wheel velocities [left, right] ∈ (−1, 1).

Training-only signals: geodesic distance-to-exit field (potential shaping in
the EXIT phase only), pose, coverage counts — never observed by the policy.
"""

import math
import random
import subprocess
import threading
import time
from collections import deque
from typing import Any, Dict, List, Optional, Tuple

import gymnasium as gym
import numpy as np
import rclpy
from gymnasium import spaces
from nav_msgs.msg import OccupancyGrid
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from geometry_msgs.msg import Pose
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Float64

from rl_training import config as C
from rl_training import maze_registry as MR
from rl_training.reward_shaping import (
    StallMonitor,
    StuckTracker,
)
from rl_training.wheel_env import (
    WHEEL_ORDER,
    compute_collision_thresholds,
    map_wheel_action,
)

PHASE_EXPLORE = 0
PHASE_EXIT = 1


def wrap_angle(a: float) -> float:
    """Wrap an angle to (−π, π]."""
    return (a + math.pi) % (2.0 * math.pi) - math.pi


def front_clearance(scan: np.ndarray, ray_angles: np.ndarray,
                    half_fov: float = math.pi / 4.0) -> float:
    """Minimum range among body-frame rays within ±``half_fov`` of heading."""
    rel = np.abs(np.arctan2(np.sin(ray_angles), np.cos(ray_angles)))
    mask = rel <= half_fov
    if not mask.any():
        return float(scan.min())
    return float(scan[mask].min())


def safe_speed_penalty(front_clear: float, speed: float,
                       thresh: float, scale: float) -> float:
    """Clearance-scaled speed penalty: 0 unless front clearance < ``thresh``.

    r = scale · v · (1 − clear/thresh): full strength at contact, fading to
    zero at the threshold — fast near-wall motion is expensive, slow creep
    nearly free ("đi chậm lại để kiểm soát tốc độ").
    """
    if front_clear >= thresh:
        return 0.0
    return float(scale) * abs(speed) * (1.0 - front_clear / thresh)


def build_explore_obs(frames: deque, odom: np.ndarray,
                      coverage_frac: float, phase: int) -> np.ndarray:
    """Stacked lidar + odometry + [coverage fraction, phase flag]."""
    parts = list(frames)
    parts.append(np.asarray(odom, dtype=np.float32))
    parts.append(np.array([coverage_frac, float(phase)], dtype=np.float32))
    return np.concatenate(parts)


def _next_maze_name(names: List[str], cursor: int, rng: random.Random,
                    mode: str) -> Tuple[str, int]:
    """Pure maze-selection helper: returns ``(name, new_cursor)``.

    ``round_robin`` walks ``names`` in order (``cursor`` increments mod n);
    ``random`` draws uniformly via ``rng`` and leaves the cursor untouched.
    Raises ValueError for any other ``mode``.
    """
    if mode == "random":
        return names[rng.randrange(len(names))], cursor
    if mode == "round_robin":
        name = names[cursor % len(names)]
        return name, (cursor + 1) % len(names)
    raise ValueError(f"unknown maze selection mode: {mode!r}")


class GazeboExploreEnv(gym.Env):
    """Multi-maze explore-then-exit env (v5) in the combined Gazebo world.

    Per-episode maze selection (``selection``):
      * ``"round_robin"`` — episodes walk ``maze_names`` in order, wrapping
        mod n (default, ``config.EXPL_V5_SELECTION``).
      * ``"random"`` — uniform draw each episode.

    ``reset(options={"maze_name": "sigma_1"})`` forces one specific maze:
    the name is validated against ``maze_names`` and the round-robin cursor
    is NOT advanced (overrides are out-of-band and must not disturb the
    training cadence). ``reset(seed=s)`` re-seeds the RNG used by ``random``
    selection.
    """

    metadata = {"render_modes": []}

    def __init__(self, robot_id: int = 1, seed: Optional[int] = None,
                 maze_names: Optional[List[str]] = None,
                 selection: Optional[str] = None,
                 world_name: Optional[str] = None,
                 prebuild: bool = False):
        super().__init__()
        self.robot_id = robot_id

        self.observation_space = spaces.Box(
            low=-1.0, high=1.0, shape=(C.EXPL_STATE_DIM,), dtype=np.float32)
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(2,), dtype=np.float32)

        # ── Maze registry (v5): which mazes, where, how to pick ─────
        self._maze_names: List[str] = (
            list(maze_names) if maze_names is not None
            else MR.sorted_registry_names())
        if not self._maze_names:
            raise ValueError("maze_names is empty — nothing to train on")
        self._selection = (selection if selection is not None
                           else C.EXPL_V5_SELECTION)
        if self._selection not in ("round_robin", "random"):
            raise ValueError(
                f"selection must be 'round_robin' or 'random', "
                f"got {self._selection!r}")
        self._world_name = (world_name if world_name is not None
                            else C.MAZE_MULTI_WORLD_NAME)

        self._registry = MR.load_registry()
        unknown = [n for n in self._maze_names if n not in self._registry]
        if unknown:
            raise ValueError(
                f"unknown maze(s) {unknown} — known: "
                f"{', '.join(sorted(self._registry))}")
        # Grid slots must match the combined-world SDF layout, which is
        # generated from the FULL sorted registry (generate_maze_sdf_multi).
        # Lookup is therefore done on the full grid even when maze_names is
        # a subset; identical to multi_placement(maze_names) by default.
        self._placements = MR.multi_placement()
        self._bundles: Dict[str, Dict[str, Any]] = {}
        self._maze_cursor = 0
        self._rng = random.Random(seed)

        # Pre-select the first maze so _maze_name/_spec are never None (the
        # first bundle is therefore built eagerly; the rest lazily on first
        # use unless prebuild=True).
        self._maze_name: str = self._maze_names[0]
        self._bundle = self._bundle_for(self._maze_name)
        self._spec = self._bundle["spec"]
        if prebuild:
            for name in self._maze_names:
                self._bundle_for(name)

        # Training-only helpers
        self._collision_thresh = compute_collision_thresholds()
        self._stuck = StuckTracker(
            C.EXPL_STUCK_WINDOW, C.EXPL_STUCK_MIN_DISP, C.EXPL_STUCK_PENALTY)
        self._stall = StallMonitor(
            C.EXPL_STUCK_WINDOW, C.EXPL_STUCK_MIN_DISP, C.EXPL_STALL_LIMIT)
        self._frames: deque = deque(maxlen=4)

        # Sensor state (guarded by _lock)
        self._lock = threading.Lock()
        self._scan: Optional[np.ndarray] = None
        self._scan_seq = 0
        self._ray_angles: Optional[np.ndarray] = None
        self._pose_xy = self._spec.start_xy_world
        self._pose_yaw = self._spec.start_yaw
        self._pose_seq = 0

        # Episode state
        self._step_count = 0
        self._phase = PHASE_EXPLORE
        self._prev_xy = self._spec.start_xy_world
        self._prev_exit_d: Optional[float] = None
        self._ep_idx = 0

        if not rclpy.ok():
            rclpy.init()
        self.node = Node(f"explore_env_node_{robot_id}")

        qos = rclpy.qos.QoSProfile(
            reliability=rclpy.qos.ReliabilityPolicy.BEST_EFFORT,
            history=rclpy.qos.HistoryPolicy.KEEP_LAST, depth=1)
        self.node.create_subscription(
            LaserScan, f"/scan{robot_id}", self._cb_scan, qos)
        self.node.create_subscription(
            Pose, f"/model/robot_{robot_id}/pose", self._cb_pose, qos)

        self._wheel_pubs = [
            self.node.create_publisher(
                Float64, f"/wheel_{w}_{robot_id}", 10)
            for w in WHEEL_ORDER
        ]

        # Latched 2D map for RViz (transient-local, like map_server)
        map_qos = rclpy.qos.QoSProfile(
            reliability=rclpy.qos.ReliabilityPolicy.RELIABLE,
            durability=rclpy.qos.DurabilityPolicy.TRANSIENT_LOCAL,
            history=rclpy.qos.HistoryPolicy.KEEP_LAST, depth=1)
        self._map_pub = self.node.create_publisher(
            OccupancyGrid, "/map", map_qos)

        self.executor = MultiThreadedExecutor(num_threads=2)
        self.executor.add_node(self.node)
        self._spin_thread = threading.Thread(
            target=self.executor.spin, daemon=True)
        self._spin_thread.start()

    # ── Per-maze artifacts ───────────────────────────────────────

    def _bundle_for(self, name: str) -> Dict[str, Any]:
        """Lazily build (and cache) the artifact bundle of maze ``name``.

        The spec is re-placed at its combined-world grid slot so all
        artifacts (distance field, mapper, zone coverage) are
        placement-consistent with the ``nhom8_maze_multi`` world.
        """
        bundle = self._bundles.get(name)
        if bundle is None:
            spec = self._registry[name].place_at(self._placements[name])
            bundle = {
                "spec": spec,
                "field": MR.build_distance_field(spec),
                "mapper": MR.build_mapper(spec),
                "zone": MR.build_zone_coverage(spec),
            }
            self._bundles[name] = bundle
        return bundle

    def _set_maze(self, name: str) -> None:
        """Activate maze ``name`` for the upcoming episode."""
        self._maze_name = name
        self._bundle = self._bundle_for(name)
        self._spec = self._bundle["spec"]

    @property
    def current_maze(self) -> str:
        """Name of the maze selected for the current (or next) episode."""
        return self._maze_name

    @property
    def _mapper(self):
        """Occupancy mapper of the current maze (v4 attribute compat)."""
        return self._bundle["mapper"]

    @property
    def _zone(self):
        """Zone-coverage tracker of the current maze."""
        return self._bundle["zone"]

    @property
    def _field(self):
        """Geodesic distance-to-exit field of the current maze."""
        return self._bundle["field"]

    # ── ROS callbacks ────────────────────────────────────────────

    def _cb_scan(self, msg: LaserScan) -> None:
        ranges = np.asarray(msg.ranges, dtype=np.float32)
        ranges = np.where(np.isfinite(ranges) & (ranges > 0.0),
                          ranges, C.LIDAR_MAX)
        ranges = np.clip(ranges, C.LIDAR_MIN, C.LIDAR_MAX)
        if ranges.shape[0] != C.N_RAYS:  # defensive resampling
            idx = (np.arange(C.N_RAYS) * ranges.shape[0]) // C.N_RAYS
            ranges = ranges[idx]
        with self._lock:
            self._scan = ranges
            self._scan_seq += 1
            if self._ray_angles is None and msg.angle_increment != 0.0:
                self._ray_angles = (
                    msg.angle_min
                    + np.arange(ranges.shape[0]) * msg.angle_increment
                ).astype(np.float32)

    def _cb_pose(self, msg: Pose) -> None:
        with self._lock:
            self._pose_xy = (msg.position.x, msg.position.y)
            self._pose_yaw = 2.0 * math.atan2(msg.orientation.z,
                                              msg.orientation.w)
            self._pose_seq += 1

    # ── Sensor access ────────────────────────────────────────────

    def _wait_fresh(self, timeout: float = 5.0) -> None:
        with self._lock:
            scan0, pose0 = self._scan_seq, self._pose_seq
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            with self._lock:
                if self._scan_seq > scan0 and self._pose_seq > pose0:
                    return
            time.sleep(0.02)
        raise RuntimeError(
            f"No fresh /scan{self.robot_id} + pose within {timeout}s. "
            "Is Gazebo running (unpaused) and spawn_robot_explore.sh active?")

    def _snapshot(self) -> Tuple[np.ndarray, Tuple[float, float], float]:
        with self._lock:
            if self._scan is None:
                raise RuntimeError(
                    f"No LaserScan received on /scan{self.robot_id} — "
                    "check the ros_gz bridge.")
            return (self._scan.copy(), self._pose_xy, self._pose_yaw)

    def _ray_angles_or_default(self) -> np.ndarray:
        with self._lock:
            if self._ray_angles is not None:
                return self._ray_angles
        return (np.arange(C.N_RAYS) * (2.0 * math.pi / C.N_RAYS)
                ).astype(np.float32)

    def _odom_vector(self, xy: Tuple[float, float],
                     yaw: float) -> np.ndarray:
        # x/y normalized to the CURRENT maze's world bbox → [−1, 1].
        # norm_bounds_world returns (xmin, ymin, width, height) with
        # width/height == xmax−xmin / ymax−ymin.
        xmin, ymin, w, h = self._spec.norm_bounds_world
        nx = float(np.clip((xy[0] - xmin) / max(w, 1e-9) * 2.0 - 1.0,
                           -1.0, 1.0))
        ny = float(np.clip((xy[1] - ymin) / max(h, 1e-9) * 2.0 - 1.0,
                           -1.0, 1.0))
        return np.array([nx, ny, math.cos(yaw), math.sin(yaw)], dtype=np.float32)

    def _stack_obs(self, scan: np.ndarray, xy: Tuple[float, float],
                   yaw: float) -> np.ndarray:
        frame = (scan / C.LIDAR_MAX).astype(np.float32)
        self._frames.append(frame)
        while len(self._frames) < 4:
            self._frames.appendleft(frame.copy())
        cov = self._zone.n_cells / float(self._zone.n_zones)
        return build_explore_obs(self._frames, self._odom_vector(xy, yaw),
                                 cov, self._phase)

    # ── Actuation / teleport ─────────────────────────────────────

    def _publish_wheels(self, wheel_vels: np.ndarray) -> None:
        for pub, w in zip(self._wheel_pubs, wheel_vels):
            pub.publish(Float64(data=float(w)))

    def _apply_action(self, action: np.ndarray) -> np.ndarray:
        action = np.clip(np.asarray(action, dtype=np.float32), -1.0, 1.0)
        self._publish_wheels(map_wheel_action(action, diff_drive=True)
                             * C.EXPL_W_MAX)
        return action

    def _stop_wheels(self) -> None:
        self._publish_wheels(np.zeros(2))

    def _teleport_start(self) -> None:
        sx, sy = self._spec.start_xy_world
        yaw = self._spec.start_yaw
        qz = math.sin(yaw / 2)
        qw = math.cos(yaw / 2)
        req = (f"name: 'robot_{self.robot_id}' "
               f"position {{ x: {sx} y: {sy} "
               f"z: 0.024 }} "
               f"orientation {{ x: 0.0 y: 0.0 z: {qz:.6f} w: {qw:.6f} }}")
        try:
            result = subprocess.run(
                ["gz", "service", "-s",
                 f"/world/{self._world_name}/set_pose",
                 "--reqtype", "gz.msgs.Pose", "--reptype", "gz.msgs.Boolean",
                 "--req", req, "--timeout", "2000"],
                capture_output=True, timeout=4.0)
            if result.returncode != 0:
                self.node.get_logger().warn(
                    f"set_pose failed: {result.stderr.decode(errors='ignore')}")
        except FileNotFoundError:
            raise RuntimeError("`gz` CLI not found — source the Gazebo env.")
        except subprocess.TimeoutExpired:
            self.node.get_logger().warn("set_pose service call timed out")

    # ── Map publishing / saving ──────────────────────────────────

    def _publish_map(self) -> None:
        grid = self._mapper.occupancy_array()
        msg = OccupancyGrid()
        msg.header.frame_id = "map"
        msg.header.stamp = self.node.get_clock().now().to_msg()
        msg.info.resolution = self._mapper.res
        msg.info.width = self._mapper.n
        msg.info.height = self._mapper.n
        msg.info.origin.position.x = self._mapper.ox
        msg.info.origin.position.y = self._mapper.oy
        msg.info.origin.orientation.w = 1.0
        msg.data = grid.flatten().tolist()
        self._map_pub.publish(msg)

    def _save_map(self) -> None:
        try:
            C.EXPL_MAP_DIR.mkdir(parents=True, exist_ok=True)
            stem = f"ep_{self._ep_idx:04d}_{self._maze_name}"
            self._mapper.save_png(C.EXPL_MAP_DIR / f"{stem}.png")
            self._mapper.save_png(C.EXPL_MAP_DIR / "map_latest.png")
        except Exception as exc:  # map saving must never kill training
            self.node.get_logger().warn(f"map save failed: {exc}")

    # ── Gymnasium API ────────────────────────────────────────────

    def reset(self, *, seed: Optional[int] = None,
              options: Optional[Dict[str, Any]] = None
              ) -> Tuple[np.ndarray, Dict[str, Any]]:
        super().reset(seed=seed)
        if seed is not None:
            self._rng = random.Random(seed)

        # Per-episode maze selection (see class docstring for semantics).
        maze_name = options.get("maze_name") if options else None
        if maze_name is not None:
            if maze_name not in self._maze_names:
                raise ValueError(
                    f"unknown maze {maze_name!r} — known: "
                    f"{', '.join(self._maze_names)}")
            self._set_maze(maze_name)   # explicit override: cursor untouched
        else:
            name, self._maze_cursor = _next_maze_name(
                self._maze_names, self._maze_cursor, self._rng,
                self._selection)
            self._set_maze(name)

        self._stop_wheels()
        self._teleport_start()
        time.sleep(C.EXPL_SETTLE_SEC)
        self._wait_fresh()

        scan, xy, yaw = self._snapshot()
        self._frames.clear()
        self._mapper.reset()
        self._zone.reset(xy)
        self._stuck.reset(xy)
        self._stall.reset(xy)
        self._step_count = 0
        self._phase = PHASE_EXPLORE
        self._prev_xy = xy
        self._prev_exit_d = None
        self._ep_idx += 1
        self._publish_map()

        obs = self._stack_obs(scan, xy, yaw)
        info: Dict[str, Any] = {
            "maze": self._maze_name,
            "pos": xy, "phase": self._phase,
            "coverage_cells": self._zone.n_cells,
            "coverage_frac": self._zone.n_cells / float(self._zone.n_zones),
            "map_pct": self._mapper.coverage(),
        }
        return obs, info

    def step(self, action: np.ndarray
             ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        action = self._apply_action(action)
        time.sleep(C.EXPL_DT)
        scan, xy, yaw = self._snapshot()
        self._step_count += 1

        # Ground-truth linear speed from consecutive poses
        speed = math.dist(xy, self._prev_xy) / C.EXPL_DT

        # 2D map update (the maze walls are the mapped obstacles)
        self._mapper.update(xy[0], xy[1], yaw, scan,
                            self._ray_angles_or_default())

        ray_angles = self._ray_angles_or_default()
        min_scan = float(scan.min())
        front_clear = front_clearance(scan, ray_angles)
        collided = bool(np.any(scan < self._collision_thresh))

        # Boundary classification: exit opening vs. elsewhere (registry-
        # driven; per-maze opening + placement).
        exited = MR.exit_crossed(self._spec, self._spec.exit_opening, xy)
        out_of_bounds = MR.out_of_bounds(self._spec, xy)

        info: Dict[str, Any] = {
            "maze": self._maze_name,
            "pos": xy, "phase": self._phase, "min_scan": min_scan,
            "speed": speed, "front_clear": front_clear,
            "coverage_cells": self._zone.n_cells,
            "coverage_frac": self._zone.n_cells / float(self._zone.n_zones),
            "map_pct": self._mapper.coverage(),
        }

        reward = C.EXPL_R_TIME
        terminated = False

        if exited and self._phase == PHASE_EXPLORE:
            reward += C.EXPL_R_EXIT_EARLY
            terminated = True
            info["early_exit"] = True
        elif exited and self._phase == PHASE_EXIT:
            reward += C.EXPL_R_EXIT
            terminated = True
            info["success"] = True
        elif out_of_bounds:
            # Sealed borders make this unreachable except via the gap;
            # treat as a failure (defensive, mirrors v3).
            reward += C.EXPL_R_COLLISION
            terminated = True
            info["out_of_bounds"] = True
        elif collided:
            reward += C.EXPL_R_COLLISION
            terminated = True
            info["collision"] = True
        else:
            # Coverage bonus (tracker returns the bonus for a brand-new zone)
            if self._zone.update(xy) > 0.0:
                reward += C.EXPL_R_CELL
                if (self._zone.n_cells >= self._zone.n_zones
                        and self._phase == PHASE_EXPLORE):
                    self._phase = PHASE_EXIT
                    info["phase"] = self._phase
                    info["map_complete"] = True
                    reward += C.EXPL_R_MAP_DONE
                    self._prev_exit_d = self._field.distance(*xy)
            # EXIT phase: potential-based shaping on geodesic distance
            if self._phase == PHASE_EXIT and self._prev_exit_d is not None:
                d = self._field.distance(*xy)
                reward += C.EXPL_EXIT_POT_SCALE * (self._prev_exit_d - d)
                self._prev_exit_d = d
            # "Slow down near a wall ahead" rule
            reward += safe_speed_penalty(
                front_clear, speed, C.EXPL_SAFE_CLEAR, C.EXPL_SAFE_SPEED_SCALE)
            # Bold-roaming bonus: brisk motion in open space earns; creeping
            # earns nothing (paired with the stall rule below, standing or
            # crawling can never be the safe choice).
            if front_clear >= C.EXPL_SAFE_CLEAR:
                reward += C.EXPL_ROAM_BONUS * speed
            # Anti-stall: immediate penalty, then hard termination — standing
            # still must never be the discounted-safe alternative to acting.
            stuck_pen = self._stuck.update(xy)
            reward += stuck_pen
            info["stuck"] = bool(stuck_pen < 0.0)
            if self._stall.update(xy):
                reward += C.EXPL_R_COLLISION
                terminated = True
                info["stall"] = True

        truncated = False
        if not terminated and self._step_count >= C.EXPL_MAX_STEPS:
            truncated = True
            info["timeout"] = True

        if self._step_count % C.EXPL_MAP_EVERY == 0:
            self._publish_map()

        obs = self._stack_obs(scan, xy, yaw)
        self._prev_xy = xy
        if terminated or truncated:
            self._stop_wheels()
            self._save_map()
        return obs, float(reward), terminated, truncated, info

    def close(self) -> None:
        try:
            self._stop_wheels()
        finally:
            self.executor.shutdown()
            self.node.destroy_node()


def make_explore_env(robot_id: int = 1, seed: Optional[int] = None,
                     maze_names: Optional[List[str]] = None,
                     selection: Optional[str] = None,
                     world_name: Optional[str] = None,
                     prebuild: bool = False) -> GazeboExploreEnv:
    """Factory shared by training and evaluation scripts."""
    return GazeboExploreEnv(robot_id=robot_id, seed=seed,
                            maze_names=maze_names, selection=selection,
                            world_name=world_name, prebuild=prebuild)
