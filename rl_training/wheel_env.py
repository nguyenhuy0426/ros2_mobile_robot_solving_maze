#!/usr/bin/env python3
"""
wheel_env.py — Gymnasium env: LiDAR(+optional odometry) policy, wheel-velocity control.

Policy interface (what the neural network sees/produces):
  Observation: WHEEL_FRAME_STACK stacked LiDAR frames, each 36 rays
               normalized to [0,1]. With ``use_odom=True`` (v3), the stack is
               followed by [x_norm, y_norm, cos(yaw), sin(yaw)] — for the FIXED
               maze this restores near-Markov structure; frame stacking alone
               cannot disambiguate aliased corridors (POMDP).
  Action:      Box(-1, 1, (A,)) → scaled to ±WHEEL_W_MAX rad/s and published
               as wheel angular velocities via gz JointController topics.
               A=4: [fl, fr, rl, rr] (legacy mecanum layout).
               A=2: [left, right] (diff-drive, ``diff_drive=True`` — matches
               the physical robot, whose cylindrical wheels provide no
               lateral propulsion).

Training-only signals (reward/termination, never observed by the policy):
  - model pose from Gazebo PosePublisher (/model/robot_N/pose)
  - geodesic distance-to-goal field parsed from the world SDF (maze_field.py)

Requires a running Gazebo world + ./spawn_robot_wheel.sh (wheel-topic bridges,
VelocityControl plugin stripped so it cannot fight the wheel controllers).
"""

import math
import subprocess
import threading
import time
from collections import deque
from typing import Any, Dict, Optional, Tuple

import gymnasium as gym
import numpy as np
import rclpy
from gymnasium import spaces
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from geometry_msgs.msg import Pose
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Float64

from rl_training import config as C
from rl_training.maze_field import MazeDistanceField
from rl_training.curriculum import StartCurriculum
from rl_training.reward_shaping import (
    CoverageTracker,
    StuckTracker,
    action_smoothness_penalty,
)

WHEEL_ORDER = ("fl", "fr", "rl", "rr")  # action index → wheel


def map_wheel_action(action: np.ndarray, diff_drive: bool) -> np.ndarray:
    """Map a normalized policy action to 4 per-wheel velocity commands.

    diff_drive=False: action is [fl, fr, rl, rr] (legacy 4-DOF mecanum).
    diff_drive=True:  action is [left, right]; each side's two wheels get the
    same command. The physical robot has plain cylindrical wheels with no
    lateral rollers, so it is effectively differential-drive — exposing only
    the 2 real DOFs removes a dead action dimension from the policy search.
    """
    action = np.asarray(action, dtype=np.float32)
    if diff_drive:
        left, right = action
        return np.array([left, right, left, right], dtype=np.float32)
    return action


def build_wheel_obs(frames: deque, use_odom: bool,
                    odom: Optional[np.ndarray] = None) -> np.ndarray:
    """Concatenate stacked LiDAR frames (oldest→newest) with optional odometry.

    Odometry = [x_norm, y_norm, cos(yaw), sin(yaw)] — x/y normalized to [0,1]
    over the maze bounds. For the FIXED maze this makes the task nearly Markov
    (the policy can memorize the corridor value landscape); for randomized
    mazes it stays disabled (legacy LiDAR-only behavior).
    """
    parts = list(frames)
    if use_odom:
        if odom is None:
            raise ValueError("use_odom=True requires an odom vector")
        parts.append(np.asarray(odom, dtype=np.float32))
    return np.concatenate(parts)


def compute_collision_thresholds(n_rays: int = C.N_RAYS,
                                 margin: float = C.COLLISION_MARGIN) -> np.ndarray:
    """Per-ray range below which the rectangular chassis touches a wall.

    Same geometry as GazeboMazeEnv (chassis 0.26x0.20 m, lidar mounted at
    x=+0.08): for each ray angle, distance from the lidar to the chassis
    boundary along that ray, plus a safety margin.
    """
    L, W = 0.26, 0.20
    x_off = 0.08
    x_lo, x_hi = -L / 2 - x_off, L / 2 - x_off
    y_lo, y_hi = -W / 2, W / 2

    thresholds = np.zeros(n_rays, dtype=np.float32)
    for i in range(n_rays):
        theta = i * (2.0 * math.pi / n_rays)
        ct, st = math.cos(theta), math.sin(theta)
        cands = []
        if ct > 0 and abs((x_hi / ct) * st) <= y_hi:
            cands.append(x_hi / ct)
        if ct < 0 and abs((x_lo / ct) * st) <= y_hi:
            cands.append(x_lo / ct)
        if st > 0 and x_lo <= (y_hi / st) * ct <= x_hi:
            cands.append(y_hi / st)
        if st < 0 and x_lo <= (y_lo / st) * ct <= x_hi:
            cands.append(y_lo / st)
        thresholds[i] = (min(cands) if cands else 0.10) + margin
    return thresholds


class GazeboWheelEnv(gym.Env):
    """Gazebo maze env: stacked-LiDAR(+optional odom) observation, wheel-velocity
    action (4-DOF legacy or 2-DOF differential-drive)."""

    metadata = {"render_modes": []}

    def __init__(self, robot_id: int = 1, shaping: bool = False,
                 curriculum: bool = False, seed: Optional[int] = None,
                 diff_drive: Optional[bool] = None,
                 use_odom: Optional[bool] = None,
                 collision_penalty: Optional[float] = None,
                 goal_radius: Optional[float] = None,
                 stuck_penalty: Optional[float] = None):
        super().__init__()
        self.robot_id = robot_id
        self._shaping = shaping

        # v3 opt-ins; None → config default (False, keeps v1/v2 reproducible).
        self._diff_drive = C.WHEEL_DIFF_DRIVE if diff_drive is None else diff_drive
        self._use_odom = C.WHEEL_USE_ODOM if use_odom is None else use_odom
        self._collision_penalty = (
            collision_penalty if collision_penalty is not None else
            (C.WR_COLLISION_V2 if shaping else C.WR_COLLISION))
        self._goal_radius = (goal_radius if goal_radius is not None
                             else C.GOAL_RADIUS)  # legacy 0.15; v3 opts into 0.20

        obs_dim = C.WHEEL_FRAME_STACK * C.N_RAYS
        if self._use_odom:
            obs_dim += C.WHEEL_ODOM_DIM
        action_dim = 2 if self._diff_drive else C.WHEEL_ACTION_DIM

        self.observation_space = spaces.Box(
            low=-1.0, high=1.0, shape=(obs_dim,), dtype=np.float32)
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(action_dim,), dtype=np.float32)

        self._geo = MazeDistanceField(C.WORLD_SDF, goal_xy=C.GOAL_XY)
        self._collision_thresh = compute_collision_thresholds()
        self._frames: deque = deque(maxlen=C.WHEEL_FRAME_STACK)

        # Normalized-odometry helper: maze spans origin..origin + 5 cells.
        self._odom_span = 5 * C.CELL_SIZE

        # Reward-shaping v2 (opt-in). Softened collision + anti-stall +
        # exploration bonus + action smoothing. Obs stays LiDAR-only.
        self._stuck = StuckTracker(
            C.WR_STUCK_WINDOW, C.WR_STUCK_MIN_DISP,
            stuck_penalty if stuck_penalty is not None
            else C.WR_STUCK_PENALTY) \
            if shaping else None
        self._coverage = CoverageTracker(
            C.MAZE_ORIGIN_XY, C.CELL_SIZE, C.WR_COVERAGE_BONUS,
            grid_cells=C.CURR_GRID_CELLS) \
            if shaping else None
        self._prev_action: Optional[np.ndarray] = None

        # Start-distance curriculum (opt-in; env-side only).
        self._curriculum = StartCurriculum(
            self._geo, C.MAZE_ORIGIN_XY, C.GOAL_XY, C.CURR_GRID_CELLS,
            C.CELL_SIZE, C.GOAL_RADIUS, C.CURR_SUCCESS_WINDOW,
            C.CURR_ADVANCE_THRESH, final_start_xy=C.START_XY, seed=seed) \
            if curriculum else None
        self._start_xy = C.START_XY

        # Latest sensor state (guarded by _lock)
        self._lock = threading.Lock()
        self._scan: Optional[np.ndarray] = None
        self._scan_seq = 0
        self._pose_xy = C.START_XY
        self._pose_seq = 0
        self._pose_yaw = C.START_YAW

        self._step_count = 0
        self._prev_geo_dist = 0.0

        if not rclpy.ok():
            rclpy.init()
        self.node = Node(f"wheel_env_node_{robot_id}")

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

        self.executor = MultiThreadedExecutor(num_threads=2)
        self.executor.add_node(self.node)
        self._spin_thread = threading.Thread(
            target=self.executor.spin, daemon=True)
        self._spin_thread.start()

    # ── ROS callbacks ────────────────────────────────────────────

    def _cb_scan(self, msg: LaserScan) -> None:
        ranges = np.asarray(msg.ranges, dtype=np.float32)
        # sanitize: NaN/Inf/invalid → max range, then clip to sensor bounds
        ranges = np.where(np.isfinite(ranges) & (ranges > 0.0),
                          ranges, C.LIDAR_MAX)
        ranges = np.clip(ranges, C.LIDAR_MIN, C.LIDAR_MAX)
        if ranges.shape[0] != C.N_RAYS:  # defensive resampling
            idx = (np.arange(C.N_RAYS) * ranges.shape[0]) // C.N_RAYS
            ranges = ranges[idx]
        with self._lock:
            self._scan = ranges
            self._scan_seq += 1

    def _cb_pose(self, msg: Pose) -> None:
        with self._lock:
            self._pose_xy = (msg.position.x, msg.position.y)
            self._pose_seq += 1
            self._pose_yaw = 2.0 * math.atan2(msg.orientation.z,
                                              msg.orientation.w)

    # ── Sensor access ────────────────────────────────────────────

    def _wait_fresh(self, timeout: float = 5.0) -> None:
        """Block until at least one new scan AND pose arrive."""
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
            "Is Gazebo running (unpaused) and spawn_robot_wheel.sh active?")

    def _snapshot(self) -> Tuple[np.ndarray, Tuple[float, float]]:
        with self._lock:
            if self._scan is None:
                raise RuntimeError(
                    f"No LaserScan received on /scan{self.robot_id} — "
                    "check the ros_gz bridge.")
            return self._scan.copy(), self._pose_xy

    def _stack_obs(self, scan: np.ndarray) -> np.ndarray:
        frame = (scan / C.LIDAR_MAX).astype(np.float32)
        self._frames.append(frame)
        while len(self._frames) < C.WHEEL_FRAME_STACK:
            self._frames.appendleft(frame.copy())
        return build_wheel_obs(self._frames, self._use_odom, self._odom_vector())

    def _odom_vector(self) -> Optional[np.ndarray]:
        """[x_norm, y_norm, cos(yaw), sin(yaw)] or None when odom is disabled.

        Yaw comes from the PosePublisher quaternion (planar robot, so the
        yaw = 2·atan2(z, w) decomposition is exact)."""
        if not self._use_odom:
            return None
        with self._lock:
            x, y = self._pose_xy
        nx = float(np.clip((x - C.MAZE_ORIGIN_XY[0]) / self._odom_span, 0.0, 1.0))
        ny = float(np.clip((y - C.MAZE_ORIGIN_XY[1]) / self._odom_span, 0.0, 1.0))
        # _pose_yaw is maintained by the pose callback
        cy = math.cos(self._pose_yaw)
        sy = math.sin(self._pose_yaw)
        return np.array([nx, ny, cy, sy], dtype=np.float32)

    # ── Actuation ────────────────────────────────────────────────

    def _publish_wheels(self, wheel_vels: np.ndarray) -> None:
        for pub, w in zip(self._wheel_pubs, wheel_vels):
            pub.publish(Float64(data=float(w)))

    def _apply_action(self, action: np.ndarray) -> np.ndarray:
        """Clip the policy action, publish per-wheel commands, and return the
        clipped action (so shaping/telemetry see what was actually sent)."""
        action = np.clip(np.asarray(action, dtype=np.float32), -1.0, 1.0)
        self._publish_wheels(map_wheel_action(action, self._diff_drive)
                             * C.WHEEL_W_MAX)
        return action

    def _stop_wheels(self) -> None:
        self._publish_wheels(np.zeros(C.WHEEL_ACTION_DIM))

    def _teleport_start(self) -> None:
        self._teleport_to(C.START_XY)

    def _teleport_to(self, xy: Tuple[float, float]) -> None:
        qz = math.sin(C.START_YAW / 2)
        qw = math.cos(C.START_YAW / 2)
        req = (f"name: 'robot_{self.robot_id}' "
               f"position {{ x: {xy[0]} y: {xy[1]} z: 0.024 }} "
               f"orientation {{ x: 0.0 y: 0.0 z: {qz:.6f} w: {qw:.6f} }}")
        try:
            result = subprocess.run(
                ["gz", "service", "-s", f"/world/{C.WORLD_NAME}/set_pose",
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

    # ── Gymnasium API ────────────────────────────────────────────

    def reset(self, *, seed: Optional[int] = None,
              options: Optional[Dict[str, Any]] = None
              ) -> Tuple[np.ndarray, Dict[str, Any]]:
        super().reset(seed=seed)
        self._stop_wheels()
        if self._curriculum is not None:
            self._start_xy = self._curriculum.sample_start()
        self._teleport_to(self._start_xy)
        time.sleep(C.WHEEL_SETTLE_SEC)
        self._wait_fresh()

        scan, xy = self._snapshot()
        self._frames.clear()
        obs = self._stack_obs(scan)

        self._step_count = 0
        self._prev_geo_dist = self._geo.distance(*xy)
        self._prev_action = None
        if self._stuck is not None:
            self._stuck.reset(xy)
        if self._coverage is not None:
            self._coverage.reset(xy)

        info: Dict[str, Any] = {"geo_dist": self._prev_geo_dist}
        if self._curriculum is not None:
            info["curriculum_level"] = self._curriculum.level
        return obs, info

    def step(self, action: np.ndarray
             ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        action = self._apply_action(action)

        time.sleep(C.WHEEL_DT)

        scan, xy = self._snapshot()
        obs = self._stack_obs(scan)
        self._step_count += 1

        geo_dist = self._geo.distance(*xy)
        euclid_goal = math.dist(xy, C.GOAL_XY)
        min_scan = float(scan.min())
        collided = bool(np.any(scan < self._collision_thresh))

        # Hard containment: the episode terminates if the robot leaves the
        # maze rectangle. Prevents any unbounded-roaming exploit regardless
        # of world edits (mirrors GazeboMazeEnv's out-of-bounds check).
        x_min, y_min = C.MAZE_ORIGIN_XY
        x_max = x_min + C.CURR_GRID_CELLS * C.CELL_SIZE
        y_max = y_min + C.CURR_GRID_CELLS * C.CELL_SIZE
        out_of_bounds = not (x_min <= xy[0] <= x_max
                             and y_min <= xy[1] <= y_max)

        terminated = False
        truncated = False
        info: Dict[str, Any] = {
            "geo_dist": geo_dist, "min_scan": min_scan, "pos": xy}

        if euclid_goal < self._goal_radius:
            reward = C.WR_GOAL
            terminated = True
            info["success"] = True
        elif out_of_bounds:
            reward = self._collision_penalty
            terminated = True
            info["out_of_bounds"] = True
            info["collision"] = True  # counted as a failed episode
        elif collided:
            reward = self._collision_penalty
            terminated = True
            info["collision"] = True
        else:
            reward = C.WR_PROGRESS * (self._prev_geo_dist - geo_dist)
            reward += C.WR_TIME
            if min_scan < C.WR_PROX_THRESH:
                reward += C.WR_PROX_SCALE * (C.WR_PROX_THRESH - min_scan)
            if self._shaping:
                reward += self._shaping_terms(action, xy, info)

        if self._step_count >= C.WHEEL_MAX_STEPS and not terminated:
            truncated = True
            info["timeout"] = True

        # Action-saturation telemetry (evaluation/diagnosis; free to compute).
        info["saturation"] = float(np.mean(np.abs(action) > 0.95))

        self._prev_geo_dist = geo_dist
        self._prev_action = action
        if terminated or truncated:
            self._stop_wheels()
            if self._curriculum is not None:
                self._curriculum.record_outcome(bool(info.get("success")))
                info["curriculum_level"] = self._curriculum.level

        return obs, float(reward), terminated, truncated, info

    def _shaping_terms(self, action: np.ndarray, xy: Tuple[float, float],
                       info: Dict[str, Any]) -> float:
        """Extra reward-v2 terms (anti-stall, exploration, smoothing).

        Only called on non-terminal steps when ``shaping`` is enabled; mutates
        ``info`` with diagnostics but never with terminal flags.
        """
        extra = 0.0
        stuck_pen = self._stuck.update(xy)
        extra += stuck_pen
        info["stuck"] = bool(stuck_pen < 0.0)
        cover_bonus = self._coverage.update(xy)
        extra += cover_bonus
        info["coverage_cells"] = self._coverage.n_cells
        extra += action_smoothness_penalty(
            action, self._prev_action, C.WR_SMOOTH_SCALE)
        return extra

    def close(self) -> None:
        try:
            self._stop_wheels()
        finally:
            self.executor.shutdown()
            self.node.destroy_node()


def make_wheel_env(robot_id: int = 1, shaping: bool = False,
                   curriculum: bool = False,
                   seed: Optional[int] = None,
                   diff_drive: Optional[bool] = None,
                   use_odom: Optional[bool] = None,
                   collision_penalty: Optional[float] = None,
                   goal_radius: Optional[float] = None,
                   stuck_penalty: Optional[float] = None) -> GazeboWheelEnv:
    """Factory shared by training and evaluation scripts.

    ``shaping``/``curriculum`` default to False so the original v1 baseline
    (sac_wheel_checkpoints) stays byte-for-byte reproducible; the v2, TD3 and
    v3 training scripts opt in. The v3 interface knobs (``diff_drive``,
    ``use_odom``, penalty/radius overrides) also default to the legacy config
    values so existing v1/v2 checkpoints keep loading unchanged.
    """
    return GazeboWheelEnv(robot_id=robot_id, shaping=shaping,
                          curriculum=curriculum, seed=seed,
                          diff_drive=diff_drive, use_odom=use_odom,
                          collision_penalty=collision_penalty,
                          goal_radius=goal_radius,
                          stuck_penalty=stuck_penalty)
