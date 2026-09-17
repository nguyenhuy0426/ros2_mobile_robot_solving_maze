#!/usr/bin/env python3
"""
gazebo_env.py — Gymnasium environment for ROS2 Gazebo maze solving.

Collision-Recovery Design:
  Collisions do NOT terminate the episode. The robot bounces back to its
  previous safe position and receives a penalty. This lets the robot learn
  what lies beyond its first collision.

7-Action Discrete Design:
  0: STOP
  1: FORWARD   (vx = VX_MAX)
  2: BACKWARD  (vx = VX_MIN)
  3: STRAFE LEFT  (vy = +VY_MAX)
  4: STRAFE RIGHT (vy = -VY_MAX)
  5: TURN LEFT    (wz = +WZ_MAX, in-place)
  6: TURN RIGHT   (wz = -WZ_MAX, in-place)

Observation (FRAME_STACK × 39 + 7 = 163 dim, matches MazeEnv and deploy_gazebo.py):
  Each frame (39):
    [0:36]  36 lidar rays normalized to [0,1]
    [36]    normalized distance to goal
    [37]    cos(heading_to_goal in robot frame)
    [38]    sin(heading_to_goal in robot frame)
  The network input stacks the last 4 frames plus a one-hot of the last action,
  so it matches what deploy_gazebo.py builds at deployment time.

Reward:
  +200   goal reached
  -5     collision (non-terminal, bounce-back)
  +15*(1 + dist_factor)  entering a new cell (closer to goal = more reward)
  -0.02  revisiting a cell
  -0.05  per step (time penalty)
  +5.0   coverage increase bonus
  -0.3   wall proximity warning
  +2.0   distance progress
"""

import math
import subprocess
import threading
import time
from collections import deque
from typing import Optional, Tuple, Dict, Any

import gymnasium as gym
from gymnasium import spaces
import numpy as np

import rclpy
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from geometry_msgs.msg import Pose, Twist
from sensor_msgs.msg import LaserScan

from rl_training import config as C


class GazeboMazeEnv(gym.Env):
    """
    Gymnasium environment that interfaces with Gazebo using ROS2.
    Collision-recovery design: collisions bounce the robot back instead of
    terminating the episode.
    Uses discrete 7-action space for cardinal movement only.
    """
    metadata = {"render_modes": []}

    def __init__(self, robot_id: int = 1,
                 start_x: float = 2.25, start_y: float = -3.25,
                 goal_x: float = 4.25, goal_y: float = -3.25,
                 maze_origin_x: float = 2.0, maze_origin_y: float = -4.5):
        super().__init__()
        self.robot_id = robot_id
        self.start_x = start_x
        self.start_y = start_y
        self.goal_x = goal_x
        self.goal_y = goal_y
        self.maze_origin_x = maze_origin_x
        self.maze_origin_y = maze_origin_y

        # Max possible distance in the maze (for reward normalization)
        self._max_dist = math.sqrt((4.5 - 2.0)**2 + (-2.0 - (-4.5))**2)

        # Initialize ROS2
        if not rclpy.ok():
            rclpy.init()

        # Action: 7 discrete cardinal actions
        self.action_space = spaces.Discrete(C.N_ACTIONS)

        # Observation: 39 dimensions
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(C.STATE_DIM,), dtype=np.float32)

        # Internal state
        self._lock = threading.Lock()
        self._scan = [C.LIDAR_MAX] * C.N_RAYS
        self._x = self._y = self._z = self._yaw = 0.0
        self._prev_dist = 0.0
        self._step_count = 0
        self._visited_cells = set()
        self._prev_coverage = 0.0
        self._collision_count = 0
        self._recent_collisions = deque(maxlen=C.STUCK_WINDOW)
        self._prev_x = start_x
        self._prev_y = start_y
        self._prev_yaw = 0.0
        # Frame-stacked observation state (mirrors MazeEnv):
        self._frame_stack = deque(maxlen=C.FRAME_STACK)
        self._last_action = 0  # STOP — seeds the last-action one-hot

        # Compute direction-dependent collision thresholds for all 36 rays
        L, W = 0.26, 0.20
        x_off = 0.08  # Lidar is mounted at the front (x=+0.08 in sdf)
        x_lo, x_hi = -L / 2 - x_off, L / 2 - x_off
        y_lo, y_hi = -W / 2, W / 2

        self.collision_thresholds = np.zeros(C.N_RAYS, dtype=np.float32)
        for i in range(C.N_RAYS):
            theta = i * (2.0 * math.pi / C.N_RAYS)
            ct, st = math.cos(theta), math.sin(theta)
            cands = []
            if ct > 0:
                d = x_hi / ct
                if abs(d * st) <= y_hi:
                    cands.append(d)
            if ct < 0:
                d = x_lo / ct
                if abs(d * st) <= y_hi:
                    cands.append(d)
            if st > 0:
                d = y_hi / st
                if x_lo <= d * ct <= x_hi:
                    cands.append(d)
            if st < 0:
                d = y_lo / st
                if x_lo <= d * ct <= x_hi:
                    cands.append(d)
            self.collision_thresholds[i] = (min(cands) if cands else 0.10) + C.COLLISION_MARGIN

        # ROS2 Node setup
        self.node = Node(f"gazebo_env_node_{robot_id}")

        qos = rclpy.qos.QoSProfile(
            reliability=rclpy.qos.ReliabilityPolicy.BEST_EFFORT,
            history=rclpy.qos.HistoryPolicy.KEEP_LAST, depth=1)

        self.node.create_subscription(
            LaserScan, f"/scan{robot_id}", self._cb_scan, qos)
        self.node.create_subscription(
            Pose, f"/model/robot_{robot_id}/pose", self._cb_pose, qos)
        self._pub = self.node.create_publisher(
            Twist, f"/model/robot_{robot_id}/cmd_vel", 10)

        # Spin executor in background
        self.executor = MultiThreadedExecutor(num_threads=2)
        self.executor.add_node(self.node)
        self.spin_thread = threading.Thread(target=self.executor.spin, daemon=True)
        self.spin_thread.start()

        time.sleep(1.0)
        self.node.get_logger().info(
            f"GazeboEnv: discrete 7-action. Goal: ({self.goal_x}, {self.goal_y})")

    # ── ROS2 callbacks ──────────────────────────────────────────

    def _cb_scan(self, msg: LaserScan):
        with self._lock:
            self._scan = [
                r if (math.isfinite(r) and r > 0) else C.LIDAR_MAX
                for r in msg.ranges
            ]

    def _cb_pose(self, msg: Pose):
        with self._lock:
            self._x = msg.position.x
            self._y = msg.position.y
            self._z = msg.position.z
            self._yaw = 2.0 * math.atan2(msg.orientation.z, msg.orientation.w)

    # ── Observation ─────────────────────────────────────────────

    def _observe(self) -> np.ndarray:
        """Frame-stacked observation matching MazeEnv._observe:
        FRAME_STACK frames (oldest→newest) + last-action one-hot = STATE_DIM."""
        frame = self._observe_single()
        if len(self._frame_stack) == 0:
            # Episode start: seed the stack with copies of the first frame so
            # the very first observation already has the full STATE_DIM width.
            for _ in range(C.FRAME_STACK):
                self._frame_stack.append(frame)
        else:
            self._frame_stack.append(frame)  # deque(maxlen) evicts the oldest

        action_onehot = np.zeros(C.N_ACTIONS, dtype=np.float32)
        action_onehot[self._last_action] = 1.0
        return np.concatenate(
            list(self._frame_stack) + [action_onehot]).astype(np.float32)

    def _observe_single(self) -> np.ndarray:
        """Build one 39-dim frame: 36 lidar + dist + cos + sin."""
        with self._lock:
            scan_raw = list(self._scan)
            x, y, yaw = self._x, self._y, self._yaw

        n_raw = len(scan_raw)
        if n_raw == C.N_RAYS:
            scan = scan_raw
        else:
            scan = [scan_raw[int(i * n_raw / C.N_RAYS) % n_raw]
                    for i in range(C.N_RAYS)]

        lidar = np.array(
            [min(s, C.LIDAR_MAX) / C.LIDAR_MAX for s in scan],
            dtype=np.float32)

        dx = self.goal_x - x
        dy = self.goal_y - y
        distance = math.sqrt(dx * dx + dy * dy)

        heading = np.array([math.cos(yaw), math.sin(yaw)])
        goal_vec = np.array([dx, dy])
        gn = np.linalg.norm(goal_vec)
        if gn > 1e-6:
            goal_vec = goal_vec / gn
        cos_goal = float(np.dot(heading, goal_vec))
        sin_goal = float(np.cross(heading, goal_vec))

        max_dist = math.sqrt(2) * 5 * C.CELL_SIZE
        norm_dist = min(distance / max_dist, 1.0)

        obs = np.zeros(C.SINGLE_OBS_DIM, dtype=np.float32)
        obs[0:36] = lidar
        obs[36] = norm_dist
        obs[37] = cos_goal
        obs[38] = sin_goal
        return obs

    # ── Helpers ──────────────────────────────────────────────────

    def _dist_to_goal(self) -> float:
        with self._lock:
            x, y = self._x, self._y
        return math.sqrt((x - self.goal_x) ** 2 + (y - self.goal_y) ** 2)

    def _get_cell(self, x: float, y: float) -> Tuple[int, int]:
        col = int(np.clip((x - self.maze_origin_x) / C.CELL_SIZE, 0, 4))
        row = int(np.clip((y - self.maze_origin_y) / C.CELL_SIZE, 0, 4))
        return (row, col)

    def _cell_center(self, row: int, col: int) -> Tuple[float, float]:
        """Get world coordinates of the center of a grid cell."""
        cx = self.maze_origin_x + (col + 0.5) * C.CELL_SIZE
        cy = self.maze_origin_y + (row + 0.5) * C.CELL_SIZE
        return cx, cy

    def teleport(self, x: float, y: float, yaw: float = 0.0):
        qz = math.sin(yaw / 2)
        qw = math.cos(yaw / 2)
        req = (f"name: 'robot_{self.robot_id}' "
               f"position {{ x: {x} y: {y} z: 0.024 }} "
               f"orientation {{ x: 0.0 y: 0.0 z: {qz:.6f} w: {qw:.6f} }}")
        try:
            subprocess.run(
                ["gz", "service", "-s", "/world/nhom8_mecanum/set_pose",
                 "--reqtype", "gz.msgs.Pose", "--reptype", "gz.msgs.Boolean",
                 "--req", req, "--timeout", "2000"],
                capture_output=True, timeout=3.0)
        except (subprocess.TimeoutExpired, FileNotFoundError):
            pass

    # ── Reset / Step ────────────────────────────────────────────

    def reset(self, *, seed: Optional[int] = None,
              options: Optional[Dict[str, Any]] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        super().reset(seed=seed)

        self._pub.publish(Twist())
        self.teleport(self.start_x, self.start_y, 0.0)
        time.sleep(1.0)

        self._step_count = 0
        self._prev_dist = self._dist_to_goal()
        self._visited_cells = set()
        self._prev_coverage = 0.0
        self._prev_x = self.start_x
        self._prev_y = self.start_y
        self._prev_yaw = 0.0
        # Reset the frame-stack state so each episode starts fresh (mirrors MazeEnv).
        self._frame_stack.clear()
        self._last_action = 0

        obs = self._observe()
        return obs, {}

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        action = int(action)
        self._last_action = action  # recorded into the next observation's one-hot

        # Map discrete action to velocities (no EMA — clean cardinal movement)
        vx, vy, wz = C.DISCRETE_ACTIONS[action]

        # Save safe position before moving
        with self._lock:
            safe_x, safe_y, safe_yaw = self._x, self._y, self._yaw

        # Send cmd_vel
        msg = Twist()
        msg.linear.x = vx
        msg.linear.y = vy
        msg.angular.z = wz
        self._pub.publish(msg)

        time.sleep(C.DT)

        # Read state after move
        obs = self._observe()
        dist = self._dist_to_goal()
        with self._lock:
            scan_list = list(self._scan)
            x, y = self._x, self._y
        min_scan = min(scan_list)

        # Ray-by-ray collision check
        collided = False
        for idx in range(min(len(scan_list), C.N_RAYS)):
            if scan_list[idx] < self.collision_thresholds[idx]:
                collided = True
                break

        # Check out-of-bounds
        grid_size = 5 # 5x5 maze
        x_min = self.maze_origin_x
        x_max = x_min + grid_size * C.CELL_SIZE
        y_min = self.maze_origin_y
        y_max = y_min + grid_size * C.CELL_SIZE
        out_of_bounds = not (x_min <= x <= x_max and y_min <= y <= y_max)

        # ── Reward computation ──────────────────────────────────
        terminated = False
        truncated = False
        info = {"dist_to_goal": dist, "min_scan": min_scan}

        if dist < C.GOAL_RADIUS:
            # GOAL REACHED
            reward = C.R_GOAL
            terminated = True
            info["success"] = True
            self.node.get_logger().info("🏁 GOAL REACHED!")

        elif out_of_bounds:
            # ESCAPED — strict termination
            reward = C.R_COLLISION
            terminated = True
            info["collision"] = True
            self.node.get_logger().warn(f"Robot escaped maze! x={x:.2f}, y={y:.2f}. Terminating.")

        elif collided:
            # COLLISION — non-terminal bounce-back
            self._collision_count += 1
            reward = C.R_COLLISION
            info["collision_bounce"] = True
            # Teleport back to safe position
            self.teleport(self._prev_x, self._prev_y, self._prev_yaw)
            time.sleep(0.05)
            if self._collision_count >= C.MAX_COLLISIONS_PER_EP:
                terminated = True
                info["collision"] = True

        else:
            reward = 0.0

            # 1) Distance progress
            reward += C.R_DIST_SCALE * (self._prev_dist - dist)

            # 2) Exploration: distance-based progressive reward
            cell = self._get_cell(x, y)
            if cell not in self._visited_cells:
                # Closer to goal = higher reward (inversely proportional to distance)
                dist_factor = max(0.0, 1.0 - dist / self._max_dist)
                cell_reward = C.R_NEW_CELL_BASE * (1.0 + dist_factor * 2.0)
                reward += cell_reward
                self._visited_cells.add(cell)
                self.node.get_logger().info(
                    f"🧭 NEW cell {cell}! Visited: {len(self._visited_cells)}/{C.MAZE_CELLS} "
                    f"(reward: +{cell_reward:.1f})")
            else:
                reward += C.R_REVISIT

            # 3) Coverage bonus
            coverage = len(self._visited_cells) / C.MAZE_CELLS
            if coverage > self._prev_coverage:
                reward += C.R_COVERAGE_BONUS * (coverage - self._prev_coverage) * C.MAZE_CELLS
            self._prev_coverage = coverage

            # 4) Proximity warning
            if min_scan < C.R_PROX_THRESH:
                reward += C.R_PROX_SCALE * (C.R_PROX_THRESH - min_scan)

            # 5) Time penalty
            reward += C.R_TIME_PENALTY

            # Save safe position for next bounce-back
            self._prev_x = x
            self._prev_y = y
            with self._lock:
                self._prev_yaw = self._yaw

        self._step_count += 1
        if self._step_count >= C.MAX_STEPS:
            truncated = True
            info["timeout"] = True

        info["cells_visited"] = len(self._visited_cells)

        self._prev_dist = dist

        if terminated or truncated:
            self._pub.publish(Twist())

        return obs, float(reward), terminated, truncated, info

    def close(self):
        self._pub.publish(Twist())
        self.executor.shutdown()
        self.node.destroy_node()
