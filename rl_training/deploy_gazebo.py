#!/usr/bin/env python3
"""
deploy_gazebo.py — Deploy trained DQN model in Gazebo simulation.

Loads a trained DQN Q-network checkpoint and runs it on a real Gazebo maze.
The trained policy uses 36 lidar rays as input, matching the real robot's sensor.

Usage:
  # Run on the existing Gazebo maze (must have gz sim running):
  rl_venv/bin/python3 -m rl_training.deploy_gazebo --model dqn_checkpoints/best

  # Run on a specific robot:
  rl_venv/bin/python3 -m rl_training.deploy_gazebo --model dqn_checkpoints/best --robot 1
"""

import argparse
import math
import subprocess
import threading
import time
from collections import deque
from pathlib import Path

import numpy as np
import rclpy
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from geometry_msgs.msg import Pose, Twist
from sensor_msgs.msg import LaserScan

from rl_training.dqn import DQN
from rl_training import config as C


WORLD_NAME = "nhom8_mecanum"


class GazeboAgent(Node):
    """ROS2 node bridging trained DQN model with Gazebo robot."""

    def __init__(self, robot_id: int,
                 goal_x: float, goal_y: float):
        super().__init__(f"rl_agent_{robot_id}")
        self.robot_id = robot_id
        self.goal_x = goal_x
        self.goal_y = goal_y

        self._lock = threading.Lock()
        self._scan = [C.LIDAR_MAX] * C.N_RAYS
        self._x = self._y = self._z = self._yaw = 0.0

        # Frame-stacked observation state (must mirror MazeEnv._observe):
        # FRAME_STACK frames (oldest→newest) + last-action one-hot = STATE_DIM.
        self._frame_stack = deque(maxlen=C.FRAME_STACK)
        self._last_action = 0  # STOP — seeds the last-action one-hot

        qos = rclpy.qos.QoSProfile(
            reliability=rclpy.qos.ReliabilityPolicy.BEST_EFFORT,
            history=rclpy.qos.HistoryPolicy.KEEP_LAST, depth=1)

        self.create_subscription(
            LaserScan, f"/scan{robot_id}", self._cb_scan, qos)
        self.create_subscription(
            Pose, f"/model/robot_{robot_id}/pose", self._cb_pose, qos)
        self._pub = self.create_publisher(
            Twist, f"/model/robot_{robot_id}/cmd_vel", 10)

        self.get_logger().info(
            f"Agent robot_{robot_id} → goal=({goal_x:.2f},{goal_y:.2f})")

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
            self._yaw = 2.0 * math.atan2(
                msg.orientation.z, msg.orientation.w)

    def observe(self) -> np.ndarray:
        """Frame-stacked observation matching MazeEnv._observe: FRAME_STACK frames
        (oldest→newest) + last-action one-hot = STATE_DIM."""
        frame = self._observe_single()
        if len(self._frame_stack) == 0:
            # Episode start: seed the stack with copies of the first frame.
            for _ in range(C.FRAME_STACK):
                self._frame_stack.append(frame)
        else:
            self._frame_stack.append(frame)  # deque(maxlen) evicts the oldest

        action_onehot = np.zeros(C.N_ACTIONS, dtype=np.float32)
        action_onehot[self._last_action] = 1.0
        return np.concatenate(
            list(self._frame_stack) + [action_onehot]).astype(np.float32)

    def _observe_single(self) -> np.ndarray:
        """Build one 39-dim frame (36 lidar + dist + cos + sin) matching training env."""
        with self._lock:
            scan_raw = list(self._scan)
            x, y, yaw = self._x, self._y, self._yaw

        # Resample to N_RAYS if needed
        n_raw = len(scan_raw)
        if n_raw == C.N_RAYS:
            scan = scan_raw
        else:
            scan = [scan_raw[int(i * n_raw / C.N_RAYS) % n_raw]
                    for i in range(C.N_RAYS)]

        lidar = np.array(
            [min(s, C.LIDAR_MAX) / C.LIDAR_MAX for s in scan],
            dtype=np.float32)

        # Goal in robot frame
        dx = self.goal_x - x
        dy = self.goal_y - y
        distance = math.sqrt(dx * dx + dy * dy)

        heading = np.array([math.cos(yaw), math.sin(yaw)])
        goal_vec = np.array([dx, dy])
        goal_norm = np.linalg.norm(goal_vec)
        if goal_norm > 1e-6:
            goal_vec = goal_vec / goal_norm
        cos_goal = float(np.dot(heading, goal_vec))
        sin_goal = float(np.cross(heading, goal_vec))

        max_dist = math.sqrt(2) * 5 * 0.5
        norm_dist = min(distance / max_dist, 1.0)

        obs = np.zeros(C.SINGLE_OBS_DIM, dtype=np.float32)
        obs[0:36] = lidar
        obs[36] = norm_dist
        obs[37] = cos_goal
        obs[38] = sin_goal
        return obs

    def act(self, action: int):
        """Send velocity command from discrete action."""
        self._last_action = action  # recorded into the next observation's one-hot
        vx, vy, wz = C.DISCRETE_ACTIONS[action]

        msg = Twist()
        msg.linear.x = float(vx)
        msg.linear.y = float(vy)
        msg.angular.z = float(wz)
        self._pub.publish(msg)

    def stop(self):
        self._pub.publish(Twist())

    def dist_to_goal(self) -> float:
        with self._lock:
            x, y = self._x, self._y
        return math.sqrt((x - self.goal_x) ** 2 + (y - self.goal_y) ** 2)


def teleport(robot_id: int, x: float, y: float, yaw: float = 0.0):
    """Teleport robot in Gazebo."""
    qz = math.sin(yaw / 2)
    qw = math.cos(yaw / 2)
    req = (f"name: 'robot_{robot_id}' "
           f"position {{ x: {x} y: {y} z: 0.024 }} "
           f"orientation {{ x: 0.0 y: 0.0 z: {qz:.6f} w: {qw:.6f} }}")
    try:
        subprocess.run(
            ["gz", "service", "-s", f"/world/{WORLD_NAME}/set_pose",
             "--reqtype", "gz.msgs.Pose", "--reptype", "gz.msgs.Boolean",
             "--req", req, "--timeout", "2000"],
            capture_output=True, timeout=3.0)
    except (subprocess.TimeoutExpired, FileNotFoundError):
        pass


def run_demo(model_path: str, robot_id: int = 1,
             start_x: float = 2.25, start_y: float = -3.25,
             goal_x: float = 4.25, goal_y: float = -3.25,
             n_episodes: int = 5):
    """Run trained DQN model in Gazebo."""

    print(f"\n  ╔══════════════════════════════════════════════════════╗")
    print(f"  ║  DQN Maze Solver — Gazebo Deployment                ║")
    print(f"  ║  Model: {model_path:<42s}  ║")
    print(f"  ╚══════════════════════════════════════════════════════╝\n")

    model = DQN()
    model.load_actor_only(Path(model_path))
    print(f"  Q-network params: "
          f"{sum(p.numel() for p in model.q_net.parameters()):,}")

    rclpy.init()
    agent = GazeboAgent(robot_id, goal_x, goal_y)
    executor = MultiThreadedExecutor(num_threads=4)
    executor.add_node(agent)
    threading.Thread(target=executor.spin, daemon=True).start()
    time.sleep(2.0)

    try:
        for ep in range(n_episodes):
            print(f"\n  ── Episode {ep + 1}/{n_episodes} ──")
            teleport(robot_id, start_x, start_y, 0.0)
            time.sleep(1.5)

            # Reset frame-stack state so each episode starts fresh (mirrors env.reset).
            agent._frame_stack.clear()
            agent._last_action = 0

            for step in range(C.MAX_STEPS):
                t0 = time.perf_counter()
                obs = agent.observe()
                action = model.get_action(obs, add_noise=False)
                agent.act(action)

                dist = agent.dist_to_goal()

                if step % 50 == 0:
                    x, y = agent._x, agent._y
                    action_name = C.ACTION_NAMES[action]
                    print(f"    step={step:3d}  pos=({x:.2f},{y:.2f})  "
                          f"dist={dist:.2f}m  act={action_name}")

                if dist < C.GOAL_RADIUS:
                    print(f"    🏁 GOAL REACHED at step {step}!")
                    break

                elapsed = time.perf_counter() - t0
                if elapsed < C.DT:
                    time.sleep(C.DT - elapsed)

            agent.stop()
            result = "✓ GOAL" if dist < C.GOAL_RADIUS else f"✗ dist={dist:.2f}m"
            print(f"    Result: {result}  steps={step + 1}")

    except KeyboardInterrupt:
        print("\n  Interrupted")
    finally:
        agent.stop()
        executor.shutdown()
        agent.destroy_node()
        rclpy.shutdown()


def main():
    ap = argparse.ArgumentParser(description="Deploy DQN in Gazebo")
    ap.add_argument("--model", required=True,
                    help="Path to checkpoint directory")
    ap.add_argument("--robot", type=int, default=1, help="Robot ID")
    ap.add_argument("--start-x", type=float, default=2.25)
    ap.add_argument("--start-y", type=float, default=-3.25)
    ap.add_argument("--goal-x", type=float, default=4.25)
    ap.add_argument("--goal-y", type=float, default=-3.25)
    ap.add_argument("--episodes", type=int, default=5)
    args = ap.parse_args()

    run_demo(args.model, args.robot,
             args.start_x, args.start_y,
             args.goal_x, args.goal_y,
             args.episodes)


if __name__ == "__main__":
    main()
