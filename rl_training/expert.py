#!/usr/bin/env python3
"""
expert.py — Perfect BFS pathfinder navigation policy for MazeEnv.
Used to pre-fill the replay buffer during warmup with expert transitions.
7-action discrete version: returns an integer action index.
"""

import math
import numpy as np
from rl_training.maze_env import MazeEnv
from rl_training import config as C

def get_expert_action(env: MazeEnv) -> int:
    """
    Compute optimal discrete action using BFS path.

    The expert:
    1. Solves BFS to get the shortest path
    2. Determines the next waypoint cell
    3. Computes the angle to the waypoint in robot frame
    4. If facing the waypoint (within ±30°): FORWARD
    5. If waypoint is to the left: first check if strafing helps, else TURN LEFT
    6. If waypoint is to the right: first check if strafing helps, else TURN RIGHT

    Args:
        env: MazeEnv instance
    Returns:
        action: integer in [0, 6]
    """
    if env._maze is None:
        return 0  # STOP

    # 1. Solve shortest path in the current maze layout
    path = env._maze.solve_bfs()  # List of (row, col) cells from start to goal
    if not path:
        return 0  # STOP

    # 2. Stateful tracking of waypoint index
    if not hasattr(env, "_expert_target_idx") or env._step_count <= 1:
        env._expert_target_idx = 0

    # Ensure index is in valid bounds
    env._expert_target_idx = min(env._expert_target_idx, len(path) - 1)
    target_cell = path[env._expert_target_idx]

    # 3. Get target cell center in world coordinates
    tx, ty = env._maze.cell_center(*target_cell)

    # 4. Compute vector to target cell center
    dx = tx - env._rx
    dy = ty - env._ry
    dist = math.hypot(dx, dy)

    # 5. If close to current target cell, advance to the next cell
    if dist < 0.07 and env._expert_target_idx < len(path) - 1:
        env._expert_target_idx += 1
        target_cell = path[env._expert_target_idx]
        tx, ty = env._maze.cell_center(*target_cell)
        dx = tx - env._rx
        dy = ty - env._ry
        dist = math.hypot(dx, dy)

    if dist < 1e-4:
        return 0  # STOP — already at waypoint

    # 6. Compute angle to target in robot frame
    target_angle_world = math.atan2(dy, dx)
    angle_diff = target_angle_world - env._yaw
    # Wrap to [-pi, pi]
    while angle_diff > math.pi:
        angle_diff -= 2 * math.pi
    while angle_diff < -math.pi:
        angle_diff += 2 * math.pi

    # 7. Decision logic: prefer facing forward, then strafe, then turn
    #    angle_diff: 0 = target is directly ahead
    #                +pi/2 = target is to the left
    #                -pi/2 = target is to the right
    #                ±pi = target is behind

    abs_angle = abs(angle_diff)

    if abs_angle < math.radians(30):
        # Target roughly ahead → FORWARD
        return 1  # FORWARD
    elif abs_angle > math.radians(150):
        # Target roughly behind → BACKWARD
        return 2  # BACKWARD
    elif math.radians(60) < abs_angle < math.radians(120):
        # Target roughly to the side → STRAFE
        if angle_diff > 0:
            return 3  # STRAFE LEFT
        else:
            return 4  # STRAFE RIGHT
    else:
        # Need to turn to face the target
        if angle_diff > 0:
            return 5  # TURN LEFT
        else:
            return 6  # TURN RIGHT
