#!/usr/bin/env python3
"""
maze_env.py — Gymnasium environment for 5×5 maze with lidar raycasting.

Fast Python-based simulator matching Gazebo robot physics:
  - Mecanum kinematics (vx, vy, wz in robot frame)
  - 36-ray lidar with 360° coverage
  - Accurate wall collision detection
  - Random maze per episode for generalization

7-Action Discrete Design:
  0: STOP
  1: FORWARD   (vx = VX_MAX)
  2: BACKWARD  (vx = VX_MIN)
  3: STRAFE LEFT  (vy = +VY_MAX)
  4: STRAFE RIGHT (vy = -VY_MAX)
  5: TURN LEFT    (wz = +WZ_MAX, in-place)
  6: TURN RIGHT   (wz = -WZ_MAX, in-place)

Observation (FRAME_STACK × 39 + 7 = 163 dim):
  Per frame (39): [0:36] 36 lidar rays normalized to [0,1], [36] goal distance
  normalized, [37] cos and [38] sin of heading-to-goal in robot frame.
  FRAME_STACK frames are concatenated (oldest→newest) plus the last-action one-hot
  (N_ACTIONS), giving temporal memory so the policy can disambiguate
  perceptually-aliased states (look-alike junctions/dead-ends).

Reward (goal-dominant, observation-only design):
  +200   goal reached (terminal — the single dominant attractor)
  -5     collision (NON-terminal, bounce-back; episodes end only on goal or timeout)
  +2.0 × distance progress toward goal (potential-based; observable via obs[36])
  -0.3 × wall-proximity warning (observable via lidar)
  -0.1   per step when the action doesn't translate (STOP/TURN) — anti-spin shaping
  -0.05  per step (time penalty)
  NOTE: no visit-history shaping (new-cell/coverage/revisit). Those depend on
        hidden state not in the observation, so they inject unobservable noise.
"""

from collections import deque
import math
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Optional, Tuple, Dict, Any

from rl_training.maze_gen import Maze, random_start_goal
from rl_training import config as C


class MazeEnv(gym.Env):
    """
    Fast Python maze environment with vectorized lidar raycasting.

    Each reset() generates a new random maze, ensuring the agent
    learns a general navigation policy rather than memorizing one layout.

    Uses discrete 7-action space for cardinal movement only.
    """

    metadata = {"render_modes": ["human", "ansi"]}

    def __init__(self, render_mode: Optional[str] = None,
                 fixed_seed: Optional[int] = None,
                 curriculum_level: int = -1):
        super().__init__()
        self.render_mode = render_mode
        self._fixed_seed = fixed_seed
        self._curriculum = curriculum_level
        self._episode_count = 0

        # Observation: 36 lidar + dist + cos + sin
        self.observation_space = spaces.Box(
            low=-1.0, high=1.0, shape=(C.STATE_DIM,), dtype=np.float32)

        # Action: 7 discrete cardinal actions
        self.action_space = spaces.Discrete(C.N_ACTIONS)

        # State
        self._maze: Optional[Maze] = None
        self._segments: Optional[np.ndarray] = None
        self._rx = self._ry = self._yaw = 0.0
        self._gx = self._gy = 0.0
        self._prev_dist = 0.0
        self._step_count = 0
        self._rng = np.random.default_rng()

    def reset(self, seed: Optional[int] = None,
              options: Optional[Dict[str, Any]] = None) -> Tuple[np.ndarray, Dict]:
        super().reset(seed=seed)
        self._episode_count += 1

        maze_seed = self._fixed_seed if self._fixed_seed is not None else (
            seed if seed is not None else self._rng.integers(0, 1_000_000))

        rng = np.random.default_rng(maze_seed)
        py_rng = __import__('random').Random(int(maze_seed))

        self._maze = Maze()

        # Curriculum: control maze complexity via start/goal distance
        if self._curriculum == 0:
            start, goal = random_start_goal(rng=py_rng, min_manhattan=2)
        elif self._curriculum == 1:
            start, goal = random_start_goal(rng=py_rng, min_manhattan=4)
        else:
            start, goal = random_start_goal(rng=py_rng, min_manhattan=5)

        self._maze.generate(seed=int(maze_seed), start=start, goal=goal)
        self._segments = self._maze.wall_segments()

        # Robot initial position + small noise
        sx, sy = self._maze.start_pos()
        self._rx = sx + rng.uniform(-0.03, 0.03)
        self._ry = sy + rng.uniform(-0.03, 0.03)
        self._yaw = rng.uniform(-math.pi, math.pi)

        self._gx, self._gy = self._maze.goal_pos()
        self._prev_dist = self._dist_to_goal()
        self._step_count = 0
        self._visited_cells = set()
        self._collision_count = 0
        self._recent_collisions = deque(maxlen=C.STUCK_WINDOW)
        self._last_action = 0  # STOP — seeds the last-action one-hot in the observation
        self._frame_stack = deque(maxlen=C.FRAME_STACK)

        obs = self._observe()
        info = {"maze_seed": int(maze_seed),
                "path_length": self._maze.path_length()}
        return obs, info

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        self._step_count += 1
        action = int(action)
        self._last_action = action  # recorded into the observation's last-action one-hot

        # Map discrete action to velocities (no EMA — clean cardinal movement)
        vx, vy, wz = C.DISCRETE_ACTIONS[action]
        is_nonmotion = (vx == 0.0 and vy == 0.0)  # STOP / TURN_L / TURN_R don't translate

        # Mecanum kinematics: robot frame → world frame
        cos_y = math.cos(self._yaw)
        sin_y = math.sin(self._yaw)
        dx_world = vx * cos_y - vy * sin_y
        dy_world = vx * sin_y + vy * cos_y

        new_x = self._rx + dx_world * C.DT
        new_y = self._ry + dy_world * C.DT
        new_yaw = self._wrap(self._yaw + wz * C.DT)

        collided = self._check_collision(new_x, new_y)

        if not collided:
            self._rx = new_x
            self._ry = new_y
            self._yaw = new_yaw

        # ── Reward (strict maze exploration design) ─────────────────
        dist = self._dist_to_goal()
        scan = self._raycast()
        min_scan = float(np.min(scan))

        # Check out-of-bounds (maze always spans local [0, GRID*CELL] on both axes)
        grid_size = 5 # 5x5 maze
        x_min = y_min = 0.0
        x_max = y_max = grid_size * C.CELL_SIZE
        out_of_bounds = not (x_min <= self._rx <= x_max and y_min <= self._ry <= y_max)

        terminated = False
        truncated = False
        info: Dict[str, Any] = {}

        if dist < C.GOAL_RADIUS:
            reward = C.R_GOAL
            terminated = True
            info["success"] = True
        elif out_of_bounds:
            # ESCAPED — strict termination
            reward = C.R_COLLISION
            terminated = True
            info["collision"] = True

        elif collided:
            # COLLISION — non-terminal bounce-back. Episodes end only on goal/timeout,
            # so translating in tight corridors is no longer punished by death; this
            # removes the incentive to freeze/spin in place to "survive".
            self._collision_count += 1
            self._recent_collisions.append(1)
            reward = C.R_COLLISION
            info["collision_bounce"] = True
            # Don't move — stay at previous safe position.
            if is_nonmotion:
                reward += C.R_NONMOTION_PENALTY  # anti-spin: STOP/TURN into a wall is doubly bad
        else:
            reward = 0.0

            # 1) Distance progress toward goal (potential-based; observable via obs[36])
            reward += C.R_DIST_SCALE * (self._prev_dist - dist)

            # 2) Wall-proximity warning (observable via lidar)
            if min_scan < C.R_PROX_THRESH:
                reward += C.R_PROX_SCALE * (C.R_PROX_THRESH - min_scan)

            # 3) Time penalty (encourages reaching the goal quickly)
            reward += C.R_TIME_PENALTY

            # 4) Anti-spin motion shaping: penalize non-translating actions so the
            #    policy can't settle into a spin-in-place / freeze equilibrium.
            if is_nonmotion:
                reward += C.R_NONMOTION_PENALTY

            # Track visited cells for the cells_visited metric only — NO reward.
            # Visit history is not in the observation, so rewarding it injects
            # unobservable noise that swamps the learnable distance signal; the
            # terminal +R_GOAL is the single dominant attractor instead.
            col = int(np.clip(self._rx / C.CELL_SIZE, 0, 4))
            row = int(np.clip(self._ry / C.CELL_SIZE, 0, 4))
            self._visited_cells.add((row, col))

        if self._step_count >= C.MAX_STEPS:
            truncated = True
            info["timeout"] = True

        self._prev_dist = dist

        obs = self._observe()
        info["dist_to_goal"] = dist
        info["steps"] = self._step_count
        info["min_scan"] = min_scan
        info["cells_visited"] = len(self._visited_cells)
        info["collisions"] = self._collision_count

        return obs, float(reward), terminated, truncated, info

    def _observe(self) -> np.ndarray:
        """Frame-stacked observation: FRAME_STACK frames (oldest→newest) + last-action
        one-hot = STATE_DIM. Provides temporal memory for perceptual disambiguation."""
        frame = self._observe_single()
        if len(self._frame_stack) == 0:
            # Episode start: seed the stack with copies of the first frame so the very
            # first observation is already the full STATE_DIM width.
            for _ in range(C.FRAME_STACK):
                self._frame_stack.append(frame)
        else:
            self._frame_stack.append(frame)  # deque(maxlen) evicts the oldest frame

        action_onehot = np.zeros(C.N_ACTIONS, dtype=np.float32)
        action_onehot[self._last_action] = 1.0
        return np.concatenate(list(self._frame_stack) + [action_onehot]).astype(np.float32)

    def _observe_single(self) -> np.ndarray:
        """Build one 39-dim frame: 36 lidar + dist + cos + sin."""
        scan = self._raycast()
        lidar = scan / C.LIDAR_MAX  # normalize to [0, 1]

        # Goal in robot frame
        dx = self._gx - self._rx
        dy = self._gy - self._ry
        distance = math.sqrt(dx * dx + dy * dy)

        # cos/sin of heading-to-goal angle
        heading = np.array([math.cos(self._yaw), math.sin(self._yaw)])
        goal_vec = np.array([dx, dy])
        goal_norm = np.linalg.norm(goal_vec)
        if goal_norm > 1e-6:
            goal_vec = goal_vec / goal_norm
        cos_goal = float(np.dot(heading, goal_vec))
        sin_goal = float(np.cross(heading, goal_vec))

        # Normalize distance
        max_dist = math.sqrt(2) * 5 * C.CELL_SIZE
        norm_dist = min(distance / max_dist, 1.0)

        obs = np.zeros(C.SINGLE_OBS_DIM, dtype=np.float32)
        obs[0:36] = lidar
        obs[36] = norm_dist
        obs[37] = cos_goal
        obs[38] = sin_goal
        return obs

    def _raycast(self) -> np.ndarray:
        """Vectorized 36-ray lidar raycasting against all wall segments."""
        if self._segments is None or len(self._segments) == 0:
            return np.full(C.N_RAYS, C.LIDAR_MAX, dtype=np.float32)

        angles = np.linspace(0, 2 * math.pi, C.N_RAYS, endpoint=False) + self._yaw
        D = np.stack([np.cos(angles), np.sin(angles)], axis=1)
        P = np.array([self._rx, self._ry], dtype=np.float32)

        A = self._segments[:, :2]
        BA = self._segments[:, 2:] - A
        PA = P[np.newaxis, :] - A

        DxBA = D[:, 0:1] * BA[:, 1:2].T - D[:, 1:2] * BA[:, 0:1].T
        PAxBA = PA[:, 0] * BA[:, 1] - PA[:, 1] * BA[:, 0]
        PAxD = (PA[:, 0:1] * D[:, 1:2].T - PA[:, 1:2] * D[:, 0:1].T).T

        eps = 1e-10
        parallel = np.abs(DxBA) < eps

        with np.errstate(divide='ignore', invalid='ignore'):
            t = PAxBA[np.newaxis, :] / DxBA
            s = PAxD / DxBA

        valid = (~parallel) & (t > C.LIDAR_MIN) & (t < C.LIDAR_MAX) & (s >= 0) & (s <= 1)
        t_masked = np.where(valid, t, C.LIDAR_MAX)
        return np.min(t_masked, axis=1).astype(np.float32)

    def _check_collision(self, x: float, y: float) -> bool:
        """Check if robot circle collides with any wall segment."""
        if self._segments is None:
            return False

        A = self._segments[:, :2]
        B = self._segments[:, 2:]
        P = np.array([x, y], dtype=np.float32)

        AB = B - A
        AP = P - A
        AB_len2 = np.sum(AB * AB, axis=1)

        with np.errstate(divide='ignore', invalid='ignore'):
            t = np.clip(np.sum(AP * AB, axis=1) / np.maximum(AB_len2, 1e-10), 0, 1)

        closest = A + t[:, np.newaxis] * AB
        dist2 = np.sum((P - closest) ** 2, axis=1)
        return bool(np.any(dist2 < C.ROBOT_RADIUS ** 2))

    def _dist_to_goal(self) -> float:
        return math.sqrt((self._rx - self._gx) ** 2 + (self._ry - self._gy) ** 2)

    @staticmethod
    def _wrap(a: float) -> float:
        while a > math.pi:
            a -= 2 * math.pi
        while a < -math.pi:
            a += 2 * math.pi
        return a

    def render(self):
        if self.render_mode == "ansi" and self._maze is not None:
            print(self._maze.ascii())
            print(f"  Robot: ({self._rx:.2f}, {self._ry:.2f}) yaw={math.degrees(self._yaw):.0f}°")
            print(f"  Goal:  ({self._gx:.2f}, {self._gy:.2f}) dist={self._dist_to_goal():.2f}m")
            print(f"  Step:  {self._step_count}/{C.MAX_STEPS}")


# ── Curriculum Wrapper ───────────────────────────────────────────────────

class CurriculumMazeEnv(MazeEnv):
    """Automatic curriculum: easy → medium → hard as success improves."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._success_history = []
        self._current_level = 0
        self._level_window = 100
        self._level_threshold = 0.6

    def reset(self, **kwargs):
        if len(self._success_history) >= self._level_window:
            recent = self._success_history[-self._level_window:]
            success_rate = sum(recent) / len(recent)
            if success_rate >= self._level_threshold and self._current_level < 2:
                self._current_level += 1
                self._success_history.clear()
                print(f"  📈 Curriculum advanced to level {self._current_level}")
        self._curriculum = self._current_level
        return super().reset(**kwargs)

    def step(self, action):
        obs, reward, terminated, truncated, info = super().step(action)
        if terminated or truncated:
            self._success_history.append(info.get("success", False))
        return obs, reward, terminated, truncated, info


if __name__ == "__main__":
    env = MazeEnv(render_mode="ansi", fixed_seed=42)
    obs, info = env.reset()
    env.render()
    print(f"\nObs shape: {obs.shape} (STATE_DIM={C.STATE_DIM})")
    print(f"Lidar (first 12): {obs[:12]}")
    print(f"Goal dist={obs[36]:.3f}, cos={obs[37]:.3f}, sin={obs[38]:.3f}")
    print(f"Action space: {env.action_space} ({C.N_ACTIONS} discrete)")
    print(f"Actions: {C.ACTION_NAMES}")

    total_r = 0.0
    for i in range(100):
        action = env.action_space.sample()
        obs, reward, term, trunc, info = env.step(action)
        total_r += reward
        if term or trunc:
            print(f"\nEpisode ended at step {i+1}: reward={total_r:.1f} info={info}")
            break
    else:
        print(f"\n100 steps done, reward={total_r:.1f}")
