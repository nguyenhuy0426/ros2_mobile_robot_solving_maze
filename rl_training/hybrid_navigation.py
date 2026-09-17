"""Map planning + a learned LiDAR/subgoal controller, independent of ROS.

Maps are supplied by the caller (SLAM in deployment, known maps in the
explicitly labelled offline diagnostic). Unknown cells are not traversable.
The safety filter is a model check, not a hardware safety guarantee.
"""
from __future__ import annotations

import heapq
import math
from dataclasses import dataclass

import numpy as np
import torch
from scipy.ndimage import distance_transform_edt
from torch import nn

from rl_training import config as C


@dataclass(frozen=True)
class HybridConfig:
    # Encloses chassis corners and wheel collision cylinders, plus 15 mm.
    radius: float = math.hypot(C.CHASSIS_L / 2, C.WHEEL_SEP / 2 + 0.01) + 0.015
    max_v: float = 0.18
    max_w: float = 1.2
    lookahead: float = 0.25
    goal_tolerance: float = 0.07
    reaction_time: float = 0.3
    braking_accel: float = 0.4


H = HybridConfig()


def wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


class GridPlanner:
    def __init__(self, occupancy, resolution, origin, radius=H.radius):
        self.grid = np.asarray(occupancy)
        self.res = float(resolution)
        self.origin = np.asarray(origin, dtype=float)
        if self.grid.ndim != 2 or self.res <= 0:
            raise ValueError("expected a 2D occupancy grid and positive resolution")
        # Occupied AND unknown cells block. Padding makes the map edge solid.
        free = np.pad(self.grid == 0, 1, constant_values=False)
        self.clearance = distance_transform_edt(free)[1:-1, 1:-1] * self.res
        # Account for the occupied cell square, not just its center.
        self.safe = self.clearance > radius + self.res * math.sqrt(2)

    def cell(self, xy):
        return tuple(np.floor((np.asarray(xy) - self.origin) / self.res).astype(int))

    def xy(self, cell):
        return self.origin + (np.asarray(cell) + 0.5) * self.res

    def valid(self, cell):
        x, y = cell
        return 0 <= y < self.safe.shape[0] and 0 <= x < self.safe.shape[1] and self.safe[y, x]

    def line_safe(self, a, b):
        n = max(2, int(math.dist(a, b) / (self.res / 3)) + 2)
        return all(self.valid(self.cell(p)) for p in np.linspace(a, b, n))

    def plan(self, start, goal):
        s, g = self.cell(start), self.cell(goal)
        if not self.valid(s) or not self.valid(g):
            return []  # No snapping across walls or hidden fallback route.
        queue, cost, parent = [(0., s)], {s: 0.}, {}
        while queue:
            _, u = heapq.heappop(queue)
            if u == g:
                cells = [u]
                while u in parent:
                    u = parent[u]
                    cells.append(u)
                return [np.asarray(start)] + [self.xy(c) for c in cells[::-1]] + [np.asarray(goal)]
            for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                v = (u[0] + dx, u[1] + dy)
                if not self.valid(v):
                    continue
                # Prefer corridor centers over scraping the inflation boundary.
                step = self.res * (1. + 0.08 / max(self.clearance[v[1], v[0]], 0.01))
                nc = cost[u] + step
                if nc < cost.get(v, math.inf):
                    cost[v], parent[v] = nc, u
                    heuristic = self.res * (abs(v[0] - g[0]) + abs(v[1] - g[1]))
                    heapq.heappush(queue, (nc + heuristic, v))
        return []

    def subgoal(self, xy, path):
        if not path:
            raise ValueError("empty path")
        nearest = int(np.argmin(np.linalg.norm(np.asarray(path) - xy, axis=1)))
        target = np.asarray(path[nearest])
        for point in path[nearest + 1:]:
            if not self.line_safe(xy, point):
                break
            target = np.asarray(point)
            if math.dist(xy, target) >= H.lookahead:
                break
        return target


def observation(scan, pose, subgoal):
    scan = np.asarray(scan, dtype=np.float32)
    if scan.shape != (C.N_RAYS,) or not np.isfinite(scan).all() or np.any(scan <= 0):
        raise ValueError("controller requires 36 valid, positive LiDAR ranges")
    dx, dy = np.asarray(subgoal) - np.asarray(pose[:2])
    bearing = wrap(math.atan2(dy, dx) - pose[2])
    return np.concatenate((np.clip(scan / C.LIDAR_MAX, 0, 1),
                           [min(math.hypot(dx, dy), 1.), bearing / math.pi])).astype(np.float32)


class LocalPolicy(nn.Module):
    """38 -> 64 -> 64 -> 2; outputs normalized forward and turn commands."""
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(C.N_RAYS + 2, 64), nn.Tanh(),
                                 nn.Linear(64, 64), nn.Tanh(), nn.Linear(64, 2))

    def forward(self, obs):
        return self.net(obs)

    @torch.inference_mode()
    def command(self, obs):
        out = self(torch.as_tensor(obs).unsqueeze(0))[0].cpu().numpy()
        return float(np.clip(out[0], 0, 1) * H.max_v), float(np.clip(out[1], -1, 1) * H.max_w)


def integrate(pose, v, w, dt):
    x, y, yaw = pose
    if abs(w) < 1e-8:
        return np.array([x + v * math.cos(yaw) * dt, y + v * math.sin(yaw) * dt, yaw])
    nyaw = yaw + w * dt
    return np.array([x + v / w * (math.sin(nyaw) - math.sin(yaw)),
                     y - v / w * (math.cos(nyaw) - math.cos(yaw)), wrap(nyaw)])


def filter_command(planner, pose, scan, v, w):
    """Veto unsafe swept disk using the map AND current LiDAR endpoints.

    Includes a braking extension at the end. Conservative circular envelope
    allows checking rotation as well as translation without corner shortcuts.
    A caller must separately stop on stale scans / localization / map data.
    """
    scan = np.asarray(scan)
    if not np.isfinite([*pose, v, w, *scan]).all() or np.any(scan <= 0):
        return 0., 0., True
    a = pose[2] + np.arange(C.N_RAYS) * 2 * np.pi / C.N_RAYS
    sensor = np.asarray(pose[:2]) + C.LIDAR_X_OFF * np.array([math.cos(pose[2]), math.sin(pose[2])])
    hits = scan < C.LIDAR_MAX - 1e-3
    points = sensor + scan[:, None] * np.column_stack((np.cos(a), np.sin(a)))
    points = points[hits]

    def safe(cv, cw):
        horizon = H.reaction_time + abs(cv) / H.braking_accel
        for t in np.linspace(0, horizon, max(12, int(horizon / 0.02) + 1)):
            future = integrate(pose, cv, cw, t)
            if not planner.valid(planner.cell(future[:2])):
                return False
            if len(points) and np.min(np.linalg.norm(points - future[:2], axis=1)) <= H.radius:
                return False
        return True

    for scale in (1., 0.5, 0.):
        if safe(v * scale, w):
            return v * scale, w, scale != 1.
    return 0., 0., True
