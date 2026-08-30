#!/usr/bin/env python3
"""
maze_field.py — Geodesic distance field for reward shaping (training-only).

Parses wall boxes from the maze world SDF, rasterizes them (inflated by the
robot radius) onto a fine occupancy grid, then runs Dijkstra from the goal to
obtain a geodesic distance-to-goal field. Euclidean distance fights corridors
in a maze (moving *away* from the goal is often required); geodesic shaping
rewards true progress along the maze solution.

Used only by the training environment for reward/termination — never fed to
the policy network (policy observation stays LiDAR-only).
"""

import heapq
import math
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence, Tuple

import numpy as np

from rl_training import config as C


@dataclass(frozen=True)
class Wall:
    """Oriented bounding-box (OBB) wall segment in world coordinates.

    The wall's long axis points along local X, rotated by ``yaw`` about the
    wall center. ``yaw`` is normalized to (-pi/2, pi/2] (an OBB is symmetric
    under a pi rotation) and ``half_length >= half_thickness`` always holds.
    """
    cx: float
    cy: float
    half_length: float      # along the wall's local long axis
    half_thickness: float   # across the wall
    yaw: float              # long-axis orientation (rad), in (-pi/2, pi/2]

    @property
    def along_x(self) -> bool:
        """Legacy compat: True when the long axis is closer to world X."""
        return abs(math.sin(self.yaw)) <= abs(math.cos(self.yaw))

    def contains(self, x: float, y: float, inflate: float = 0.0) -> bool:
        """Point-in-OBB test with the box grown by ``inflate`` on all sides."""
        dx, dy = x - self.cx, y - self.cy
        c, s = math.cos(self.yaw), math.sin(self.yaw)
        lx = c * dx + s * dy     # along the wall
        ly = -s * dx + c * dy    # across the wall
        return (abs(lx) <= self.half_length + inflate
                and abs(ly) <= self.half_thickness + inflate)

    def contains_array(self, xs, ys, inflate: float = 0.0) -> np.ndarray:
        """Vectorized :meth:`contains` over numpy arrays of x/y points."""
        xs = np.asarray(xs, dtype=np.float64)
        ys = np.asarray(ys, dtype=np.float64)
        dx, dy = xs - self.cx, ys - self.cy
        c, s = math.cos(self.yaw), math.sin(self.yaw)
        lx = c * dx + s * dy
        ly = -s * dx + c * dy
        return ((np.abs(lx) <= self.half_length + inflate)
                & (np.abs(ly) <= self.half_thickness + inflate))

    def translated(self, dx: float, dy: float) -> "Wall":
        """Copy of this wall shifted by (dx, dy) in world frame."""
        return Wall(self.cx + dx, self.cy + dy,
                    self.half_length, self.half_thickness, self.yaw)


def wall_from_segment(x0: float, y0: float, x1: float, y1: float,
                      thickness: float) -> Wall:
    """OBB wall covering the segment [x0,y0]-[x1,y1] with the given thickness.

    NOTE: the box spans exactly the segment endpoints (callers pre-extend
    segments to seal junctions).
    """
    cx, cy = (x0 + x1) / 2.0, (y0 + y1) / 2.0
    length = math.hypot(x1 - x0, y1 - y0)
    yaw = math.atan2(y1 - y0, x1 - x0)
    # normalize to (-pi/2, pi/2]
    if yaw > math.pi / 2.0:
        yaw -= math.pi
    elif yaw <= -math.pi / 2.0:
        yaw += math.pi
    return Wall(cx, cy, length / 2.0, thickness / 2.0, yaw)


def parse_walls(world_sdf: Path) -> list:
    """Extract thin box wall segments from a Gazebo world SDF.

    A model counts as a wall when its collision geometry is a box whose
    smaller horizontal dimension is < 0.1 m (walls here are 0.03 m thick).
    Arbitrary yaw is supported: each wall is returned as an oriented
    bounding box (OBB) whose long local-X axis is rotated by the pose yaw.
    """
    world_sdf = Path(world_sdf)
    if not world_sdf.is_file():
        raise FileNotFoundError(f"world SDF not found: {world_sdf}")

    root = ET.parse(world_sdf).getroot()
    walls = []
    for model in root.iter("model"):
        pose_el = model.find("pose")
        box = model.find(".//collision/geometry/box/size")
        if pose_el is None or box is None:
            continue
        sx, sy, _sz = (float(v) for v in box.text.split()[:3])
        if min(sx, sy) >= 0.1:  # not a thin wall (e.g. ground plane)
            continue
        px, py, _pz, _r, _p, yaw = (float(v) for v in pose_el.text.split())

        # The box's long axis is model-local X when sx >= sy, else local Y.
        if sx >= sy:
            half_length, half_thickness, theta = sx / 2.0, sy / 2.0, yaw
        else:
            half_length, half_thickness, theta = sy / 2.0, sx / 2.0, yaw + math.pi / 2.0
        # normalize theta to (-pi/2, pi/2]
        if theta > math.pi / 2.0:
            theta -= math.pi
        elif theta <= -math.pi / 2.0:
            theta += math.pi
        walls.append(Wall(cx=px, cy=py, half_length=half_length,
                          half_thickness=half_thickness, yaw=theta))
    if not walls:
        raise ValueError(f"no wall boxes found in {world_sdf}")
    return walls


class MazeDistanceField:
    """Geodesic distance-to-goal (meters) over the free space of the maze."""

    def __init__(self, world_sdf: Path,
                 goal_xy: Tuple[float, float],
                 origin_xy: Tuple[float, float] = (2.0, -4.5),
                 extent: float = 5 * C.CELL_SIZE,
                 resolution: float = 0.05,
                 inflate: float = C.ROBOT_RADIUS,
                 extent_xy: Optional[Tuple[float, float]] = None,
                 require_connected: bool = True):
        self._build(parse_walls(world_sdf), goal_xy, origin_xy,
                    extent, resolution, inflate, extent_xy, require_connected)

    @classmethod
    def from_walls(cls, walls: Sequence[Wall],
                   goal_xy: Tuple[float, float],
                   origin_xy: Tuple[float, float] = (2.0, -4.5),
                   extent: float = 5 * C.CELL_SIZE,
                   resolution: float = 0.05,
                   inflate: float = C.ROBOT_RADIUS,
                   extent_xy: Optional[Tuple[float, float]] = None,
                   require_connected: bool = True) -> "MazeDistanceField":
        """Build the geodesic field from pre-parsed OBB walls.

        ``extent_xy=(w, h)`` selects a rectangular (ny, nx) raster; when it is
        omitted the legacy square ``extent``-sized grid is built. With
        ``require_connected=False`` free cells unreachable from the goal are
        allowed (they fall back to the nearest finite distance in
        :meth:`distance`); the goal-inside-wall check still always raises.
        """
        self = object.__new__(cls)
        self._build(walls, goal_xy, origin_xy, extent, resolution, inflate,
                    extent_xy, require_connected)
        return self

    def _build(self, walls, goal_xy, origin_xy, extent, resolution, inflate,
               extent_xy, require_connected):
        self.origin = origin_xy
        self.res = resolution
        if extent_xy is None:
            self.nx = self.ny = int(round(extent / resolution))  # square grid
        else:
            self.nx = int(round(extent_xy[0] / resolution))
            self.ny = int(round(extent_xy[1] / resolution))
        nx, ny = self.nx, self.ny

        # Occupancy: cell center inside any inflated wall → blocked
        xs = self.origin[0] + (np.arange(nx) + 0.5) * self.res
        ys = self.origin[1] + (np.arange(ny) + 0.5) * self.res
        gx, gy = np.meshgrid(xs, ys)
        self.blocked = np.zeros((ny, nx), dtype=bool)
        for w in walls:
            self.blocked |= w.contains_array(gx, gy, inflate)

        # Dijkstra (8-connected, metric costs) from the goal cell
        self.dist = np.full((ny, nx), np.inf, dtype=np.float64)
        gix, giy = self._to_index(*goal_xy)
        if self.blocked[giy, gix]:
            raise ValueError(f"goal {goal_xy} lies inside an inflated wall")
        self.dist[giy, gix] = 0.0
        pq = [(0.0, gix, giy)]
        diag = math.sqrt(2.0) * self.res
        while pq:
            d, ix, iy = heapq.heappop(pq)
            if d > self.dist[iy, ix]:
                continue
            for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1),
                           (1, 1), (1, -1), (-1, 1), (-1, -1)):
                jx, jy = ix + dx, iy + dy
                if not (0 <= jx < nx and 0 <= jy < ny):
                    continue
                if self.blocked[jy, jx]:
                    continue
                nd = d + (diag if dx and dy else self.res)
                if nd < self.dist[jy, jx]:
                    self.dist[jy, jx] = nd
                    heapq.heappush(pq, (nd, jx, jy))

        if require_connected and not np.isfinite(self.dist[~self.blocked]).all():
            raise ValueError(
                "free space is not fully connected to the goal — "
                "check wall parsing / inflation radius")

        self.max_dist = float(self.dist[~self.blocked].max())

    @property
    def n(self) -> int:
        """Legacy compat: cells per side of a square grid (max of nx/ny)."""
        return max(self.nx, self.ny)

    def _to_index(self, x: float, y: float) -> Tuple[int, int]:
        ix = int(np.clip((x - self.origin[0]) / self.res, 0, self.nx - 1))
        iy = int(np.clip((y - self.origin[1]) / self.res, 0, self.ny - 1))
        return ix, iy

    def distance(self, x: float, y: float) -> float:
        """Geodesic distance to goal (m). Points inside walls — or free cells
        disconnected from the goal — fall back to the nearest finite cell
        within a small search window."""
        ix, iy = self._to_index(x, y)
        if not self.blocked[iy, ix] and np.isfinite(self.dist[iy, ix]):
            return float(self.dist[iy, ix])
        # nearest-free fallback (robot briefly overlapping inflation zone)
        for r in range(1, self.n):
            x0, x1 = max(0, ix - r), min(self.nx, ix + r + 1)
            y0, y1 = max(0, iy - r), min(self.ny, iy + r + 1)
            window = self.dist[y0:y1, x0:x1]
            finite = window[np.isfinite(window)]
            if finite.size:
                return float(finite.min())
        return self.max_dist
