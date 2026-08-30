#!/usr/bin/env python3
"""
occ_map.py — Lightweight 2D occupancy-grid mapper built from LiDAR + pose.

Each env step the robot's lidar rays are traced into a fixed-resolution grid:
cells along every ray are marked FREE and the ray endpoint (when the beam hit
something inside the sensor range) is marked OCCUPIED. The maze walls are the
obstacles being mapped, exactly like a hand-driven SLAM scan, but without a
full SLAM stack: the pose comes from the Gazebo PosePublisher (ground-truth
odometry), so there is no drift to correct.

Outputs (all Gazebo-free, unit-testable offline):
  * coverage()      — fraction of the maze's free cells observed so far
  * occupancy_array — (-1 unknown / 0 free / 100 occupied) grid
  * save_png        — ROS-style map image (free=white, occupied=black,
                      unknown=gray)
"""

from __future__ import annotations

import math
from typing import Iterable, Optional, Sequence, Tuple

import numpy as np

from rl_training.maze_field import Wall


def _bresenham_cells(x0: float, y0: float, x1: float, y1: float,
                     res: float, ox: float, oy: float, n: int,
                     ) -> Iterable[Tuple[int, int]]:
    """Grid-cell indices (ix, iy) sampled every ``res`` metres along a segment."""
    dist = math.hypot(x1 - x0, y1 - y0)
    steps = max(1, int(dist / res))
    for s in range(steps):  # exclude the far endpoint
        t = s / steps
        x = x0 + t * (x1 - x0)
        y = y0 + t * (y1 - y0)
        ix = int((x - ox) / res)
        iy = int((y - oy) / res)
        if 0 <= ix < n and 0 <= iy < n:
            yield ix, iy


class OccupancyGridMapper:
    """Traced-lidar occupancy grid over the maze plus a small outer margin.

    Walls may have arbitrary yaw — each is an oriented bounding box (OBB)
    from :mod:`rl_training.maze_field`.
    """

    def __init__(self, origin_xy: Tuple[float, float], extent: float,
                 walls: Sequence[Wall], res: float = 0.05,
                 margin: float = 0.10, max_range: float = 3.0,
                 inner_rect: Optional[Tuple[float, float, float, float]] = None
                 ) -> None:
        self.res = float(res)
        self.max_range = float(max_range)
        self.ox = origin_xy[0] - margin
        self.oy = origin_xy[1] - margin
        self.n = int(math.ceil((extent + 2.0 * margin) / self.res))

        # Inner-maze mask: cell centers strictly inside the maze rectangle.
        # Default (legacy): the square [origin, origin + extent] block. When
        # inner_rect=(xmin, ymin, xmax, ymax) is given (world frame), the mask
        # is confined to that rectangle instead — used when the square grid
        # covers a rectangular maze whose shorter side is smaller than extent.
        cx = self.ox + (np.arange(self.n) + 0.5) * self.res
        cy = self.oy + (np.arange(self.n) + 0.5) * self.res
        if inner_rect is None:
            inner_x = (cx > origin_xy[0]) & (cx < origin_xy[0] + extent)
            inner_y = (cy > origin_xy[1]) & (cy < origin_xy[1] + extent)
        else:
            inner_x = (cx > inner_rect[0]) & (cx < inner_rect[2])
            inner_y = (cy > inner_rect[1]) & (cy < inner_rect[3])
        self._inner = inner_x[None, :] & inner_y[:, None]

        # Expected free space: maze interior minus the wall footprint
        # (rasterized with no inflation — this is what the map should discover).
        # Walls may have arbitrary yaw (OBB), handled by Wall.contains_array.
        blocked = np.zeros((self.n, self.n), dtype=bool)
        gx, gy = np.meshgrid(cx, cy)
        for w in walls:
            blocked |= w.contains_array(gx, gy)
        self._expected_free = int((~blocked & self._inner).sum())
        if self._expected_free <= 0:
            raise ValueError("expected free space is empty — check walls/extent")

        self.free = np.zeros((self.n, self.n), dtype=bool)
        self.occ = np.zeros((self.n, self.n), dtype=bool)

    # ── State ────────────────────────────────────────────────────────

    def reset(self) -> None:
        self.free[:] = False
        self.occ[:] = False

    @property
    def expected_free(self) -> int:
        """Number of grid cells the maze's free space should occupy."""
        return self._expected_free

    def update(self, x: float, y: float, yaw: float,
               ranges: Sequence[float],
               ray_angles: Optional[Sequence[float]] = None) -> None:
        """Trace one scan: ``ranges`` measured at body-frame ``ray_angles``.

        ``ray_angles`` defaults to the uniform 36-ray 360° pattern used by
        this robot's lidar.
        """
        if ray_angles is None:
            ray_angles = [i * (2.0 * math.pi / len(ranges))
                          for i in range(len(ranges))]
        n, res = self.n, self.res
        free, occ = self.free, self.occ
        for r, a in zip(ranges, ray_angles):
            th = yaw + float(a)
            hit = r < self.max_range - 1e-3
            d = float(r) if hit else self.max_range
            ex = x + d * math.cos(th)
            ey = y + d * math.sin(th)
            for ix, iy in _bresenham_cells(x, y, ex, ey, res, self.ox, self.oy, n):
                free[iy, ix] = True
            if hit:
                eix = int((ex - self.ox) / res)
                eiy = int((ey - self.oy) / res)
                if 0 <= eix < n and 0 <= eiy < n:
                    occ[eiy, eix] = True

    def coverage(self) -> float:
        """Observed free cells / expected free cells (capped at 1.0)."""
        observed = int((self.free & self._inner).sum())
        return min(1.0, observed / self._expected_free)

    # ── Outputs ──────────────────────────────────────────────────────

    def occupancy_array(self) -> np.ndarray:
        """int8 grid: -1 unknown, 0 free, 100 occupied ([iy, ix] indexing)."""
        grid = np.full((self.n, self.n), -1, dtype=np.int8)
        grid[self.free & ~self.occ] = 0
        grid[self.occ] = 100
        return grid

    def save_png(self, path) -> None:
        """Write a ROS-style map image (row 0 = max y, like map_server)."""
        from PIL import Image

        arr = self.occupancy_array().astype(np.uint8)
        arr[arr == -1] = 205   # unknown → gray
        arr[arr == 0] = 254    # free → white
        arr[arr == 100] = 0    # occupied → black
        img = Image.fromarray(np.flipud(arr), mode="L")
        path = str(path)
        if not path.endswith(".png"):
            path += ".png"
        img.save(path)
