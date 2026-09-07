#!/usr/bin/env python3
"""
zone_field.py — one geodesic distance field per watershed zone (v8).

``MazeDistanceField`` answers "how far to the exit"; this answers "how far to
zone z" for every z, which is what the EXPLORE phase needs. The 25 fields are
built on exactly the same raster convention as the exit field (origin at the
``spec.bounds`` minimum, cell centres at ``origin + (i + 0.5) · res``,
occupancy = centre inside a wall inflated by the robot radius) so distances
from the two are directly comparable.

Each field is a MULTI-SOURCE Dijkstra whose sources are every raster cell
carrying that zone's label, not the zone centroid. A centroid can sit inside
the inflation band of a wall that clips the zone, which is what produced the
false "maze impossible" verdict recorded in the project notes; seeding the
whole label region cannot. Blocked cells are legal sources (their zone is
still reachable, just not stand-on-able) but are never relaxation TARGETS, so
a field cannot leak through a wall: the maze walls are 0.03 m thick against a
0.10 m inflation, so the band on the far side of any wall is always blocked.
"""

import heapq
import math
from typing import AbstractSet, Tuple

import numpy as np

from rl_training import config as C

_NEIGHBOURS = ((1, 0), (-1, 0), (0, 1), (0, -1),
               (1, 1), (1, -1), (-1, 1), (-1, -1))


class ZoneDistanceFields:
    """Geodesic distance (m) from any point to each of ``n_zones`` zones."""

    def __init__(self, origin_xy: Tuple[float, float], res: float,
                 blocked: np.ndarray, dist: np.ndarray) -> None:
        self.origin = (float(origin_xy[0]), float(origin_xy[1]))
        self.res = float(res)
        self.blocked = blocked                      # (ny, nx) bool
        self.dist = dist                            # (n_zones, ny, nx) float32
        self.n_zones, self.ny, self.nx = dist.shape

    # ── Construction ─────────────────────────────────────────────────

    @classmethod
    def from_spec(cls, spec, resolution: float = C.EXPL_ZONE_FIELD_RES,
                  inflate: float = C.ROBOT_RADIUS,
                  n_zones: int = C.EXPL_N_ZONES) -> "ZoneDistanceFields":
        """Build all zone fields for a PLACED ``MazeSpec`` (world frame)."""
        ox, oy = spec.placement_offset
        bx0, by0, bx1, by1 = spec.bounds
        origin = (bx0 + ox, by0 + oy)
        nx = int(round((bx1 - bx0) / resolution))
        ny = int(round((by1 - by0) / resolution))

        xs = origin[0] + (np.arange(nx) + 0.5) * resolution
        ys = origin[1] + (np.arange(ny) + 0.5) * resolution
        gx, gy = np.meshgrid(xs, ys)

        blocked = np.zeros((ny, nx), dtype=bool)
        for w in spec.walls_world:
            blocked |= w.contains_array(gx, gy, inflate)

        labels = cls._sample_labels(spec, gx, gy, nx, ny)
        dist = cls._solve(labels, blocked, resolution, n_zones)
        return cls(origin, resolution, blocked, dist)

    @staticmethod
    def _sample_labels(spec, gx: np.ndarray, gy: np.ndarray,
                       nx: int, ny: int) -> np.ndarray:
        """Zone label of every raster cell, sampled like ``zone_label_at_world``.

        The label raster has its OWN resolution and a corner (round) sampling
        convention; the field raster uses cell centres. Sampling rather than
        assuming the two grids coincide keeps this correct if ``zone_res`` and
        ``EXPL_ZONE_FIELD_RES`` ever diverge.
        """
        ox, oy = spec.placement_offset
        bx0, by0 = spec.bounds[0], spec.bounds[1]
        lny, lnx = spec.zone_labels.shape
        ix = np.rint((gx - ox - bx0) / spec.zone_res).astype(np.int64)
        iy = np.rint((gy - oy - by0) / spec.zone_res).astype(np.int64)
        inside = (ix >= 0) & (ix < lnx) & (iy >= 0) & (iy < lny)
        out = np.full((ny, nx), -1, dtype=np.int16)
        out[inside] = spec.zone_labels[iy[inside], ix[inside]]
        return out

    @staticmethod
    def _solve(labels: np.ndarray, blocked: np.ndarray, res: float,
               n_zones: int) -> np.ndarray:
        ny, nx = labels.shape
        dist = np.full((n_zones, ny, nx), np.inf, dtype=np.float32)
        diag = math.sqrt(2.0) * res
        for z in range(n_zones):
            # Relax in float64 and only narrow to float32 for STORAGE: a
            # float32 array rounds the tentative distance on write, and a
            # value that rounds DOWN makes the settled node fail its own
            # `dd > d` staleness check on pop, so its whole subtree is never
            # expanded and the field stops mid-maze.
            d = np.full((ny, nx), np.inf, dtype=np.float64)
            src_y, src_x = np.nonzero(labels == z)
            if src_y.size == 0:
                continue                      # zone absent from this raster
            d[src_y, src_x] = 0.0
            pq = [(0.0, int(x), int(y)) for x, y in zip(src_x, src_y)]
            heapq.heapify(pq)
            while pq:
                dd, ix, iy = heapq.heappop(pq)
                if dd > d[iy, ix]:
                    continue
                for dx, dy in _NEIGHBOURS:
                    jx, jy = ix + dx, iy + dy
                    if not (0 <= jx < nx and 0 <= jy < ny):
                        continue
                    if blocked[jy, jx]:
                        continue              # never relax INTO a wall
                    nd = dd + (diag if dx and dy else res)
                    if nd < d[jy, jx]:
                        d[jy, jx] = nd
                        heapq.heappush(pq, (nd, jx, jy))
            dist[z] = d
        return dist

    # ── Queries ──────────────────────────────────────────────────────

    def _to_index(self, x: float, y: float) -> Tuple[int, int]:
        ix = int(np.clip((x - self.origin[0]) / self.res, 0, self.nx - 1))
        iy = int(np.clip((y - self.origin[1]) / self.res, 0, self.ny - 1))
        return ix, iy

    def distance_to(self, zone: int, x: float, y: float) -> float:
        """Geodesic distance from (x, y) to ``zone``; inf when disconnected.

        A robot momentarily overlapping the inflation band reads an infinite
        cell, so — exactly like ``MazeDistanceField.distance`` — the lookup
        falls back to the nearest finite cell in a growing window rather than
        reporting a spurious inf that would zero the shaping term.
        """
        if not (0 <= zone < self.n_zones):
            return math.inf
        field = self.dist[zone]
        ix, iy = self._to_index(x, y)
        if math.isfinite(field[iy, ix]):
            return float(field[iy, ix])
        for r in range(1, max(self.nx, self.ny)):
            x0, x1 = max(0, ix - r), min(self.nx, ix + r + 1)
            y0, y1 = max(0, iy - r), min(self.ny, iy + r + 1)
            window = field[y0:y1, x0:x1]
            finite = window[np.isfinite(window)]
            if finite.size:
                return float(finite.min())
        return math.inf

    def nearest(self, x: float, y: float,
                unvisited: AbstractSet[int]) -> Tuple[int, float]:
        """Closest zone in ``unvisited`` — ``(-1, inf)`` when the set is empty."""
        best, best_d = -1, math.inf
        for z in unvisited:
            d = self.distance_to(z, x, y)
            if d < best_d:
                best, best_d = int(z), d
        return best, best_d

    def descent_direction(self, zone: int, x: float, y: float,
                          radius: int = C.EXPL_ZONE_DESCENT_R
                          ) -> Tuple[float, float]:
        """Unit heading that reduces the distance to ``zone`` fastest."""
        if not (0 <= zone < self.n_zones):
            return (0.0, 0.0)
        return descent_direction(self.dist[zone], self.origin, self.res,
                                 x, y, radius)


def descent_direction(field: np.ndarray, origin_xy: Tuple[float, float],
                      res: float, x: float, y: float,
                      radius: int = C.EXPL_ZONE_DESCENT_R
                      ) -> Tuple[float, float]:
    """Unit heading down a geodesic field, from a ``radius``-cell search.

    Candidates are the cells on the OUTER rim of the disc: driving toward a
    cell nearer than one step would overshoot it, and the rim is far enough
    out (0.15 m by default) to clear the 0.10 m inflation band that a
    wall-hugging robot sits in — inside that band every cell is infinite and
    the search would have no opinion at all.

    Returns (0, 0) when no rim cell has a finite distance; the caller must
    read that as "no opinion", never as "drive toward the origin".

    Shared by the per-zone fields and the exit field, which are rasterized on
    the identical grid convention (``origin + (i + 0.5)·res``).
    """
    ny, nx = field.shape
    ix = int(np.clip((x - origin_xy[0]) / res, 0, nx - 1))
    iy = int(np.clip((y - origin_xy[1]) / res, 0, ny - 1))
    step = radius * res
    best, best_d = None, math.inf
    for jy in range(max(0, iy - radius), min(ny, iy + radius + 1)):
        cy = origin_xy[1] + (jy + 0.5) * res
        for jx in range(max(0, ix - radius), min(nx, ix + radius + 1)):
            cx = origin_xy[0] + (jx + 0.5) * res
            if math.hypot(cx - x, cy - y) < step:
                continue                      # inside the rim: skip
            d = field[jy, jx]
            if d < best_d:
                best, best_d = (cx, cy), float(d)
    if best is None or not math.isfinite(best_d):
        return (0.0, 0.0)
    dx, dy = best[0] - x, best[1] - y
    norm = math.hypot(dx, dy)
    if norm < 1e-9:
        return (0.0, 0.0)
    return (dx / norm, dy / norm)
