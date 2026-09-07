#!/usr/bin/env python3
"""
reward_shaping.py — Stateful reward-shaping helpers for the wheel env.

These are deliberately small, pure-Python, Gazebo-free objects so they can be
unit-tested offline and reused by any env variant without duplicating the
reward arithmetic (the env still owns the final reward formula; these only
track per-episode state and return scalar contributions).

Addresses the diagnosed failure modes of the v1 SAC run:
  * StuckTracker    → penalizes near-zero displacement (anti-stall).
  * CoverageTracker → count-based novelty bonus for visiting new maze cells
                      (dense exploration signal toward a far goal).
See rl_training/reports/last_train_analysis.md.
"""

from __future__ import annotations

import math
from collections import deque
from typing import Deque, FrozenSet, Optional, Set, Tuple

import numpy as np


class StuckTracker:
    """Detects and penalizes a robot that stops making net progress.

    Keeps a sliding window of world positions; if the straight-line
    displacement across the window falls below ``min_disp`` metres, the robot
    is considered stuck and :meth:`penalty` returns a (negative) shaping term.
    """

    def __init__(self, window: int, min_disp: float, penalty: float) -> None:
        self._window = max(2, int(window))
        self._min_disp = float(min_disp)
        self._penalty = float(penalty)
        self._positions: Deque[Tuple[float, float]] = deque(maxlen=self._window)

    def reset(self, start_xy: Tuple[float, float]) -> None:
        self._positions.clear()
        self._positions.append(start_xy)

    def update(self, xy: Tuple[float, float]) -> float:
        """Record a new position and return the shaping penalty for this step."""
        self._positions.append(xy)
        if len(self._positions) < self._window:
            return 0.0
        x0, y0 = self._positions[0]
        x1, y1 = self._positions[-1]
        disp = math.hypot(x1 - x0, y1 - y0)
        return self._penalty if disp < self._min_disp else 0.0

    @property
    def is_stuck(self) -> bool:
        if len(self._positions) < self._window:
            return False
        x0, y0 = self._positions[0]
        x1, y1 = self._positions[-1]
        return math.hypot(x1 - x0, y1 - y0) < self._min_disp


class CoverageTracker:
    """Count-based exploration bonus over a discrete maze-cell grid.

    Maps a world (x, y) to an integer cell index and awards ``bonus`` the first
    time each new cell is entered within an episode. This gives a dense,
    bounded incentive to traverse unseen corridors even before the sparse goal
    reward is ever reached.

    ``grid_cells`` bounds the tracked area to the maze: positions outside the
    ``grid_cells × grid_cells`` block rooted at ``origin_xy`` earn NO bonus.
    Without this bound a robot that leaves the maze can farm novelty bonuses
    on the infinite plane (observed: +60 reward episodes spent roaming outside).
    """

    def __init__(self, origin_xy: Tuple[float, float], cell_size: float,
                 bonus: float, grid_cells: int = 5,
                 pitch: float = None, wall_offset: float = 0.0) -> None:
        self._ox, self._oy = origin_xy
        self._cell = float(cell_size)
        # Maze cells are laid out on a (cell_size + wall thickness) pitch
        # starting wall_offset behind the origin; when pitch is omitted the
        # grid degenerates to the naive uniform layout (legacy behaviour).
        self._pitch = float(pitch) if pitch is not None else float(cell_size)
        self._offset = float(wall_offset)
        self._bonus = float(bonus)
        self._grid = int(grid_cells)
        self._visited: Set[Tuple[int, int]] = set()

    def _cell_of(self, xy: Tuple[float, float]) -> Tuple[int, int]:
        cx = int(math.floor((xy[0] - self._ox - self._offset) / self._pitch))
        cy = int(math.floor((xy[1] - self._oy - self._offset) / self._pitch))
        return cx, cy

    def reset(self, start_xy: Tuple[float, float]) -> None:
        self._visited.clear()
        self._visited.add(self._cell_of(start_xy))

    def update(self, xy: Tuple[float, float]) -> float:
        """Return ``bonus`` if ``xy`` is a newly visited in-maze cell, else 0.0."""
        cell = self._cell_of(xy)
        if not (0 <= cell[0] < self._grid and 0 <= cell[1] < self._grid):
            return 0.0  # outside the maze: never reward, never track
        if cell in self._visited:
            return 0.0
        self._visited.add(cell)
        return self._bonus

    @property
    def n_cells(self) -> int:
        return len(self._visited)


class ZoneCoverage:
    """Count-based exploration bonus over precomputed maze zones.

    Coverage is defined by a label raster from the maze registry: every free
    cell of a maze belongs to exactly one of ``n_zones`` watershed zones
    (labels >= 0; -1 marks walls / outside). A world (x, y) is converted to
    maze-local coordinates by subtracting ``offset_xy`` (world = local +
    offset), sampled into the label raster at ``res`` resolution anchored at
    ``bounds_min_xy`` (the maze-local bounds minimum), and the zone label is
    marked visited the first time it is entered — earning ``bonus`` once, like
    CoverageTracker.

    Labels < 0 never count and are never tracked, so a robot outside the maze
    or inside walls cannot farm novelty bonuses (same guarantee as
    CoverageTracker's grid-bounds check).
    """

    def __init__(self, labels, bounds_min_xy, res, offset_xy, bonus,
                 n_zones: int) -> None:
        self._labels = np.asarray(labels)
        self._x0, self._y0 = float(bounds_min_xy[0]), float(bounds_min_xy[1])
        self._res = float(res)
        self._ox, self._oy = float(offset_xy[0]), float(offset_xy[1])
        self._bonus = float(bonus)
        self._n_zones = int(n_zones)
        self._visited: Set[int] = set()

    def _label_at(self, x: float, y: float) -> int:
        """Zone label at a WORLD-frame point; -1 when outside/blocked."""
        lx, ly = x - self._ox, y - self._oy
        ny, nx = self._labels.shape
        ix = int(round((lx - self._x0) / self._res))
        iy = int(round((ly - self._y0) / self._res))
        if not (0 <= ix < nx and 0 <= iy < ny):
            return -1
        return int(self._labels[iy, ix])

    def label_at(self, x: float, y: float) -> int:
        """Public read-only zone lookup (world frame)."""
        return self._label_at(x, y)

    def reset(self, start_xy) -> None:
        self._visited.clear()
        lbl = self._label_at(start_xy[0], start_xy[1])
        if lbl >= 0:
            self._visited.add(lbl)

    def update(self, xy) -> float:
        """Return ``bonus`` the first time a new zone (label >= 0) is entered."""
        lbl = self._label_at(xy[0], xy[1])
        if lbl < 0 or lbl in self._visited:
            return 0.0
        self._visited.add(lbl)
        return self._bonus

    @property
    def n_cells(self) -> int:
        """Distinct zones visited (name kept for CoverageTracker compat)."""
        return len(self._visited)

    @property
    def n_zones(self) -> int:
        return self._n_zones

    @property
    def visited(self) -> FrozenSet[int]:
        """Zones already ticked off, as an immutable snapshot.

        The EXPLORE shaping needs the COMPLEMENT of this set to pick the
        nearest unvisited zone; handing out the live set would let a caller
        mark zones visited without paying the bonus.
        """
        return frozenset(self._visited)


class StallMonitor:
    """Hard stop for a robot that stops moving.

    While StuckTracker only penalizes near-zero displacement, a discounted
    critic can still prefer the slow penalty drip over a one-shot collision
    penalty (the v1 stall optimum resurfacing). This monitor ends the episode
    after ``limit`` consecutive ``window``-step windows with less than
    ``min_disp`` metres of net travel, making standing still exactly as bad
    as colliding — and much slower to reach than any coverage progress.
    """

    def __init__(self, window: int, min_disp: float, limit: int) -> None:
        self._window = max(2, int(window))
        self._min_disp = float(min_disp)
        self._limit = max(1, int(limit))
        self._positions: Deque[Tuple[float, float]] = deque(maxlen=self._window)
        self._consecutive = 0

    def reset(self, start_xy: Tuple[float, float]) -> None:
        self._positions.clear()
        self._positions.append(start_xy)
        self._consecutive = 0

    def update(self, xy: Tuple[float, float]) -> bool:
        """Record a position; return True when the stall limit is reached."""
        self._positions.append(xy)
        if len(self._positions) < self._window:
            return False
        x0, y0 = self._positions[0]
        x1, y1 = self._positions[-1]
        if math.hypot(x1 - x0, y1 - y0) < self._min_disp:
            self._consecutive += 1
        else:
            self._consecutive = 0
        return self._consecutive >= self._limit


class ShieldLockMonitor:
    """Flicker-robust detector for a robot the safety shield has taken over.

    The env used to count CONSECUTIVE override steps and reset on any single
    release. A staircase-diagonal corridor makes the shield's cone slack
    oscillate across its release band, so the override drops for one step every
    few steps and the consecutive counter never reached its threshold -- a
    locked robot rode the full episode timeout instead of terminating.

    This counts override steps over a SLIDING window of ``window`` steps and
    reports a lock once the window is full and at least ``frac`` of it was
    shield-driven. A brief release no longer zeroes the progress.
    """

    def __init__(self, window: int, frac: float) -> None:
        self._window = max(2, int(window))
        self._frac = float(frac)
        self._flags: Deque[bool] = deque(maxlen=self._window)

    def reset(self) -> None:
        self._flags.clear()

    def update(self, overridden: bool) -> bool:
        """Record this step's override flag; return True once it is a lock."""
        self._flags.append(bool(overridden))
        if len(self._flags) < self._window:
            return False
        return sum(self._flags) >= self._frac * self._window


def action_smoothness_penalty(action, prev_action, scale: float) -> float:
    """Negative L2 penalty on the change in action (jerk), scaled by ``scale``.

    ``prev_action`` may be ``None`` on the first step of an episode, in which
    case the penalty is zero.
    """
    if prev_action is None:
        return 0.0

    delta = np.asarray(action, dtype=float) - np.asarray(prev_action, dtype=float)
    return float(scale) * float(np.linalg.norm(delta))


def action_rate_penalty(prev: Tuple[float, float], cmd: Tuple[float, float],
                        scale: float) -> float:
    """Quadratic cost on the per-step change of the (forward, turn) command.

    The SQUARED form matters, and is why this is not
    :func:`action_smoothness_penalty` with different arguments. An L2-norm
    penalty is linear in |da|, so it charges a gentle mid-corridor correction
    at the same rate per unit as a full command reversal, and the cheapest
    response is to stop steering at all. Squaring makes small corrections
    almost free and a saw-tooth reversal expensive — "hold a heading, then
    commit to the turn", which is what the operator means by smooth.

    It is measured on the command AFTER the slew limiter, so it prices what
    the wheels actually did, and it is what makes the limiter more than a
    cosmetic filter: a policy that saturates the rate cap every step still
    weaves at the cap, and nothing in the v8 reward ever charged it for that.
    """
    df = float(cmd[0]) - float(prev[0])
    dt = float(cmd[1]) - float(prev[1])
    return float(scale) * (df * df + dt * dt)


def zone_approach_shaping(prev_d: Optional[float], d: float, scale: float,
                          retargeted: bool) -> float:
    """Potential-based shaping on the distance to the nearest UNVISITED zone.

    Same non-discounted Ng et al. form the EXIT leg uses — ``scale · (prev −
    now)`` — telescoping over an episode so oscillating in place cannot farm
    it. The one addition is ``retargeted``: when a zone is ticked off, the
    target becomes the NEXT zone out and the distance jumps upward by metres,
    which would charge a large negative on the very step that earns
    EXPL_R_CELL. That step re-baselines instead of being scored, exactly as
    ``_prev_exit_d`` is re-baselined at the EXPLORE→EXIT switch.

    Returns 0.0 whenever there is nothing meaningful to compare: the first
    step of an episode (``prev_d is None``), a retarget, or a non-finite
    distance (a zone sealed off at the inflation radius).
    """
    if retargeted or prev_d is None or d is None:
        return 0.0
    if not (math.isfinite(prev_d) and math.isfinite(d)):
        return 0.0
    return float(scale) * (float(prev_d) - float(d))
