#!/usr/bin/env python3
"""
curriculum.py — Start-distance curriculum for the fixed-maze wheel env.

The v1 SAC run never once reached the goal (0/818), so the +200 terminal was
never sampled and the value function had nothing to bootstrap from. A
start-distance curriculum fixes reachability: begin with the start cell close
to the goal (short, easy paths that the agent can actually complete), then push
the start progressively farther as the recent success rate rises — ending at
the true ``config.START_XY``.

The curriculum ranks the maze's free cell centres by their *geodesic* distance
to the goal (reusing the env's existing ``MazeDistanceField`` — no duplicate SDF
parsing) and exposes a widening pool of candidate start cells per level.
"""

from __future__ import annotations

import math
import random
from collections import deque
from typing import Deque, List, Optional, Tuple


class StartCurriculum:
    """Level-based start-cell sampler with success-driven advancement."""

    def __init__(self, distance_field, origin_xy: Tuple[float, float],
                 goal_xy: Tuple[float, float], grid_cells: int,
                 cell_size: float, goal_radius: float,
                 success_window: int, advance_threshold: float,
                 final_start_xy: Optional[Tuple[float, float]] = None,
                 seed: Optional[int] = None) -> None:
        self._goal_xy = goal_xy
        self._rng = random.Random(seed)
        self._success_window = max(1, int(success_window))
        self._advance_threshold = float(advance_threshold)
        self._recent: Deque[bool] = deque(maxlen=self._success_window)

        # Rank reachable free cell centres by geodesic distance to the goal.
        candidates: List[Tuple[float, Tuple[float, float]]] = []
        half = cell_size / 2.0
        for row in range(grid_cells):
            for col in range(grid_cells):
                x = origin_xy[0] + half + col * cell_size
                y = origin_xy[1] + half + row * cell_size
                if math.dist((x, y), goal_xy) < goal_radius:
                    continue  # skip the goal cell itself
                d = distance_field.distance(x, y)
                if math.isfinite(d) and d < 1e3:
                    candidates.append((d, (x, y)))
        candidates.sort(key=lambda t: t[0])
        self._cells: List[Tuple[float, float]] = [xy for _, xy in candidates]

        # Guarantee the true start is the final, hardest cell.
        if final_start_xy is not None:
            self._cells = [c for c in self._cells
                           if math.dist(c, final_start_xy) > 1e-6]
            self._cells.append(final_start_xy)

        if not self._cells:
            raise ValueError("curriculum found no reachable start cells")

        self._level = 0  # pool = self._cells[: level + 1]

    @property
    def level(self) -> int:
        return self._level

    @property
    def max_level(self) -> int:
        return len(self._cells) - 1

    def sample_start(self) -> Tuple[float, float]:
        """Uniformly sample a start cell from the current widening pool."""
        pool = self._cells[: self._level + 1]
        return self._rng.choice(pool)

    def record_outcome(self, success: bool) -> bool:
        """Log an episode outcome; advance a level if the window clears the bar.

        Returns True if the level advanced on this call.
        """
        self._recent.append(bool(success))
        if (len(self._recent) >= self._success_window
                and self._level < self.max_level
                and (sum(self._recent) / len(self._recent))
                >= self._advance_threshold):
            self._level += 1
            self._recent.clear()
            return True
        return False
