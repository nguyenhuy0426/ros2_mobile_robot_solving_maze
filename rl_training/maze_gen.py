#!/usr/bin/env python3
"""
maze_gen.py — Random 5×5 maze generator using recursive backtracker (DFS).

Generates perfect mazes (exactly one path between any two cells).
Output: wall arrays suitable for raycasting and SDF export.

Coordinate system (local, meters):
  Cell (r,c): center at (c*CELL + CELL/2, r*CELL + CELL/2)
  Maze spans [0, COLS*CELL] × [0, ROWS*CELL]
  Row 0 = bottom, Col 0 = left
"""

import random
import numpy as np
from typing import Tuple, List, Optional


ROWS, COLS = 5, 5
CELL = 0.50  # meters per cell


class Maze:
    """
    5×5 maze represented by horizontal and vertical wall arrays.

    h_walls[r][c]: horizontal wall on TOP edge of cell (r,c)
      - r ∈ [0, ROWS]:  r=0 is bottom boundary, r=ROWS is top boundary
      - c ∈ [0, COLS-1]

    v_walls[r][c]: vertical wall on LEFT edge of cell (r,c)
      - r ∈ [0, ROWS-1]
      - c ∈ [0, COLS]:  c=0 is left boundary, c=COLS is right boundary
    """

    def __init__(self, rows: int = ROWS, cols: int = COLS, cell: float = CELL):
        self.rows = rows
        self.cols = cols
        self.cell = cell

        # All walls present initially
        self.h_walls = [[True] * cols for _ in range(rows + 1)]
        self.v_walls = [[True] * (cols + 1) for _ in range(rows)]

        self.start: Tuple[int, int] = (0, 0)
        self.goal: Tuple[int, int] = (rows - 1, cols - 1)

        # Precomputed wall segments (filled after generate)
        self._segments: Optional[np.ndarray] = None

    def generate(self, seed: Optional[int] = None,
                 start: Optional[Tuple[int, int]] = None,
                 goal: Optional[Tuple[int, int]] = None):
        """Generate random perfect maze using recursive backtracker."""
        rng = random.Random(seed)

        if start is not None:
            self.start = start
        if goal is not None:
            self.goal = goal

        visited = [[False] * self.cols for _ in range(self.rows)]
        stack = [self.start]
        visited[self.start[0]][self.start[1]] = True

        while stack:
            r, c = stack[-1]
            neighbors = []
            for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                nr, nc = r + dr, c + dc
                if 0 <= nr < self.rows and 0 <= nc < self.cols and not visited[nr][nc]:
                    neighbors.append((nr, nc, dr, dc))

            if neighbors:
                nr, nc, dr, dc = rng.choice(neighbors)
                # Remove wall between (r,c) and (nr,nc)
                if dr == 1:   # neighbor is above
                    self.h_walls[r + 1][c] = False
                elif dr == -1:  # neighbor is below
                    self.h_walls[r][c] = False
                elif dc == 1:  # neighbor is right
                    self.v_walls[r][c + 1] = False
                elif dc == -1:  # neighbor is left
                    self.v_walls[r][c] = False

                visited[nr][nc] = True
                stack.append((nr, nc))
            else:
                stack.pop()

        self._segments = None  # invalidate cache
        return self

    def wall_segments(self) -> np.ndarray:
        """
        Return all wall segments as numpy array of shape (N, 4).
        Each row: [x1, y1, x2, y2] — a line segment in meters.
        """
        if self._segments is not None:
            return self._segments

        segs = []
        c = self.cell

        # Horizontal walls
        for r in range(self.rows + 1):
            for col in range(self.cols):
                if self.h_walls[r][col]:
                    segs.append([col * c, r * c, (col + 1) * c, r * c])

        # Vertical walls
        for row in range(self.rows):
            for col in range(self.cols + 1):
                if self.v_walls[row][col]:
                    segs.append([col * c, row * c, col * c, (row + 1) * c])

        self._segments = np.array(segs, dtype=np.float32)
        return self._segments

    def cell_center(self, r: int, c: int) -> Tuple[float, float]:
        """World coordinates of cell center."""
        return (c * self.cell + self.cell / 2, r * self.cell + self.cell / 2)

    def start_pos(self) -> Tuple[float, float]:
        return self.cell_center(*self.start)

    def goal_pos(self) -> Tuple[float, float]:
        return self.cell_center(*self.goal)

    def solve_bfs(self) -> List[Tuple[int, int]]:
        """BFS shortest path from start to goal. Returns list of cells."""
        from collections import deque
        q = deque([(self.start, [self.start])])
        visited = {self.start}

        while q:
            (r, c), path = q.popleft()
            if (r, c) == self.goal:
                return path

            for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                nr, nc = r + dr, c + dc
                if not (0 <= nr < self.rows and 0 <= nc < self.cols):
                    continue
                if (nr, nc) in visited:
                    continue
                # Check wall between (r,c) and (nr,nc)
                if dr == 1 and self.h_walls[r + 1][c]:
                    continue
                if dr == -1 and self.h_walls[r][c]:
                    continue
                if dc == 1 and self.v_walls[r][c + 1]:
                    continue
                if dc == -1 and self.v_walls[r][c]:
                    continue

                visited.add((nr, nc))
                q.append(((nr, nc), path + [(nr, nc)]))

        return []  # no path (shouldn't happen for perfect maze)

    def path_length(self) -> float:
        """Shortest path length in meters."""
        path = self.solve_bfs()
        if len(path) < 2:
            return 0.0
        dist = 0.0
        for i in range(len(path) - 1):
            c1 = self.cell_center(*path[i])
            c2 = self.cell_center(*path[i + 1])
            dist += abs(c1[0] - c2[0]) + abs(c1[1] - c2[1])
        return dist

    def ascii(self) -> str:
        """ASCII representation for debugging."""
        lines = []
        # Top boundary
        top = "+"
        for c in range(self.cols):
            top += "--+" if self.h_walls[self.rows][c] else "  +"
        lines.append(top)

        for r in range(self.rows - 1, -1, -1):
            # Cell row
            row = "|" if self.v_walls[r][0] else " "
            for c in range(self.cols):
                if (r, c) == self.start:
                    cell = "S "
                elif (r, c) == self.goal:
                    cell = "G "
                else:
                    cell = "  "
                wall = "|" if self.v_walls[r][c + 1] else " "
                row += cell + wall
            lines.append(row)

            # Bottom boundary of this row
            bot = "+"
            for c in range(self.cols):
                bot += "--+" if self.h_walls[r][c] else "  +"
            lines.append(bot)

        return "\n".join(lines)


def random_start_goal(rows: int = ROWS, cols: int = COLS,
                      rng: Optional[random.Random] = None,
                      min_manhattan: int = 4) -> Tuple[Tuple[int, int], Tuple[int, int]]:
    """Generate random start/goal with minimum Manhattan distance."""
    if rng is None:
        rng = random.Random()

    while True:
        sr, sc = rng.randint(0, rows - 1), rng.randint(0, cols - 1)
        gr, gc = rng.randint(0, rows - 1), rng.randint(0, cols - 1)
        if abs(sr - gr) + abs(sc - gc) >= min_manhattan:
            return (sr, sc), (gr, gc)


def generate_batch(n: int, base_seed: int = 0,
                   randomize_endpoints: bool = True) -> List[Maze]:
    """Generate n random mazes."""
    mazes = []
    for i in range(n):
        rng = random.Random(base_seed + i)
        if randomize_endpoints:
            start, goal = random_start_goal(rng=rng)
        else:
            start, goal = (0, 0), (ROWS - 1, COLS - 1)
        m = Maze()
        m.generate(seed=base_seed + i, start=start, goal=goal)
        mazes.append(m)
    return mazes


if __name__ == "__main__":
    m = Maze()
    m.generate(seed=42, start=(0, 0), goal=(4, 4))
    print(m.ascii())
    print(f"\nPath length: {m.path_length():.1f}m")
    print(f"BFS path: {m.solve_bfs()}")
    print(f"Wall segments: {len(m.wall_segments())}")
