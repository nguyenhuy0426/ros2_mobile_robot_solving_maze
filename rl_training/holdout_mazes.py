"""Deterministic, training-free perfect mazes for generalization tests."""
import math

import numpy as np

from rl_training import config as C
from rl_training.maze_field import Wall
from rl_training.maze_registry import MazeSpec, Opening


def generate_holdout(seed, size=5, cell=0.75):
    """Generate a connected acyclic grid maze from a seed.

    These specs are created directly from the seed and are never added to the
    training registry. The north border is sealed; the only opening is the
    south exit.
    """
    if size < 3:
        raise ValueError("holdout maze size must be at least 3")
    rng = np.random.default_rng(seed)
    start_col, exit_col = rng.choice(size, 2, replace=False)
    start = (start_col, size - 1)
    visited = {start}
    stack = [start]
    open_edges = set()
    while stack:
        x, y = stack[-1]
        candidates = [(nx, ny) for nx, ny in ((x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1))
                      if 0 <= nx < size and 0 <= ny < size and (nx, ny) not in visited]
        if not candidates:
            stack.pop()
            continue
        nxt = candidates[int(rng.integers(len(candidates)))]
        open_edges.add(frozenset(((x, y), nxt)))
        visited.add(nxt)
        stack.append(nxt)

    half = size * cell / 2
    thickness = C.EXPL_WALL_T
    overlap = thickness
    walls = []

    def horizontal(y, col):
        walls.append(Wall(-half + (col + .5) * cell, y, cell / 2 + overlap,
                          thickness / 2, 0.))

    def vertical(x, row):
        walls.append(Wall(x, -half + (row + .5) * cell, cell / 2 + overlap,
                          thickness / 2, math.pi / 2))

    for col in range(size):
        horizontal(half, col)
        if col != exit_col:
            horizontal(-half, col)
    for row in range(size):
        vertical(-half, row)
        vertical(half, row)
    for row in range(size):
        for col in range(size - 1):
            if frozenset(((col, row), (col + 1, row))) not in open_edges:
                vertical(-half + (col + 1) * cell, row)
    for row in range(size - 1):
        for col in range(size):
            if frozenset(((col, row), (col, row + 1))) not in open_edges:
                horizontal(-half + (row + 1) * cell, col)

    start_xy = (-half + (start_col + .5) * cell, half - cell / 2)
    exit_xy = (-half + (exit_col + .5) * cell, -half)
    labels = np.zeros((2, 2), dtype=np.int16)
    name = f"holdout_{seed}" if size == 5 and cell == .75 else f"holdout_{size}x{size}_{round(cell * 100):02d}cm_{seed}"
    spec = MazeSpec(name=name, family="holdout", walls_local=tuple(walls),
                    wall_bbox=(-half, -half, half, half),
                    bounds=(-half - .4, -half - .4, half + .4, half + .4),
                    openings=(Opening("exit", exit_xy, "south", cell / 2),),
                    start_xy_local=start_xy, start_yaw=-math.pi / 2,
                    zone_res=1., zone_labels=labels, zone_centroids_local=((0., 0.),),
                    zone_start=0, placement_offset=tuple(C.MAZE_CENTER_XY))
    return spec
