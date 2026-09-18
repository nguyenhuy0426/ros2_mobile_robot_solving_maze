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


def generate_hex_holdout(seed, diameter=7, side=0.65):
    """Generate a perfect maze on a regular hexagonal tiling.

    ``diameter`` is the number of cells across opposite corners and must be an
    odd number. A diameter of seven contains 37 hexagonal cells.
    """
    if diameter < 3 or diameter % 2 == 0:
        raise ValueError("hex holdout diameter must be an odd number >= 3")
    if side <= 0:
        raise ValueError("hex holdout side must be positive")
    radius = diameter // 2
    cells = {(q, r) for q in range(-radius, radius + 1)
             for r in range(-radius, radius + 1)
             if abs(q + r) <= radius}
    directions = (
        ((1, 0), math.pi / 6), ((1, -1), -math.pi / 6),
        ((0, -1), -math.pi / 2), ((-1, 0), -5 * math.pi / 6),
        ((-1, 1), 5 * math.pi / 6), ((0, 1), math.pi / 2),
    )

    def center(cell):
        q, r = cell
        return np.array((1.5 * side * q,
                         math.sqrt(3) * side * (r + q / 2)), dtype=float)

    rng = np.random.default_rng(seed)
    top = [c for c in cells if (c[0], c[1] + 1) not in cells]
    bottom = [c for c in cells if (c[0], c[1] - 1) not in cells]
    start = top[int(rng.integers(len(top)))]
    exit_cell = bottom[int(rng.integers(len(bottom)))]

    visited, stack, open_edges = {start}, [start], set()
    while stack:
        current = stack[-1]
        candidates = []
        for (dq, dr), _ in directions:
            nxt = (current[0] + dq, current[1] + dr)
            if nxt in cells and nxt not in visited:
                candidates.append(nxt)
        if not candidates:
            stack.pop()
            continue
        nxt = candidates[int(rng.integers(len(candidates)))]
        open_edges.add(frozenset((current, nxt)))
        visited.add(nxt)
        stack.append(nxt)

    thickness = C.EXPL_WALL_T
    apothem = math.sqrt(3) * side / 2
    walls = []
    for cell in sorted(cells):
        cxy = center(cell)
        for (dq, dr), normal in directions:
            nxt = (cell[0] + dq, cell[1] + dr)
            if nxt in cells:
                if cell > nxt or frozenset((cell, nxt)) in open_edges:
                    continue
            elif cell == exit_cell and (dq, dr) == (0, -1):
                continue
            midpoint = cxy + apothem * np.array((math.cos(normal), math.sin(normal)))
            walls.append(Wall(float(midpoint[0]), float(midpoint[1]),
                              side / 2 + thickness, thickness / 2,
                              normal + math.pi / 2))

    all_centers = np.asarray([center(c) for c in cells])
    margin = side + .4
    x0, y0 = all_centers.min(axis=0) - margin
    x1, y1 = all_centers.max(axis=0) + margin
    exit_xy = center(exit_cell) + np.array((0., -apothem))
    labels = np.zeros((2, 2), dtype=np.int16)
    return MazeSpec(
        name=f"hexagon_{diameter}x{diameter}_{round(side * 100):02d}cm_{seed}",
        family="hex_holdout", walls_local=tuple(walls),
        wall_bbox=(float(x0 + .4), float(y0 + .4),
                   float(x1 - .4), float(y1 - .4)),
        bounds=(float(x0), float(y0), float(x1), float(y1)),
        openings=(Opening("exit", tuple(exit_xy), "south", side / 2),),
        start_xy_local=tuple(center(start)), start_yaw=-math.pi / 2,
        zone_res=1., zone_labels=labels, zone_centroids_local=((0., 0.),),
        zone_start=0, placement_offset=tuple(C.MAZE_CENTER_XY))
