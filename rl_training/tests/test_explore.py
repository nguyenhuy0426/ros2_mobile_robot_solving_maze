#!/usr/bin/env python3
"""
test_explore.py — Offline tests for the explore-then-exit stack.

Covers the SDF generator invariants, the occupancy mapper, the observation
builder, and the safe-speed / exit-geometry helpers. No Gazebo required.
"""

import math
import sys
from pathlib import Path

import numpy as np
import pytest

WS_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(WS_ROOT / "scripts"))

from generate_maze_sdf import (  # noqa: E402
    CELL, EXIT_ROW, GRID, OUTER, ORIGIN_XY, PITCH, START_CELL,
    cell_center, gap_y_range, gridline, reachable_cells, verify_topology,
)
from rl_training import config as C  # noqa: E402
from rl_training.maze_field import MazeDistanceField, parse_walls  # noqa: E402
from rl_training.occ_map import OccupancyGridMapper  # noqa: E402


# ══════════════════════════════════════════════════════════════════════════
# Maze topology / SDF
# ══════════════════════════════════════════════════════════════════════════

def test_topology_fully_connected_and_exit_present():
    verify_topology()
    all_cells = {(c, r) for c in range(GRID) for r in range(GRID)}
    assert reachable_cells(START_CELL) == all_cells


def test_config_matches_generator_geometry():
    assert C.EXPL_CELL_SIZE == CELL
    assert C.EXPL_PITCH == PITCH
    assert C.EXPL_OUTER == OUTER
    assert C.EXPL_ORIGIN_XY == ORIGIN_XY
    assert C.EXPL_START_XY == cell_center(*START_CELL)
    gy0, gy1 = gap_y_range()
    assert C.EXPL_EXIT_XY[0] == gridline("x", GRID)
    assert gy0 <= C.EXPL_EXIT_XY[1] <= gy1
    assert (gy1 - gy0) == pytest.approx(PITCH)
    assert C.EXPL_EXIT_ROW == EXIT_ROW


def test_border_walls_sealed():
    """Every 1 cm point on the border lines lies inside a wall box, except
    the intentional exit gap (this is the physical 'tường kín' guarantee).

    inflate=2 mm absorbs float-accumulation noise at box boundaries while a
    real 3 cm seam (the bug this test guards against) stays 26 mm wide and
    still fails the check.
    """
    walls = parse_walls(C.EXPL_WORLD_SDF)

    def sealed(x: float, y: float) -> bool:
        return any(w.contains(x, y, 0.002) for w in walls)

    gy0, gy1 = gap_y_range()
    step = 0.01
    # West border band (x-gridline 0), full height
    x = gridline("x", 0)
    y = gridline("y", 0)
    while y <= gridline("y", GRID):
        assert sealed(x, y), f"hole in west border at y={y:.3f}"
        y += step
    # East border band, except the exit gap
    x = gridline("x", GRID)
    y = gridline("y", 0)
    while y <= gridline("y", GRID):
        if not (gy0 - 1e-9 <= y <= gy1 + 1e-9):
            assert sealed(x, y), f"hole in east border at y={y:.3f}"
        y += step
    # South / north border bands, full width
    for j in (0, GRID):
        y = gridline("y", j)
        x = gridline("x", 0)
        while x <= gridline("x", GRID):
            assert sealed(x, y), f"hole in border row {j} at x={x:.3f}"
            x += step


def test_stall_monitor():
    from rl_training.reward_shaping import StallMonitor
    m = StallMonitor(window=5, min_disp=0.08, limit=3)
    m.reset((0.0, 0.0))                        # reset pre-fills the window
    assert not m.update((0.0, 0.0))           # 2 of 5
    assert not m.update((0.0, 0.0))           # 3 of 5
    assert not m.update((0.01, 0.0))          # 4 of 5 — window not full yet
    assert not m.update((0.0, 0.0))           # window full: stuck #1
    assert not m.update((0.0, 0.0))           # stuck #2
    assert m.update((0.0, 0.0))               # stuck #3 → terminate
    # Steady real motion (4 cm/s) never trips the monitor…
    m.reset((0.0, 0.0))
    for k in range(1, 11):
        assert not m.update((0.1 * k, 0.0))
    # …and creeping to a stop only terminates after `limit` stationary windows
    for _ in range(3):
        assert not m.update((1.0, 0.0))       # window disp 0.3 / 0.2 / 0.1 m
    assert not m.update((1.0, 0.0))           # stationary window #1
    assert not m.update((1.0, 0.0))           # stationary window #2
    assert m.update((1.0, 0.0))               # stationary window #3 → terminate


def test_world_sdf_exists_and_parses():
    walls = parse_walls(C.EXPL_WORLD_SDF)
    # 20 border segments − 1 exit gap + 16 interior = 35
    assert len(walls) == 35


def test_exit_field_builds_and_reaches_everywhere():
    t = C.EXPL_WALL_T
    field = MazeDistanceField(
        C.EXPL_WORLD_SDF, goal_xy=C.EXPL_EXIT_XY,
        origin_xy=(ORIGIN_XY[0] - t / 2, ORIGIN_XY[1] - t / 2),
        extent=OUTER)
    for c in range(GRID):
        for r in range(GRID):
            x, y = cell_center(c, r)
            assert math.isfinite(field.distance(x, y))
    sx, sy = cell_center(*START_CELL)
    assert 0.0 < field.distance(sx, sy) < OUTER * 4


# ══════════════════════════════════════════════════════════════════════════
# Occupancy mapper
# ══════════════════════════════════════════════════════════════════════════

@pytest.fixture(scope="module")
def mapper():
    walls = parse_walls(C.EXPL_WORLD_SDF)
    m = OccupancyGridMapper(ORIGIN_XY, OUTER, walls, res=C.EXPL_MAP_RES,
                            margin=C.EXPL_MAP_MARGIN, max_range=C.LIDAR_MAX)
    return m


def test_mapper_expected_free_matches_cell_area(mapper):
    # Free area = 25 cells of 0.75² minus nothing (walls excluded from free);
    # grid res 0.05 → cell-area approximation within a few percent.
    expected_area = 25 * CELL * CELL
    approx = mapper.expected_free * C.EXPL_MAP_RES ** 2
    assert abs(approx - expected_area) / expected_area < 0.10


def test_mapper_scan_from_start_marks_free_and_occupied(mapper):
    mapper.reset()
    assert mapper.coverage() == 0.0
    sx, sy = cell_center(*START_CELL)
    angles = [i * (2.0 * math.pi / C.N_RAYS) for i in range(C.N_RAYS)]
    # Corridor is 0.69 m wide → east ray hits the wall ~0.35 m away; north/south
    # also hit nearby. Simulate plausible start-cell readings.
    ranges = np.full(C.N_RAYS, C.LIDAR_MAX, dtype=np.float32)
    ranges[0] = 0.35          # east (body frame 0° with yaw 0)
    ranges[9] = 0.35          # north
    ranges[27] = 0.35         # south
    mapper.update(sx, sy, 0.0, ranges, angles)
    cov = mapper.coverage()
    assert 0.0 < cov < 1.0
    occ = mapper.occupancy_array()
    assert (occ == 100).sum() >= 3          # the three wall hits
    # Second scan from the same pose adds nothing new
    before = mapper.free.sum()
    mapper.update(sx, sy, 0.0, ranges, angles)
    assert mapper.free.sum() == before


def test_mapper_long_range_sweep_raises_coverage(mapper):
    mapper.reset()
    sx, sy = cell_center(*START_CELL)
    angles = [i * (2.0 * math.pi / C.N_RAYS) for i in range(C.N_RAYS)]
    ranges = np.full(C.N_RAYS, 2.5, dtype=np.float32)
    ranges[0] = 0.35
    mapper.update(sx, sy, 0.0, ranges, angles)
    cov_one = mapper.coverage()
    # Move one cell east (col 1, row 2) and scan again
    x2, y2 = cell_center(1, 2)
    ranges2 = np.full(C.N_RAYS, 2.5, dtype=np.float32)
    ranges2[18] = 0.35
    mapper.update(x2, y2, 0.0, ranges2, angles)
    assert mapper.coverage() > cov_one


def test_mapper_save_png(tmp_path, mapper):
    mapper.reset()
    p = tmp_path / "map"
    mapper.save_png(p)
    assert (tmp_path / "map.png").exists()
    from PIL import Image
    img = Image.open(tmp_path / "map.png")
    assert img.size == (mapper.n, mapper.n)


# ══════════════════════════════════════════════════════════════════════════
# Helpers used by the env
# ══════════════════════════════════════════════════════════════════════════

def test_build_explore_obs_dim_and_values():
    from rl_training.explore_env import build_explore_obs
    from collections import deque
    frames = deque([np.full(36, 0.5, np.float32) for _ in range(4)],
                   maxlen=4)
    odom = np.array([0.1, 0.2, 1.0, 0.0], np.float32)
    obs = build_explore_obs(frames, odom, 0.4, 1)
    assert obs.shape == (C.EXPL_STATE_DIM,)
    assert C.EXPL_STATE_DIM == 4 * 36 + 4 + 2
    assert obs[-1] == 1.0        # phase flag
    assert obs[-2] == pytest.approx(0.4)   # coverage fraction


def test_safe_speed_penalty():
    from rl_training.explore_env import safe_speed_penalty
    assert safe_speed_penalty(0.40, 0.35, 0.35, -2.0) == 0.0   # clear → none
    p_slow = safe_speed_penalty(0.05, 0.05, 0.35, -2.0)
    p_fast = safe_speed_penalty(0.05, 0.35, 0.35, -2.0)
    assert p_slow == 0.0 or abs(p_slow) < abs(p_fast)          # slow ≪ fast
    assert p_fast == pytest.approx(-2.0 * 0.35 * (1 - 0.05 / 0.35))


def test_front_clearance():
    from rl_training.explore_env import front_clearance
    angles = np.arange(36) * (2 * math.pi / 36)
    scan = np.full(36, 3.0, np.float32)
    scan[0] = 0.2     # straight ahead
    scan[18] = 0.1    # behind — must be ignored
    assert front_clearance(scan, angles) == pytest.approx(0.2)


def test_coverage_tracker_075_grid():
    from rl_training.reward_shaping import CoverageTracker
    t = CoverageTracker(ORIGIN_XY, CELL, 1.0, grid_cells=5,
                        pitch=PITCH, wall_offset=C.EXPL_WALL_T)
    x0, y0 = cell_center(*START_CELL)
    t.reset((x0, y0))
    assert t.n_cells == 1
    assert t.update((x0, y0)) == 0.0                    # same cell
    x1, y1 = cell_center(1, 2)
    assert t.update((x1, y1)) == 1.0                    # new cell
    assert t.update((x1, y1)) == 0.0                    # revisit
    # A point inside real cell 0 but beyond the naive 0.75 m band boundary:
    # must still map to cell 0 (pitch-aligned), i.e. NO new-cell bonus.
    x_edge, y_edge = ORIGIN_XY[0] + C.EXPL_WALL_T + CELL - 0.01, y0
    assert t.update((x_edge, y_edge)) == 0.0
    # Outside the maze never counts
    assert t.update((ORIGIN_XY[0] - 5.0, ORIGIN_XY[1] - 5.0)) == 0.0
    assert t.n_cells == 2


def test_exit_phase_potential_shaping_sign():
    """Moving toward the exit must give positive shaping."""
    t = C.EXPL_WALL_T
    field = MazeDistanceField(
        C.EXPL_WORLD_SDF, goal_xy=C.EXPL_EXIT_XY,
        origin_xy=(ORIGIN_XY[0] - t / 2, ORIGIN_XY[1] - t / 2),
        extent=OUTER)
    xa, ya = cell_center(3, 2)
    xb, yb = cell_center(4, 2)          # one cell closer to the exit
    d_a, d_b = field.distance(xa, ya), field.distance(xb, yb)
    assert d_b < d_a
    shaping = C.EXPL_EXIT_POT_SCALE * (d_a - d_b)
    assert shaping > 0.0
