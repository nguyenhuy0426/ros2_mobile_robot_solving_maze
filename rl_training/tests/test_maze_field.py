"""Tests for maze_field.py — SDF wall parsing and geodesic distance field.

Runs without ROS/Gazebo: pure XML parsing + numpy.
"""

import math
from pathlib import Path

import pytest

from rl_training.maze_field import MazeDistanceField, parse_walls

WORLD_SDF = Path(__file__).resolve().parents[2] / "worlds" / "nhom8_maze.sdf"

START_XY = (2.25, -3.25)   # cell (row 2, col 0) center
GOAL_XY = (4.25, -3.25)    # cell (row 2, col 4) center


def test_world_file_exists():
    assert WORLD_SDF.is_file(), f"missing world file: {WORLD_SDF}"


def test_parse_walls_finds_thin_boxes():
    walls = parse_walls(WORLD_SDF)
    # 5x5 maze: 2*5 boundary walls per side pair + interior walls.
    assert len(walls) >= 20
    for w in walls:
        # every wall is a thin box on the maze grid
        assert w.half_thickness < 0.05
        assert 0.2 <= 2 * w.half_length <= 0.6


@pytest.fixture(scope="module")
def field():
    return MazeDistanceField(WORLD_SDF, goal_xy=GOAL_XY)


def test_goal_distance_is_zero(field):
    assert field.distance(*GOAL_XY) < 0.10


def test_start_reachable_and_geodesic_longer_than_euclidean(field):
    d = field.distance(*START_XY)
    euclid = math.dist(START_XY, GOAL_XY)  # 2.0 m
    assert math.isfinite(d)
    assert d >= euclid - 0.15  # geodesic never (meaningfully) shorter
    assert d < 15.0            # sane upper bound for a 5x5 maze


def test_all_cell_centers_reachable(field):
    # every cell center of the 5x5 maze must have a finite geodesic distance
    for row in range(5):
        for col in range(5):
            x = 2.0 + (col + 0.5) * 0.5
            y = -4.5 + (row + 0.5) * 0.5
            d = field.distance(x, y)
            assert math.isfinite(d), f"cell ({row},{col}) unreachable"


def test_point_inside_wall_falls_back_to_finite(field):
    # a point on the outer boundary wall line — nearest-free fallback applies
    d = field.distance(2.0, -3.25)
    assert math.isfinite(d)


def test_progress_decreases_toward_goal(field):
    # moving from start cell toward the open neighbor must reduce distance:
    # take min over the 4 neighbors — at least one must be strictly closer.
    d0 = field.distance(*START_XY)
    neighbors = [(0.5, 0.0), (-0.5, 0.0), (0.0, 0.5), (0.0, -0.5)]
    best = min(
        field.distance(START_XY[0] + dx, START_XY[1] + dy)
        for dx, dy in neighbors
    )
    assert best < d0
