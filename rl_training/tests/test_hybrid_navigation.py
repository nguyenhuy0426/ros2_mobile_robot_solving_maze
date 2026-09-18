import numpy as np
import pytest

from rl_training.hybrid_navigation import GridPlanner, H, filter_command, integrate, observation
from rl_training.eval_hybrid import Geometry
from rl_training.maze_field import Wall


def test_planner_cannot_cross_sealed_wall_or_unknown_space():
    grid = np.zeros((80, 80), dtype=np.int8)
    grid[:, 40] = 100
    planner = GridPlanner(grid, 0.025, (0., 0.))
    assert planner.plan((0.5, 1.), (1.5, 1.)) == []
    grid[:, 40] = -1
    assert GridPlanner(grid, 0.025, (0., 0.)).plan((0.5, 1.), (1.5, 1.)) == []


def test_planner_takes_gap_with_footprint_clearance():
    grid = np.zeros((100, 100), dtype=np.int8)
    grid[:60, 50] = 100
    planner = GridPlanner(grid, 0.025, (0., 0.))
    path = planner.plan((0.5, 0.5), (2., 0.5))
    assert path
    assert max(p[1] for p in path) > 1.5 + H.radius
    assert all(planner.line_safe(a, b) for a, b in zip(path, path[1:]))


def test_sensor_offset_matches_exact_wall_range():
    geom = Geometry([Wall(1., 0., 2., 0.02, np.pi / 2)])
    scan = geom.scan(np.array([0., 0., 0.]))
    assert scan[0] == pytest.approx(0.90, abs=1e-6)


def test_swept_filter_stops_before_wall_despite_safe_current_pose():
    grid = np.zeros((100, 100), dtype=np.int8)
    grid[:, 50] = 100
    planner = GridPlanner(grid, 0.025, (0., 0.))
    v, _, overridden = filter_command(planner, (1.00, 1., 0.), np.full(36, 3.), 0.18, 0.)
    assert overridden
    assert v < 0.18


def test_invalid_lidar_cannot_generate_motion():
    planner = GridPlanner(np.zeros((100, 100)), 0.025, (0., 0.))
    scan = np.full(36, 3.)
    scan[0] = np.nan
    assert filter_command(planner, (1., 1., 0.), scan, 0.18, 0.) == (0., 0., True)
    with pytest.raises(ValueError):
        observation(scan, (1., 1., 0.), (1.5, 1.))


def test_close_obstacle_vetoes_translation_but_keeps_neural_pivot():
    planner = GridPlanner(np.zeros((100, 100)), 0.025, (0., 0.))
    scan = np.full(36, 3.)
    scan[0] = .08
    assert filter_command(planner, (1., 1., 0.), scan, .18, .5) == (0., .5, True)


def test_pivot_does_not_translate():
    pose = integrate((1., 2., 0.), 0., 1., 0.2)
    assert pose == pytest.approx([1., 2., 0.2])
