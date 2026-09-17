import math
from types import SimpleNamespace

import numpy as np
import pytest

from rl_training.eval_hybrid import Geometry
from rl_training.hybrid_navigation import integrate
from rl_training.oriented_navigation import filter_rectangular, tracking_observation, tracking_teacher
from rl_training.eval_hybrid_gazebo import worker_identity


def corridor():
    return Geometry([SimpleNamespace(cx=0., cy=y, half_length=2., half_thickness=.01, yaw=0.)
                     for y in (-.16, .16)])


def test_rectangle_can_move_in_corridor_too_narrow_for_enclosing_disk():
    g = corridor()
    pose = np.zeros(3)
    v, w, override = filter_rectangular(g, pose, g.scan(pose), .08, 0.)
    assert v == .08 and w == 0 and not override


def test_rotation_is_reduced_before_corner_sweeps_into_wall():
    g = corridor()
    pose = np.zeros(3)
    v, w, override = filter_rectangular(g, pose, g.scan(pose), 0., 1.2)
    assert override and abs(w) < 1.2
    for t in np.linspace(0, .3 + abs(w) / 4., 301):
        assert not g.collides(integrate(pose, v, w, t))


@pytest.mark.parametrize("bad", [np.nan, -np.inf, 0., -.1])
def test_invalid_scan_stops_robot(bad):
    scan = np.full(36, 3.)
    scan[5] = bad
    assert filter_rectangular(corridor(), np.zeros(3), scan, .1, .5) == (0., 0., True)


def test_lidar_obstacle_absent_from_known_map_blocks_forward_motion():
    scan = np.full(36, 3.)
    scan[0] = .08  # Sensor is 8 cm ahead: return at body x=16 cm.
    v, w, override = filter_rectangular(corridor(), np.zeros(3), scan, .18, 0.)
    assert override and v < .18


def test_reference_features_are_in_robot_frame():
    scan = np.full(36, 3.)
    obs = tracking_observation(scan, np.array([1., 2., math.pi / 2]),
                               np.array([1., 2.1, math.pi / 2]), np.array([.06, .2]))
    np.testing.assert_allclose(obs[36:], [.5, 0., 0., 1., .6, .25], atol=1e-6)


def test_teacher_uses_reference_curvature_at_zero_tracking_error():
    obs = tracking_observation(np.full(36, 3.), np.zeros(3), np.zeros(3), np.array([.06, .2]))
    np.testing.assert_allclose(tracking_teacher(obs[None])[0], [.6, .25], atol=1e-6)


def test_lateral_error_cannot_cancel_an_in_place_reference_pivot():
    # Gazebo sigma_1 regression: a -3.1 cm lateral error used to cancel
    # the +0.196 rad/s reference turn and freeze the policy mid-route.
    obs = tracking_observation(np.full(36, 3.), np.zeros(3),
                               np.array([-.016, -.031, .02]), np.array([0., .196]))
    action = tracking_teacher(obs[None])[0]
    assert action[0] == 0.
    assert action[1] * .8 > .19


def test_reusing_ros_domain_does_not_reuse_robot_topics():
    partition_a, robot_a = worker_identity(92)
    partition_b, robot_b = worker_identity(92)
    assert partition_a != partition_b and robot_a != robot_b
    assert robot_a.startswith("92_") and robot_b.startswith("92_")
