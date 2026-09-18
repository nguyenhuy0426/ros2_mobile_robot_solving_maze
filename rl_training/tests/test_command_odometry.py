import math

import numpy as np

from rl_training import config as C
from rl_training.eval_hybrid import maze_setup
from rl_training.eval_hybrid_gazebo import CommandOdometry, corrupt_lidar, scan_match_pose
from rl_training.holdout_mazes import generate_holdout


def test_command_odometry_integrates_straight_motion_from_sim_time():
    odom = CommandOdometry([1., 2., math.pi / 2])
    odom.update(10.)
    odom.set_command(.2, 0.)
    np.testing.assert_allclose(odom.update(12.), [1., 2., math.pi / 2])  # stale gap rejected
    odom.update(12.1)
    np.testing.assert_allclose(odom.pose, [1., 2.02, math.pi / 2], atol=1e-7)


def test_command_odometry_uses_exact_constant_turn_arc():
    odom = CommandOdometry([0., 0., 0.])
    odom.update(0.)
    odom.set_command(1., math.pi / 2)
    pose = odom.update(1.)
    np.testing.assert_allclose(pose, [2 / math.pi, 2 / math.pi, math.pi / 2], atol=1e-7)


def test_command_odometry_applies_systematic_scale_error():
    odom = CommandOdometry([0., 0., 0.], linear_scale=1.1, angular_scale=.9)
    odom.update(0.)
    odom.set_command(.2, 0.)
    np.testing.assert_allclose(odom.update(.5), [.11, 0., 0.], atol=1e-7)


def test_lidar_corruption_is_seeded_clipped_and_can_drop_rays():
    scan = np.linspace(.01, 5., 64, dtype=np.float32)
    a = corrupt_lidar(scan, np.random.default_rng(7), noise_std=.02, dropout=.2)
    b = corrupt_lidar(scan, np.random.default_rng(7), noise_std=.02, dropout=.2)
    np.testing.assert_array_equal(a, b)
    assert np.all((a >= C.LIDAR_MIN) & (a <= C.LIDAR_MAX))
    assert np.count_nonzero(a == C.LIDAR_MAX) > 0


def test_scan_match_reduces_local_pose_error_from_lidar_and_known_map():
    spec = generate_holdout(31001)
    _, geometry, _ = maze_setup(spec)
    truth = np.array([*spec.start_xy_local, spec.start_yaw], dtype=float)
    estimate = truth + [.08, -.06, .12]
    corrected, score = scan_match_pose(geometry, estimate, geometry.scan(truth))
    assert np.linalg.norm(corrected[:2] - truth[:2]) < np.linalg.norm(estimate[:2] - truth[:2])
    assert abs(math.atan2(math.sin(corrected[2] - truth[2]),
                          math.cos(corrected[2] - truth[2]))) < .12
    assert score < .08


def test_scan_match_rejects_scan_inconsistent_with_known_map():
    spec = generate_holdout(51002)
    _, geometry, _ = maze_setup(spec)
    estimate = np.array([*spec.start_xy_local, spec.start_yaw], dtype=float)
    inconsistent = np.full(C.N_RAYS, C.LIDAR_MIN, dtype=np.float32)
    corrected, score = scan_match_pose(geometry, estimate, inconsistent)
    np.testing.assert_array_equal(corrected, estimate)
    assert score > .08
