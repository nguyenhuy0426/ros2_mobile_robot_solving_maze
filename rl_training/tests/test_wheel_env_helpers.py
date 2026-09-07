#!/usr/bin/env python3
"""Offline tests for the wheel-env pure helpers (no Gazebo/ROS required)."""

import numpy as np
import pytest

from rl_training import config as C
from rl_training.wheel_env import build_wheel_obs, map_wheel_action


class TestMapWheelAction:
    def test_legacy_passthrough(self):
        action = np.array([0.1, -0.2, 0.3, -0.4], dtype=np.float32)
        np.testing.assert_allclose(map_wheel_action(action, diff_drive=False),
                                   action)

    def test_diff_drive_duplicates_sides(self):
        action = np.array([0.5, -1.0], dtype=np.float32)
        expected = np.array([0.5, -1.0, 0.5, -1.0], dtype=np.float32)
        np.testing.assert_allclose(map_wheel_action(action, diff_drive=True),
                                   expected)

    def test_diff_drive_forward_is_straight(self):
        # Equal left/right commands must drive all four wheels equally.
        out = map_wheel_action(np.array([1.0, 1.0], dtype=np.float32),
                               diff_drive=True)
        assert out.shape == (4,)
        assert np.allclose(out, 1.0)


class TestBuildWheelObs:
    def _frames(self, n=C.WHEEL_FRAME_STACK, dim=C.N_RAYS):
        return [np.full(dim, 0.25 * i, dtype=np.float32) for i in range(n)]

    def test_legacy_lidar_only(self):
        obs = build_wheel_obs(self._frames(), use_odom=False)
        assert obs.shape == (C.WHEEL_STATE_DIM,)   # 144
        np.testing.assert_allclose(obs[:C.N_RAYS], 0.0)
        np.testing.assert_allclose(obs[-C.N_RAYS:], 0.75)

    def test_with_odom(self):
        odom = np.array([0.1, 0.9, 0.0, -1.0], dtype=np.float32)
        obs = build_wheel_obs(self._frames(), use_odom=True, odom=odom)
        expected = C.WHEEL_FRAME_STACK * C.N_RAYS + C.WHEEL_ODOM_DIM
        assert obs.shape == (expected,)            # 148
        np.testing.assert_allclose(obs[-C.WHEEL_ODOM_DIM:], odom)

    def test_odom_required_when_enabled(self):
        with pytest.raises(ValueError):
            build_wheel_obs(self._frames(), use_odom=True, odom=None)


class TestConfigDims:
    def test_legacy_and_v3_dims(self):
        assert C.STATE_DIM == C.FRAME_STACK * C.SINGLE_OBS_DIM + C.N_ACTIONS
        assert C.WHEEL_STATE_DIM == C.WHEEL_FRAME_STACK * C.N_RAYS
        # v3 policy input must match the env's obs builder
        legacy = C.WHEEL_FRAME_STACK * C.N_RAYS
        assert legacy + C.WHEEL_ODOM_DIM == 148
        # v3 terminal penalties keep crashing worse than a bootstrapped timeout
        assert C.WR_COLLISION_V3 < C.WR_GOAL

    def test_diff_drive_action_dim(self):
        assert C.WHEEL_ACTION_DIM == 4  # legacy default; env maps 2-DOF at runtime
