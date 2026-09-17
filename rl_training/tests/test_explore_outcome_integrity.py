"""Regression tests on the real step method; no running Gazebo required."""
import math
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

from rl_training import config as C
from rl_training.explore_env import GazeboExploreEnv, PHASE_EXIT


@pytest.fixture
def terminal_env(monkeypatch):
    monkeypatch.setattr("rl_training.explore_env.MR.exit_crossed", lambda *a: True)
    monkeypatch.setattr("rl_training.explore_env.MR.out_of_bounds", lambda *a: False)
    return SimpleNamespace(
        _pending_action=None, _apply_action=lambda a: a,
        _wait_fresh_step=lambda: None,
        _snapshot=lambda: (np.full(C.N_RAYS, 0.01), (1.0, 2.0), math.pi / 2),
        _step_count=0, _prev_xy=(1.0, 2.0), _mapper=Mock(),
        _ray_angles_or_default=lambda: np.arange(C.N_RAYS) * 2 * math.pi / C.N_RAYS,
        _collision_thresh=np.full(C.N_RAYS, 0.15),
        _spec=SimpleNamespace(exit_opening=None), _maze_name="test",
        _last_action_scale=1., _last_safety_override=False,
        _zone=SimpleNamespace(n_cells=25, n_zones=25), _phase=PHASE_EXIT,
        _success_streak={}, _exit_thresholds={}, _exit_min_zones=25,
        _exit_entered_at=1, _stack_obs=lambda *a: np.zeros(150),
        _stop_wheels=Mock(), _save_map=Mock(),
    )


def test_exit_with_collision_is_failure(terminal_env):
    _, _, terminated, _, info = GazeboExploreEnv.step(terminal_env, np.zeros(2))
    assert terminated
    assert info.get("collision") is True
    assert not info.get("success", False)


def test_mapping_uses_rotated_lidar_mount(terminal_env):
    GazeboExploreEnv.step(terminal_env, np.zeros(2))
    x, y = terminal_env._mapper.update.call_args.args[:2]
    assert x == pytest.approx(1.0)
    assert y == pytest.approx(2.0 + C.LIDAR_X_OFF)
