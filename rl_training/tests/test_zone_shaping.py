#!/usr/bin/env python3
"""
test_zone_shaping.py — Tests for the EXPLORE-phase geodesic machinery (v8).

Covers the per-zone distance fields (rl_training/zone_field.py), the shaping
term computed from them, and the scripted demonstrator that steers down their
gradient. No ROS/Gazebo required.
"""

import math

import numpy as np
import pytest

import rl_training.maze_registry as MR
from rl_training import config as C
from rl_training.reward_shaping import zone_approach_shaping
from rl_training.zone_field import ZoneDistanceFields


@pytest.fixture(scope="module")
def placed():
    """delta_1 placed at its combined-world slot, like the env uses it."""
    name = "delta_1"
    return MR.load_registry()[name].place_at(MR.multi_placement()[name])


@pytest.fixture(scope="module")
def fields(placed):
    return ZoneDistanceFields.from_spec(placed)


def test_distance_to_the_zone_you_stand_in_is_zero(placed, fields):
    """The field of zone z must bottom out inside z itself.

    The whole EXPLORE shaping rests on "d shrinks as I approach zone z", so a
    field whose minimum is not in its own zone would shape toward the wrong
    place — the exact failure mode the exit field's wrong-goal guard exists
    for.
    """
    sx, sy = placed.start_xy_world
    z = placed.zone_label_at_world(sx, sy)
    assert z >= 0
    assert fields.distance_to(z, sx, sy) < 0.30


def test_every_zone_is_reachable_from_the_start(placed, fields):
    """A zone with no finite distance can never be shaped toward.

    All 13 mazes are fully solvable (every zone reachable), so an infinite
    entry means the raster or the inflation radius broke the connectivity,
    not that the maze is hard.
    """
    sx, sy = placed.start_xy_world
    unreachable = [z for z in range(MR.N_ZONES)
                   if not math.isfinite(fields.distance_to(z, sx, sy))]
    assert unreachable == []


def test_nearest_unvisited_ignores_visited_zones(placed, fields):
    sx, sy = placed.start_xy_world
    start_zone = placed.zone_label_at_world(sx, sy)
    all_zones = frozenset(range(MR.N_ZONES))

    z0, d0 = fields.nearest(sx, sy, all_zones)
    assert z0 == start_zone and d0 < 0.30

    z1, d1 = fields.nearest(sx, sy, all_zones - {start_zone})
    assert z1 != start_zone and d1 > d0

    assert fields.nearest(sx, sy, frozenset())[0] == -1


def test_descent_direction_points_at_a_closer_cell(placed, fields):
    """One step along the returned heading must reduce the distance.

    This is what makes the direction usable as a demonstrator command: the
    scripted explorer turns onto it and drives, so a direction that does not
    descend would walk the robot into the wall behind it.
    """
    sx, sy = placed.start_xy_world
    target, d0 = fields.nearest(sx, sy,
                                frozenset(range(MR.N_ZONES))
                                - {placed.zone_label_at_world(sx, sy)})
    ux, uy = fields.descent_direction(target, sx, sy)
    assert math.isclose(math.hypot(ux, uy), 1.0, abs_tol=1e-6)
    step = C.EXPL_ZONE_DESCENT_R * C.EXPL_ZONE_FIELD_RES
    assert fields.distance_to(target, sx + ux * step, sy + uy * step) < d0


def test_shaping_is_zero_on_the_step_that_retargets():
    """Entering a new zone must never be punished by the shaping term.

    The nearest-unvisited distance JUMPS when a zone is ticked off (the
    target becomes the next zone out), so charging that jump would subtract
    a large negative from the very step that earns EXPL_R_CELL. The step
    that retargets re-baselines instead.
    """
    assert zone_approach_shaping(0.10, 1.40, 2.0, retargeted=True) == 0.0
    assert zone_approach_shaping(1.00, 0.90, 2.0, retargeted=False) == \
        pytest.approx(0.2)
    assert zone_approach_shaping(0.90, 1.00, 2.0, retargeted=False) == \
        pytest.approx(-0.2)
    assert zone_approach_shaping(None, 1.0, 2.0, retargeted=False) == 0.0
    assert zone_approach_shaping(1.0, math.inf, 2.0, retargeted=False) == 0.0


def test_unmap_explore_action_inverts_the_twist_mapping():
    """The demonstrator asks for (forward, turn); the policy speaks actions.

    map_explore_action squashes the forward channel onto
    [-EXPL_REVERSE_FRAC, 1], so a demonstrator that wrote its wheel targets
    straight into the action would be driving a different robot than the one
    the buffer is trained on.
    """
    from rl_training.explore_env import map_explore_action, unmap_explore_action
    tmax = C.EXPL_TURN_MAX
    for fwd, turn in ((1.0, 0.0), (0.5, 0.4 * tmax), (0.0, -tmax),
                      (-0.2, 0.1)):
        action = unmap_explore_action(fwd, turn, mode="twist")
        assert action.shape == (2,)
        assert np.all(np.abs(action) <= 1.0 + 1e-6)
        wheels = map_explore_action(action, mode="twist")
        assert 0.5 * (wheels[0] + wheels[1]) == pytest.approx(fwd, abs=1e-5)
        assert 0.5 * (wheels[1] - wheels[0]) == pytest.approx(turn, abs=1e-5)


def test_steer_to_heading_pivots_before_it_drives():
    """The demonstrator must not drive forward while facing the wrong way.

    Its output goes through explore_shield like any action, but the shield
    only reacts to what is already in front of the robot; a demonstrator that
    kept full forward authority while 150 deg off-heading would spend the
    seeded transitions ploughing into the wall it is trying to turn away
    from.
    """
    from rl_training.explore_env import steer_to_heading, map_explore_action

    def twist(ux, uy, yaw):
        w = map_explore_action(steer_to_heading(ux, uy, yaw, mode="twist"),
                               mode="twist")
        return 0.5 * (w[0] + w[1]), 0.5 * (w[1] - w[0])

    fwd, turn = twist(1.0, 0.0, 0.0)                 # dead ahead
    assert fwd == pytest.approx(1.0, abs=1e-5)
    assert turn == pytest.approx(0.0, abs=1e-6)

    fwd, turn = twist(0.0, 1.0, 0.0)                 # 90 deg to the left
    assert fwd == pytest.approx(0.0, abs=1e-5)
    assert turn == pytest.approx(C.EXPL_TURN_MAX, abs=1e-5)

    fwd, turn = twist(-1.0, 0.0, 0.0)                # behind: pivot, never reverse
    assert fwd == pytest.approx(0.0, abs=1e-5)
    assert abs(turn) == pytest.approx(C.EXPL_TURN_MAX, abs=1e-5)

    # Wrapping: the target is 10 deg to the RIGHT of a robot facing +175 deg,
    # so the turn must be negative rather than the long way round.
    yaw = math.radians(175.0)
    ux, uy = math.cos(math.radians(165.0)), math.sin(math.radians(165.0))
    assert twist(ux, uy, yaw)[1] < 0.0


def test_bootstrap_scripted_fills_the_buffer_and_drops_learning_starts():
    """Seeded transitions are worthless if SAC still waits out learning_starts.

    A .zip carries no replay buffer, so after --load SB3 sets
    learning_starts relative to num_timesteps; the bootstrap must both WRITE
    the demonstrator transitions and clear that gate, otherwise the fleet
    replays the scripted data without ever taking a gradient step on it.
    """
    from rl_training.train_explore_multi import _bootstrap_scripted

    class FakeVec:
        num_envs = 2
        def __init__(self):
            self.resets = 0
        def reset(self):
            self.resets += 1
            return np.full((self.num_envs, 4), self.resets, dtype=np.float32)
        def env_method(self, name, *a, **kw):
            assert name == "scripted_action"
            return [np.array([0.5, -0.2], dtype=np.float32)] * self.num_envs
        def step(self, actions):
            obs = np.zeros((self.num_envs, 4), dtype=np.float32)
            return obs, np.ones(self.num_envs), np.zeros(self.num_envs, bool), \
                [{}, {}]

    class FakeModel:
        learning_starts = 5000
        _vec_normalize_env = None
        replay_buffer = object()
        def __init__(self):
            self._last_obs = np.zeros((2, 4), dtype=np.float32)
            self.stored = 0
        def _store_transition(self, buf, actions, new_obs, rewards, dones,
                              infos):
            assert buf is self.replay_buffer
            self.stored += len(actions)

    model, env = FakeModel(), FakeVec()
    n = _bootstrap_scripted(model, env, 6)
    assert n == 6                      # steps, not transitions
    assert model.stored == 12          # 6 steps x 2 robots
    assert model.learning_starts == 0


def test_bootstrap_leaves_no_episode_straddling_the_learn_boundary():
    """An episode half-driven by the demonstrator logs an incoherent row.

    episode_return and path_length are accumulated by the TRAINING callback,
    which only runs inside model.learn(); ep_len, coverage_frac and min_slack
    come from the env's own info and span the whole episode. So an episode
    still in flight when the bootstrap ends is logged with the demonstrator's
    608 steps and 25/25 coverage next to a return and a path length covering
    only the handful of steps that happened under learn() — the ortho_5 row at
    ts 243052 read `ep_len=608, coverage=1.0, path_length=0.198 m`, which is
    physically impossible and reads like a policy solve.

    Resetting at the END of the bootstrap keeps the seeded transitions (they
    are already in the buffer) while guaranteeing every logged episode is
    entirely on-policy.
    """
    from rl_training.train_explore_multi import _bootstrap_scripted

    class FakeVec:
        num_envs = 2
        def __init__(self):
            self.resets = 0
        def reset(self):
            self.resets += 1
            return np.full((self.num_envs, 4), self.resets, dtype=np.float32)
        def env_method(self, name, *a, **kw):
            return [np.array([0.5, -0.2], dtype=np.float32)] * self.num_envs
        def step(self, actions):
            obs = np.zeros((self.num_envs, 4), dtype=np.float32)
            return obs, np.ones(self.num_envs), np.zeros(self.num_envs, bool), \
                [{}, {}]

    class FakeModel:
        learning_starts = 5000
        _vec_normalize_env = None
        replay_buffer = object()
        def __init__(self):
            self._last_obs = None
        def _store_transition(self, buf, actions, new_obs, rewards, dones,
                              infos):
            pass

    model, env = FakeModel(), FakeVec()
    _bootstrap_scripted(model, env, 6)
    assert env.resets == 2, "bootstrap must reset before AND after seeding"
    # learn() continues from _last_obs, so it has to be the post-reset one.
    assert np.all(model._last_obs == 2.0)
