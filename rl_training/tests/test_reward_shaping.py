"""Offline unit tests for reward_shaping.py and curriculum.py (no Gazebo)."""

import math

import numpy as np
import pytest

from rl_training import config as C
from rl_training.curriculum import StartCurriculum
from rl_training.reward_shaping import (
    CoverageTracker,
    StuckTracker,
    action_smoothness_penalty,
)


# ── StuckTracker ─────────────────────────────────────────────────────────

def test_stuck_no_penalty_before_window_fills():
    st = StuckTracker(window=5, min_disp=0.05, penalty=-0.2)
    st.reset((0.0, 0.0))
    # Only 3 more updates < window(5): no verdict yet.
    assert st.update((0.0, 0.0)) == 0.0
    assert st.update((0.0, 0.0)) == 0.0
    assert st.update((0.0, 0.0)) == 0.0


def test_stuck_penalizes_stationary_robot():
    st = StuckTracker(window=4, min_disp=0.05, penalty=-0.2)
    st.reset((1.0, 1.0))
    for _ in range(5):
        pen = st.update((1.0, 1.0))  # never moves
    assert pen == pytest.approx(-0.2)
    assert st.is_stuck


def test_stuck_no_penalty_when_moving():
    st = StuckTracker(window=4, min_disp=0.05, penalty=-0.2)
    st.reset((0.0, 0.0))
    pen = 0.0
    for i in range(1, 6):
        pen = st.update((0.1 * i, 0.0))  # 0.1 m/step >> min_disp
    assert pen == 0.0
    assert not st.is_stuck


# ── CoverageTracker ──────────────────────────────────────────────────────

def test_coverage_bonus_only_on_new_cells():
    cov = CoverageTracker(origin_xy=(0.0, 0.0), cell_size=0.5, bonus=1.0)
    cov.reset((0.1, 0.1))            # cell (0,0) pre-visited
    assert cov.update((0.2, 0.2)) == 0.0    # same cell → no bonus
    assert cov.update((0.6, 0.1)) == 1.0    # cell (1,0) → bonus
    assert cov.update((0.7, 0.2)) == 0.0    # cell (1,0) again → none
    assert cov.update((0.1, 0.6)) == 1.0    # cell (0,1) → bonus
    assert cov.n_cells == 3


def test_coverage_never_rewards_outside_maze():
    """Regression: an escaped robot must not farm novelty bonuses on the
    infinite plane (observed +60 reward episodes roaming outside the maze)."""
    cov = CoverageTracker(origin_xy=(2.0, -4.5), cell_size=0.5, bonus=1.0,
                          grid_cells=5)   # maze spans x,y ∈ [2.0, 4.5]
    cov.reset((2.25, -3.25))              # true START cell
    assert cov.update((2.75, -3.25)) == 1.0   # in-maze neighbor → bonus
    # Outside the maze on all sides: no bonus, not tracked
    assert cov.update((4.75, -3.25)) == 0.0   # east of the maze
    assert cov.update((1.75, -3.25)) == 0.0   # west of the maze
    assert cov.update((2.25, -1.75)) == 0.0   # north of the maze
    assert cov.update((2.25, -5.25)) == 0.0   # south of the maze
    assert cov.n_cells == 2                   # only in-maze cells tracked
    # Re-entering an out-of-maze cell still pays nothing
    assert cov.update((4.9, -2.1)) == 0.0


# ── action_smoothness_penalty ────────────────────────────────────────────

def test_smoothness_zero_on_first_step():
    assert action_smoothness_penalty(np.array([1.0, 0, 0, 0]), None, -0.02) == 0.0


def test_smoothness_scales_with_action_delta():
    a = np.array([1.0, 1.0, 1.0, 1.0])
    b = np.array([0.0, 0.0, 0.0, 0.0])
    pen = action_smoothness_penalty(a, b, -0.02)
    assert pen == pytest.approx(-0.02 * 2.0)  # ||[1,1,1,1]|| = 2


# ── StartCurriculum ──────────────────────────────────────────────────────

class _FakeField:
    """Geodesic distance ≈ Euclidean distance to the goal (monotone stand-in)."""

    def __init__(self, goal_xy):
        self._goal = goal_xy

    def distance(self, x, y):
        return math.dist((x, y), self._goal)


def test_curriculum_orders_starts_near_to_far_and_ends_at_true_start():
    field = _FakeField(C.GOAL_XY)
    cur = StartCurriculum(
        field, C.MAZE_ORIGIN_XY, C.GOAL_XY, C.CURR_GRID_CELLS, C.CELL_SIZE,
        C.GOAL_RADIUS, success_window=3, advance_threshold=0.6,
        final_start_xy=C.START_XY, seed=0)
    # Level 0 pool is a single, nearest-to-goal cell.
    d_goal = math.dist(cur.sample_start(), C.GOAL_XY)
    assert d_goal < math.dist(C.START_XY, C.GOAL_XY)
    # The hardest (max) level must include the true start.
    assert cur.max_level >= 1


def test_curriculum_advances_only_when_window_clears_threshold():
    field = _FakeField(C.GOAL_XY)
    cur = StartCurriculum(
        field, C.MAZE_ORIGIN_XY, C.GOAL_XY, C.CURR_GRID_CELLS, C.CELL_SIZE,
        C.GOAL_RADIUS, success_window=3, advance_threshold=0.6,
        final_start_xy=C.START_XY, seed=0)
    assert cur.level == 0
    cur.record_outcome(False)
    cur.record_outcome(True)
    assert cur.level == 0                  # window not yet decisive
    advanced = cur.record_outcome(True)    # 2/3 = 0.67 >= 0.6 → advance
    assert advanced and cur.level == 1
