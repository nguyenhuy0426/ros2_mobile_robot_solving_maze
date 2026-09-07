#!/usr/bin/env python3
"""
test_shield_escape_sign_latch.py — the shield's escape must hold its turn
direction across steps.

MEASURED MOTIVE. Watching robot_3 in sigma_3 (attempt_20260907_120924) the
chassis sat in a ~15 cm box for the whole 1500-step episode, rocking left then
right, coverage 0.04, R = -169.7, safety_override = 1 on 88% of steps. The
escape chooses its turn sign from ``left_clear`` vs ``right_clear`` (60 deg
cones at +-90 deg) and recomputes it every step. In a staircase-diagonal
corridor the nearer wall alternates tooth to tooth, so that sign flips every
few steps: the robot never completes the ~90 deg sweep that would bring a side
opening into the front cone and let ``base`` release to +-floor, so it is
trapped rotating back and forth until the timeout.

THE FIX. Once the escape is engaged, keep the previous step's turn direction
unless the other side is clearer by more than EXPL_ESCAPE_SIGN_MARGIN. This is
the same hysteresis idea the throttle band already uses for ``engaged``, applied
to the escape's steering sign. ``escape_sign == 0`` (the default) means "not
escaping last step" and reproduces today's behaviour exactly.
"""

import numpy as np
import pytest

from rl_training import config as C
from rl_training.explore_env import explore_shield
from rl_training.wheel_env import compute_collision_thresholds

THRESH = compute_collision_thresholds()
N = THRESH.size
DEG = np.arange(N) * (360.0 / N)


def _scan(default: float = 3.0, **sectors) -> np.ndarray:
    """Uniform scan, optionally overridden on named sectors (metres)."""
    s = np.full(N, default, dtype=np.float32)
    for name, val in sectors.items():
        centre = {"front": 0.0, "rear": 180.0,
                  "left": 90.0, "right": 270.0}[name]
        rel = np.abs((DEG - centre + 180.0) % 360.0 - 180.0)
        s[rel <= 55.0] = val
    return s


def _turn_sign(wheels: np.ndarray) -> int:
    turn = 0.5 * (float(wheels[1]) - float(wheels[0]))
    return 1 if turn > 0.0 else (-1 if turn < 0.0 else 0)


# Both ends tight forces the escape branch; the sides carry the clearance the
# sign is chosen from. The front/rear notches are kept narrow so they do not
# bleed into the +-60 deg side masks explore_shield reads left_clear/right_clear
# from (rel in [30, 150] and [-150, -30]).
def _boxed(left_range: float, right_range: float) -> np.ndarray:
    s = np.full(N, 3.0, dtype=np.float32)
    s[(DEG >= 30.0) & (DEG <= 150.0)] = left_range
    s[(DEG >= 210.0) & (DEG <= 330.0)] = right_range
    s[(DEG <= 20.0) | (DEG >= 340.0)] = 0.12          # wall ahead
    s[(DEG >= 160.0) & (DEG <= 200.0)] = 0.30         # wall astern
    return s


def test_default_sign_zero_reproduces_the_unlatched_choice():
    """escape_sign=0 must not change what the escape does today."""
    scan = _boxed(left_range=2.05, right_range=2.00)   # left marginally clearer
    base, _s, ov = explore_shield(np.array([1.0, 1.0], np.float32), scan, THRESH)
    latched, _s2, ov2 = explore_shield(np.array([1.0, 1.0], np.float32), scan,
                                       THRESH, escape_sign=0)
    assert ov and ov2
    np.testing.assert_array_equal(base, latched)


def test_a_marginal_swing_in_side_clearance_does_not_flip_the_escape():
    """The sawtooth alternation this file exists for."""
    left_clearer = _boxed(left_range=2.05, right_range=2.00)
    w0, _s, ov0 = explore_shield(np.array([1.0, 1.0], np.float32),
                                 left_clearer, THRESH)
    assert ov0
    held = _turn_sign(w0)
    assert held != 0

    # Next step the OTHER side is a hair closer -- inside the margin.
    right_clearer = _boxed(left_range=2.00, right_range=2.05)
    w1, _s1, ov1 = explore_shield(np.array([1.0, 1.0], np.float32),
                                  right_clearer, THRESH, escape_sign=held)
    assert ov1
    assert _turn_sign(w1) == held, "a marginal swing flipped the escape sign"

    # Without the latch, the same step flips.
    w1u, _su, _ovu = explore_shield(np.array([1.0, 1.0], np.float32),
                                    right_clearer, THRESH, escape_sign=0)
    assert _turn_sign(w1u) == -held


def test_a_decisive_swing_still_flips_the_escape():
    """Hysteresis holds a marginal side, not a wall."""
    w0, _s, _ov = explore_shield(np.array([1.0, 1.0], np.float32),
                                 _boxed(left_range=2.05, right_range=2.00),
                                 THRESH)
    held = _turn_sign(w0)

    decisive = _boxed(left_range=2.00, right_range=3.00)   # right wide open
    w1, _s1, ov1 = explore_shield(np.array([1.0, 1.0], np.float32),
                                  decisive, THRESH, escape_sign=held)
    assert ov1
    assert _turn_sign(w1) == -held, "a decisive swing failed to flip the sign"


def test_the_latch_does_nothing_when_the_shield_is_not_escaping():
    """A clear path ahead is untouched no matter what escape_sign says."""
    clear = _scan()
    w, scale, ov = explore_shield(np.array([1.0, 1.0], np.float32), clear,
                                  THRESH, escape_sign=1)
    assert not ov
    assert scale == pytest.approx(1.0)
    np.testing.assert_allclose(w, [1.0, 1.0])
