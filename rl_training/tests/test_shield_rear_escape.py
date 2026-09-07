#!/usr/bin/env python3
"""
test_shield_rear_escape.py — The shield's escape manoeuvre must not spin the
robot into the wall behind it.

Measured over the v9 campaign (446 episodes surviving the teleport bug), the
sector whose ray breached the collision threshold at the moment of death:

    family   n     REAR    LEFT    RIGHT   FRONT
    delta    36     16%     16%     63%      2%
    ortho   161     27%     36%     33%      1%
    sigma   159     44%     35%     18%      1%

FRONT is 1-2% everywhere: the front cone the shield watches is the one
direction that essentially never kills the robot. REAR is the single largest
bucket on sigma, whose 69-wall diagonal geometry (only 19 of 69 walls
axis-aligned, the rest at 52-65 deg) puts an oblique surface behind the robot
far more often than the rectilinear ortho family does.

explore_shield's own comment already identified the mechanism -- "the lidar is
mounted 0.08 m off the rotation centre, so a pure in-place turn ORBITS it by
up to 0.16 m without moving the chassis: with a wall behind, that drives a
rear ray further in" -- and fixed it by creeping FORWARD while turning. But
that creep is granted only when the front cone is clear; when the front is
also tight, ``base`` falls back to 0.0 and the command is once again the pure
in-place spin the comment warns about. That fallback is what these tests
close: with the front blocked and the rear open, backing off is available and
strictly better than spinning.
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
    """Scan with named sectors pushed to a given range (metres)."""
    s = np.full(N, default, dtype=np.float32)
    for name, val in sectors.items():
        centre = {"front": 0.0, "rear": 180.0,
                  "left": 90.0, "right": 270.0}[name]
        rel = np.abs((DEG - centre + 180.0) % 360.0 - 180.0)
        s[rel <= 40.0] = val
    return s


def _twist(wheels):
    """(forward, turn) of a wheel pair."""
    return 0.5 * (wheels[0] + wheels[1]), 0.5 * (wheels[1] - wheels[0])


def test_front_blocked_and_rear_open_backs_off_instead_of_spinning():
    """The regression this file exists for.

    A pure in-place spin (forward == 0) leaves the chassis centred on the
    wall it is trying to escape and swings the 0.26 m rear threshold through
    it. With the rear demonstrably clear there is a strictly better move.
    """
    scan = _scan(front=THRESH[0] + 0.5 * C.EXPL_SHIELD_TURN)   # front tight
    wheels, _scale, overridden = explore_shield(np.array([1.0, 1.0]),
                                                scan, THRESH)
    assert overridden
    fwd, turn = _twist(wheels)
    assert abs(turn) > 0.0, "the shield must still steer away"
    assert fwd < 0.0, (
        "front blocked, rear clear: the escape must move the chassis "
        f"backwards off the wall, got forward={fwd:+.3f}")


def test_it_still_creeps_forward_when_the_front_is_the_clear_side():
    """The v7 behaviour must survive: a rear-only breach creeps FORWARD."""
    scan = _scan(rear=THRESH[N // 2] + 0.5 * C.EXPL_SHIELD_TURN)
    wheels, _scale, overridden = explore_shield(np.array([-1.0, -1.0]),
                                                scan, THRESH)
    assert overridden
    fwd, _turn = _twist(wheels)
    assert fwd > 0.0


def test_both_ends_blocked_falls_back_to_turning_in_place():
    """With nowhere to translate, rotating is all that is left.

    This is the one case where the in-place spin is correct, so it must not
    be traded for a reverse that would drive the rear ray in.
    """
    scan = _scan(front=THRESH[0] + 0.5 * C.EXPL_SHIELD_TURN,
                 rear=THRESH[N // 2] + 0.5 * C.EXPL_SHIELD_TURN)
    wheels, _scale, overridden = explore_shield(np.array([1.0, 1.0]),
                                                scan, THRESH)
    assert overridden
    fwd, turn = _twist(wheels)
    assert fwd == pytest.approx(0.0, abs=1e-6)
    assert abs(turn) > 0.0


def test_the_reverse_creep_is_slow_enough_to_stay_recoverable():
    """Backing off is a nudge, not a manoeuvre.

    The rear collision threshold is 0.260 m against the front's 0.100 m
    (the lidar sits 0.08 m forward of the chassis centre), so reverse eats
    its own margin 2.6x faster than forward does. A creep bounded by the
    same EXPL_SHIELD_FLOOR the forward creep uses keeps one control step of
    reverse well inside the slack that triggered it.
    """
    scan = _scan(front=THRESH[0] + 0.5 * C.EXPL_SHIELD_TURN)
    wheels, _scale, _ov = explore_shield(np.array([1.0, 1.0]), scan, THRESH)
    fwd, _turn = _twist(wheels)
    assert abs(fwd) <= C.EXPL_SHIELD_FLOOR + 1e-6
    travel = abs(fwd) * C.EXPL_W_MAX * C.WHEEL_RADIUS * C.EXPL_DT
    assert travel < C.EXPL_SHIELD_TURN, (
        f"one step of reverse travels {travel:.3f} m into a "
        f"{C.EXPL_SHIELD_TURN:.3f} m slack band")
