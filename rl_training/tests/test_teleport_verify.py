#!/usr/bin/env python3
"""
test_teleport_verify.py — Reset must never hand back a robot that never moved.

Measured on two consecutive campaigns (v8_sacfd 112/993 episodes, v9_smooth
1059/1503): **100%** of `out_of_bounds` terminations had `ep_len == 1` and a
terminal pose 7-15 m outside the bounds of EVERY maze in the world. No robot
ever drove out through an opening — the Gazebo `set_pose` service stopped
answering, `_teleport_start` logged a warning and returned, and `reset`
snapshotted the stale pose and reported it as a legal start state. The next
step then scored an immediate -30 out-of-bounds terminal and SB3 wrote it
straight into the replay buffer.

That is a data-poisoning bug, not a navigation one: 1059 identical -30
terminals drove mean coverage from 0.35 to 0.00 and mean episode length from
673 steps to 1. The env must verify the robot actually arrived, and must fail
LOUDLY when it did not so the supervisor restarts instead of training on
garbage.
"""

import pytest

from rl_training import config as C
from rl_training.explore_env import SimUnresponsiveError, teleport_landed


def test_a_robot_that_arrived_counts_as_landed():
    """Settling jitter after a teleport is millimetres, not metres."""
    assert teleport_landed((-12.0, -12.0), (-12.0, -12.0))
    assert teleport_landed((-12.0, -12.0), (-11.99, -12.01))


def test_the_measured_failure_poses_are_rejected():
    """The regression this file exists for.

    These are verbatim terminal poses from out_of_bounds episodes in the two
    campaigns; every one of them was reported as a valid reset state.
    """
    for target, observed in (
            ((-12.0, -12.0), (-7.9933, -0.4603)),   # v9 robot_1, delta_1
            ((-12.0, 12.0), (-7.9640, -0.4411)),    # v9 robot_1, sigma_5
            ((-12.0, -4.0), (-7.9502, -0.4351)),    # v9 robot_1, ortho_3
    ):
        assert not teleport_landed(target, observed)


def test_the_tolerance_cannot_reach_a_neighbouring_maze():
    """Slots are 8 m apart; a tolerance near that would accept the wrong maze.

    The check has to be tight enough that landing in the maze NEXT DOOR is
    still a failure, since that robot would be scored against a maze it is
    not standing in.
    """
    assert C.EXPL_TELEPORT_TOL < 4.0
    assert not teleport_landed((-12.0, -12.0), (-4.0, -12.0))


def test_a_pose_just_outside_the_tolerance_is_a_failure():
    """Boundary: the predicate must not be vacuously true."""
    tol = C.EXPL_TELEPORT_TOL
    assert teleport_landed((0.0, 0.0), (0.9 * tol, 0.0))
    assert not teleport_landed((0.0, 0.0), (1.1 * tol, 0.0))


def test_retries_are_bounded_and_more_than_one():
    """One dropped service call is transient; a hung server never recovers.

    Retrying forever would hang the fleet silently — the same invisible
    failure we are fixing — so the count must be finite and small.
    """
    assert 1 < C.EXPL_TELEPORT_TRIES <= 5


def test_the_failure_is_a_distinct_loud_exception():
    """The supervisor restarts on rc != 0; a warning log restarts nothing.

    A dedicated type also stops a bare `except Exception` elsewhere from
    quietly swallowing a dead simulator.
    """
    assert issubclass(SimUnresponsiveError, RuntimeError)
    with pytest.raises(SimUnresponsiveError):
        raise SimUnresponsiveError("boom")
