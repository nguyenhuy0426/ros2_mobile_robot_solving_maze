#!/usr/bin/env python3
"""
test_explore_shield.py — Offline tests for the v7 action mapping + shield.

Both helpers are pure functions of (action, scan, thresholds), so they can be
exercised without Gazebo. They exist to fix the measured failure mode of the
2026-09-06 run: 79.6% of terminal contacts were REAR rays, because the lidar
sits 0.08 m ahead of the chassis (rear collision threshold 0.26 m vs 0.10 m
front) while the legacy action space let SAC command full reverse.
"""

import numpy as np
import pytest

from rl_training import config as C
from rl_training.explore_env import explore_shield, map_explore_action
from rl_training.wheel_env import compute_collision_thresholds

THRESH = compute_collision_thresholds()
N = THRESH.size
DEG = np.arange(N) * (360.0 / N)


def _scan(default: float = 3.0, **sectors) -> np.ndarray:
    """Uniform scan, optionally overridden on a named sector."""
    s = np.full(N, default, dtype=np.float32)
    for name, val in sectors.items():
        centre = {"front": 0.0, "rear": 180.0,
                  "left": 90.0, "right": 270.0}[name]
        rel = np.abs((DEG - centre + 180.0) % 360.0 - 180.0)
        s[rel <= 25.0] = val
    return s


# ── action mapping ──────────────────────────────────────────────────────

def test_twist_zero_action_drifts_forward_instead_of_standing_still():
    left, right = map_explore_action(np.zeros(2), mode="twist")
    assert left == pytest.approx(right)
    assert left > 0.0


def test_twist_caps_reverse_at_the_configured_fraction():
    left, right = map_explore_action(np.array([-1.0, 0.0]), mode="twist")
    assert left == pytest.approx(-C.EXPL_REVERSE_FRAC)
    assert right == pytest.approx(-C.EXPL_REVERSE_FRAC)


def test_twist_keeps_full_forward_authority():
    left, right = map_explore_action(np.array([1.0, 0.0]), mode="twist")
    assert left == pytest.approx(1.0)
    assert right == pytest.approx(1.0)


def test_twist_turn_channel_is_a_differential():
    left, right = map_explore_action(np.array([0.0, 0.5]), mode="twist")
    assert right - left == pytest.approx(C.EXPL_TURN_MAX)


def test_twist_saturation_preserves_the_commanded_arc():
    # Full forward + full turn overflows the outer wheel. Clipping only that
    # wheel would turn a hard arc into a forward lurch; scaling both keeps
    # the turn-per-unit-forward ratio the policy asked for. The command is
    # picked to actually saturate at the CURRENT EXPL_TURN_MAX rather than
    # hardcoding the v7 value of 0.9.
    fwd = 1.0
    assert fwd + C.EXPL_TURN_MAX > 1.0, "pick a command that saturates"
    left, right = map_explore_action(np.array([1.0, 1.0]), mode="twist")
    assert max(abs(left), abs(right)) == pytest.approx(1.0)
    assert (right - left) / (right + left) == pytest.approx(
        C.EXPL_TURN_MAX / fwd)


def test_wheels_mode_is_the_legacy_identity_mapping():
    a = np.array([-0.7, 0.3])
    assert map_explore_action(a, mode="wheels") == pytest.approx(a)


def test_unknown_action_mode_is_rejected():
    with pytest.raises(ValueError):
        map_explore_action(np.zeros(2), mode="tank")


# ── shield ──────────────────────────────────────────────────────────────

def test_open_space_leaves_the_command_untouched():
    out, scale, overridden = explore_shield(np.array([1.0, 1.0]), _scan(),
                                            THRESH)
    assert scale == 1.0 and not overridden
    assert out == pytest.approx(np.array([1.0, 1.0]))


def test_side_walls_of_a_corridor_do_not_trip_the_shield():
    # Centred in a 0.75 m corridor: side slack 0.247 m, below the 0.30 m
    # SLOW band but outside the front cone, so full speed must survive.
    scan = _scan(left=0.375, right=0.375)
    out, scale, overridden = explore_shield(np.array([1.0, 1.0]), scan, THRESH)
    assert scale == 1.0 and not overridden
    assert out == pytest.approx(np.array([1.0, 1.0]))


def test_wall_ahead_throttles_forward_before_it_overrides():
    scan = _scan(front=float(THRESH[0]) + 0.25)   # inside SLOW, above TURN
    out, scale, overridden = explore_shield(np.array([1.0, 1.0]), scan, THRESH)
    assert not overridden
    assert C.EXPL_SHIELD_FLOOR <= scale < 1.0
    assert float(out.mean()) == pytest.approx(scale)


def test_wall_close_ahead_overrides_into_a_turn_toward_the_open_side():
    scan = _scan(front=float(THRESH[0]) + 0.05, left=0.4)
    out, _scale, overridden = explore_shield(np.array([1.0, 1.0]), scan, THRESH)
    assert overridden
    assert out[1] < out[0]          # right wheel slower ⇒ turning right


def test_shield_never_reverses_without_checking_behind_it():
    """Reverse is allowed only against a MEASURED rear cone.

    This test used to ban reverse outright. The ban was a proxy for the fact
    that the escape branch never looked behind itself, and reversing blind is
    genuinely worse than not reversing: the rear collision threshold is
    0.260 m against the front's 0.100 m (the lidar sits 0.08 m forward of the
    chassis centre), so an unchecked reverse eats its margin 2.6x faster --
    the effect the neighbouring rear-cone test attributes 79.6% of one run's
    deaths to.

    But the ban's cost was larger than its benefit. Forbidding reverse left
    ``base = 0.0`` -- a pure in-place turn -- as the only escape when the
    front was blocked, and explore_shield's own comment explains why that is
    the worst available move: the off-centre lidar ORBITS by up to 0.16 m
    without the chassis leaving the wall. Measured over the 446 v9 episodes
    that survived the teleport bug, the breaching ray at death was REAR on
    44% of sigma collisions, 27% of ortho and 16% of delta, against 1-2%
    FRONT everywhere.

    So the contract is now conditional rather than absolute: with the rear
    unverified or tight, still no reverse; with it measured clear, backing
    off is permitted and bounded by EXPL_SHIELD_FLOOR.
    """
    # Rear tight as well: nothing to translate into, so rotate in place.
    boxed = _scan(front=0.11, rear=float(THRESH[N // 2]) + 0.05)
    out, _s, overridden = explore_shield(np.array([1.0, 1.0]), boxed, THRESH)
    assert overridden
    assert float(out.mean()) == pytest.approx(0.0, abs=1e-6)

    # Rear measured clear: reverse is allowed, and bounded.
    for sectors in ({"front": 0.11}, {"front": 0.11, "left": 0.2},
                    {"front": 0.11, "right": 0.2}):
        out, _s, overridden = explore_shield(np.array([1.0, 1.0]),
                                             _scan(**sectors), THRESH)
        assert overridden
        assert -C.EXPL_SHIELD_FLOOR - 1e-6 <= float(out.mean()) < 0.0


def test_a_reversing_command_is_judged_by_the_rear_cone():
    # Rear threshold is 0.26 m, so a wall 0.30 m behind is already inside the
    # SLOW band — the case that produced 79.6% of the run's deaths.
    scan = _scan(rear=0.30)
    out, scale, _ov = explore_shield(np.array([-1.0, -1.0]), scan, THRESH)
    assert scale < 1.0
    assert float(out.mean()) > -1.0


def test_degenerate_inputs_pass_through_unchanged():
    a = np.array([0.5, -0.5])
    out, scale, overridden = explore_shield(a, np.array([]), THRESH)
    assert scale == 1.0 and not overridden and out == pytest.approx(a)
