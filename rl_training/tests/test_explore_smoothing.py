#!/usr/bin/env python3
"""
test_explore_smoothing.py — Offline tests for the v9 motion-smoothing layer.

Measured on the live v8 policy (checkpoint 310k, 2026-09-06): the turn channel
is resampled i.i.d. by SAC every 100 ms control step, and on an IDENTICAL
observation two consecutive samples differ by |da| = 0.33 on average (p90
1.10), flipping SIGN on 17% of steps. At the v8 EXPL_TURN_MAX = 0.90 the
full-scale differential is 275 deg/s over a 0.18 m wheel separation, so that
resampling noise alone swung the heading by 9 deg per step on average and
30 deg at the p90 — the "goldfish" weave the operator sees in Gazebo on a
straight corridor. The shield made it worse: it toggles between full speed and
a hard in-place pivot at a single 0.18 m threshold with no hysteresis, so
action_scale averaged 0.623 with a 1.000 median (bang-bang, not a smooth
throttle).

Three fixes, all tested here:
  * a slew-rate limit on the (forward, turn) command;
  * an action-rate penalty so smoothness is actually TRAINED, not just
    filtered;
  * hysteresis on the shield's pivot override.
"""

import numpy as np
import pytest

from rl_training import config as C
from rl_training.explore_env import (explore_shield, map_explore_action,
                                     slew_limit, twist_of_wheels)
from rl_training.reward_shaping import action_rate_penalty
from rl_training.wheel_env import compute_collision_thresholds

THRESH = compute_collision_thresholds()
N = THRESH.size
DEG = np.arange(N) * (360.0 / N)


def _scan(default: float = 3.0, **sectors) -> np.ndarray:
    s = np.full(N, default, dtype=np.float32)
    for name, val in sectors.items():
        centre = {"front": 0.0, "rear": 180.0,
                  "left": 90.0, "right": 270.0}[name]
        rel = np.abs((DEG - centre + 180.0) % 360.0 - 180.0)
        s[rel <= 25.0] = val
    return s


# ── slew limiting ───────────────────────────────────────────────────────

def test_twist_of_wheels_inverts_the_wheel_pair():
    """The limiter works on (forward, turn); the pipeline speaks wheels."""
    for fwd, turn in ((1.0, 0.0), (0.4, -0.3), (-0.2, 0.5)):
        f, t = twist_of_wheels(np.array([fwd - turn, fwd + turn]))
        assert f == pytest.approx(fwd, abs=1e-6)
        assert t == pytest.approx(turn, abs=1e-6)


def test_slew_limit_caps_the_per_step_turn_change():
    """A step change in the turn command is what the operator sees as weave.

    SAC resamples the turn channel independently every 100 ms, so without a
    rate cap the heading can swing by a full EXPL_TURN_MAX reversal in one
    step. Capping the CHANGE leaves the policy full authority — it just has
    to hold an opinion for two consecutive steps to use it.
    """
    out = slew_limit((0.0, 0.0), (1.0, 1.0))
    assert out[1] == pytest.approx(C.EXPL_TURN_RATE)
    assert out[0] == pytest.approx(C.EXPL_FWD_RATE)

    # A reversal is capped in the other direction too.
    out = slew_limit((0.5, C.EXPL_TURN_RATE), (0.5, -1.0))
    assert out[1] == pytest.approx(0.0)


def test_slew_limit_passes_small_changes_through_untouched():
    """The limiter must not fight a policy that is already smooth."""
    prev = (0.5, 0.10)
    cmd = (0.5 + 0.5 * C.EXPL_FWD_RATE, 0.10 + 0.5 * C.EXPL_TURN_RATE)
    out = slew_limit(prev, cmd)
    assert out == pytest.approx(cmd)


def test_full_scale_turn_reaches_a_sane_yaw_rate():
    """0.90 spun the chassis at 275 deg/s inside a 0.75 m corridor.

    A junction turn needs 90 deg in well under a second; anything beyond
    that is only ever seen as spinning. Wheel separation 0.18 m (links at
    y = +-0.09 in one_robot.sdf), rim speed EXPL_W_MAX * WHEEL_RADIUS.
    """
    rim = C.EXPL_W_MAX * C.WHEEL_RADIUS
    yaw_rate = np.degrees(C.EXPL_TURN_MAX * 2.0 * rim / 0.18)
    assert 90.0 <= yaw_rate <= 180.0


def test_a_sustained_turn_is_still_reachable_quickly():
    """Rate limiting must not cost the robot its ability to turn a corner."""
    turn, steps = 0.0, 0
    while turn < 0.99 and steps < 10:
        turn = slew_limit((0.0, turn), (0.0, 1.0))[1]
        steps += 1
    assert steps <= 5


# ── action-rate penalty ─────────────────────────────────────────────────

def test_action_rate_penalty_is_zero_for_a_held_command():
    """Driving straight must cost nothing, or the agent learns to stop."""
    assert action_rate_penalty((0.8, 0.1), (0.8, 0.1), C.EXPL_R_SMOOTH) == 0.0


def test_action_rate_penalty_punishes_sawing_and_never_rewards():
    """Filtering alone hides the weave; the penalty is what removes it.

    The slew limiter clamps how fast the command may move, but a policy that
    saturates the limiter every step still weaves at the limit. Only a cost
    on |da| makes holding a heading strictly better than sawing.
    """
    saw = action_rate_penalty((0.5, -C.EXPL_TURN_RATE), (0.5, C.EXPL_TURN_RATE),
                              C.EXPL_R_SMOOTH)
    mild = action_rate_penalty((0.5, 0.0), (0.5, 0.25 * C.EXPL_TURN_RATE),
                               C.EXPL_R_SMOOTH)
    assert saw < mild <= 0.0
    # Comparable to, and not overwhelming, the per-step time penalty: a
    # smoothness term that dwarfs EXPL_R_CELL would buy stillness.
    assert abs(saw) < abs(C.EXPL_R_CELL)


# ── shield hysteresis ───────────────────────────────────────────────────

def test_shield_override_holds_until_slack_clears_the_release_band():
    """One threshold = bang-bang chatter at 10 Hz; that is the stutter.

    Sitting just at EXPL_SHIELD_TURN the cone slack dithers across the
    trigger from lidar noise alone, so the command alternates between full
    forward and an in-place pivot. Re-arming only above a HIGHER release
    threshold turns that into a single clean manoeuvre.
    """
    trip = _scan(front=THRESH[0] + 0.5 * C.EXPL_SHIELD_TURN)
    _w, _s, engaged = explore_shield(np.array([1.0, 1.0]), trip, THRESH)
    assert engaged

    # Slack recovers a little, but not past the release band: still pivoting.
    mid = _scan(front=THRESH[0] + 0.5 * (C.EXPL_SHIELD_TURN
                                         + C.EXPL_SHIELD_RELEASE))
    _w, _s, still = explore_shield(np.array([1.0, 1.0]), mid, THRESH,
                                   engaged=True)
    assert still
    # The same scan with the shield DISENGAGED must not trip it.
    _w, _s, fresh = explore_shield(np.array([1.0, 1.0]), mid, THRESH,
                                   engaged=False)
    assert not fresh

    clear = _scan()
    _w, _s, released = explore_shield(np.array([1.0, 1.0]), clear, THRESH,
                                      engaged=True)
    assert not released


def test_shield_release_is_above_its_trigger():
    """A release band at or below the trigger is not hysteresis at all."""
    assert C.EXPL_SHIELD_RELEASE > C.EXPL_SHIELD_TURN


# ── end-to-end: the weave the operator reported ─────────────────────────

def test_resampling_noise_no_longer_swings_the_heading_per_step():
    """The regression this whole file exists for.

    Replays the MEASURED consecutive-sample turn deltas of the live v8
    policy (mean 0.33, p90 1.10 on identical observations) through the new
    mapping + limiter and checks the resulting per-step heading change is
    small enough that a straight corridor looks straight.
    """
    rim = C.EXPL_W_MAX * C.WHEEL_RADIUS
    prev = (0.6, 0.0)
    worst = 0.0
    rng = np.random.default_rng(0)
    for _ in range(200):
        a = np.array([rng.uniform(-1, 1), rng.uniform(-1, 1)], dtype=np.float32)
        cmd = twist_of_wheels(map_explore_action(a, mode="twist"))
        nxt = slew_limit(prev, cmd)
        dyaw = np.degrees(nxt[1] * 2.0 * rim / 0.18) * C.EXPL_DT
        worst = max(worst, abs(dyaw - np.degrees(prev[1] * 2.0 * rim / 0.18)
                               * C.EXPL_DT))
        prev = nxt
    # v8 measured 30 deg at the p90 from noise alone; anything under a third
    # of that reads as a steady arc rather than a weave.
    assert worst < 10.0
