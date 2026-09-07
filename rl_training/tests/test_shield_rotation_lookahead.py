#!/usr/bin/env python3
"""
test_shield_rotation_lookahead.py — the shield must watch the whole scan, and
must see the wall its own rotation is about to sweep the chassis into.

Two measured facts drive this file.

1. THE SHIELD CANNOT SEE WHERE THE ROBOT DIES. It triggers on the slack of a
   +-35 deg cone centred on the direction of travel. Over the 355 collisions
   of the v9 run that survived the teleport bug, the ray that breached at the
   moment of death was FRONT on 1%: REAR 34%, LEFT 34%, RIGHT 30%. On the
   44 collisions of the patched v10 run it is FRONT 15%, RIGHT 38%, LEFT 22%,
   REAR 22%. So the sector the shield watches is the one sector the robot
   almost never dies in.

2. ROTATION DESTROYS SLACK FASTER THAN TRANSLATION DOES. The collision
   threshold is a function of bearing -- 0.100 m at the nose, 0.127 m at the
   flank, 0.260 m astern, because the lidar sits 0.08 m forward of the
   chassis centre -- and it steps by 0.0685 m between the 150 deg and 160 deg
   rays, where a ray stops striking the side of the chassis rectangle and
   starts striking its rear face. A stationary obstacle therefore LOSES
   0.0685 m of slack for every 10 deg the robot yaws, without moving at all.
   At EXPL_TURN_MAX the chassis yaws 13.8 deg per control step (19.9 deg
   under the shield's own escape turn), so a spin burns 0.094 m of slack per
   step against a 0.18 m trigger: 1.9 steps from trigger to contact. That is
   why the robot dies on its flanks and its back while turning, and why
   raising the trigger cannot fix it -- a threshold high enough to give a
   spin room to stop would fire continuously in a 0.75 m corridor, whose
   flank slack is only 0.248 m.

The fix is to judge the command by the slack it will LEAVE, not the slack it
starts from: rotate the threshold curve by the yaw the command is asking for
and re-subtract. That is direction-aware for free -- turning away from the
wall predicts more slack, turning into it predicts less -- which a scalar
threshold can never be.

3. THE PREDICTION IS A VETO, NOT A THRESHOLD. The first cut of this file
   triggered the override on the GLOBAL minimum of the prediction at the cone
   band's own 0.18 m. Measured over 1560 collision-free poses sampled across
   all 13 mazes, that fires on 84% of them with a STRAIGHT command -- because
   a global minimum is not comparable to a cone minimum. The rear threshold is
   0.260 m, so a wall 0.30 m astern (an ordinary fact of a 0.75 m maze) already
   reads 0.04 m of slack while the robot is perfectly safe driving away from
   it. A guard that fires everywhere buys survival by refusing to explore --
   the passive-survival trap that already cost this project a campaign.

   So the prediction gets its OWN, small margin (EXPL_SHIELD_ROT_MARGIN): it
   asks "will this turn leave me anything?", not "am I near a wall?". On the
   same 1094 poses that are not already hugging a wall, it adds 9.8 points of
   firing on top of the cone rule's 30.1%, and 9.7 of those 9.8 are commands
   where the OPPOSITE turn is clear -- so the guard mirrors the turn and the
   robot keeps driving. Only 0.1% are boxed in badly enough to need the escape.
"""

import numpy as np
import pytest

from rl_training import config as C
from rl_training.explore_env import explore_shield, rotation_lookahead_slack
from rl_training.wheel_env import compute_collision_thresholds

THRESH = compute_collision_thresholds()
N = THRESH.size
DEG = np.arange(N) * (360.0 / N)


def _open(default: float = 3.0) -> np.ndarray:
    return np.full(N, default, dtype=np.float32)


def _corridor(half_width: float = 0.375) -> np.ndarray:
    """A real 0.75 m corridor, not a flat block of equal ranges.

    Range to a straight wall grows as 1/cos away from the perpendicular, so
    a constant-range side sector understates the clearance at the oblique
    rays and would make this regression guard pass for the wrong reason.
    """
    scan = _open()
    for i, d in enumerate(DEG):
        for wall in (90.0, 270.0):
            off = np.radians(abs((d - wall + 180.0) % 360.0 - 180.0))
            if off < np.radians(80.0):
                scan[i] = min(scan[i], half_width / np.cos(off))
    return scan


def _wheels(fwd: float, turn: float) -> np.ndarray:
    return np.array([fwd - turn, fwd + turn], dtype=np.float32)


# ── the lookahead itself ────────────────────────────────────────────────

def test_a_straight_command_predicts_the_slack_it_already_has():
    """No yaw, no bearing shift: the guard must reduce to plain slack."""
    scan = _corridor()
    pred = rotation_lookahead_slack(scan, THRESH, turn=0.0)
    assert pred == pytest.approx(scan - THRESH, abs=1e-6)


def test_the_lookahead_is_pessimistic_never_optimistic():
    """It guards; it must never hand back MORE slack than measured now.

    Predicting a rosier future than the present would let a command that is
    already inside the collision threshold pass the trigger.
    """
    rng = np.random.default_rng(0)
    for _ in range(50):
        scan = rng.uniform(0.06, 3.0, N).astype(np.float32)
        turn = float(rng.uniform(-1.0, 1.0))
        assert np.all(rotation_lookahead_slack(scan, THRESH, turn)
                      <= scan - THRESH + 1e-6)


def test_a_spin_next_to_a_wall_predicts_less_slack_than_a_spin_in_free_space():
    """The whole point: identical yaw, different consequence."""
    near = _open()
    near[15] = 0.42                       # one return 150 deg off the nose
    turn = C.EXPL_TURN_MAX
    assert rotation_lookahead_slack(near, THRESH, -turn).min() < \
        rotation_lookahead_slack(_open(), THRESH, -turn).min()


# ── the trigger ─────────────────────────────────────────────────────────

def test_the_override_sees_a_flank_wall_the_travel_cone_cannot():
    """85-99% of the deaths breach outside the +-35 deg travel cone.

    A wall 0.16 m off the left flank leaves 0.033 m of slack -- under one
    control step of travel at 0.48 m/s -- while the front cone reads 2.9 m and
    reports the robot perfectly safe. The prediction is taken on the whole
    scan precisely so that this pose cannot be invisible.
    """
    scan = _open()
    flank = np.abs((DEG - 90.0 + 180.0) % 360.0 - 180.0) <= 10.0
    scan[flank] = 0.16
    assert float((scan - THRESH)[np.abs(DEG) <= 35.0].min()) > C.EXPL_SHIELD_SLOW
    assert float((scan - THRESH).min()) <= C.EXPL_SHIELD_ROT_MARGIN

    _out, _scale, overridden = explore_shield(_wheels(1.0, 0.0), scan, THRESH)
    assert overridden, "the shield ignored a wall 3 cm from the chassis"


def test_a_turn_into_the_rear_cliff_is_vetoed_while_the_scan_looks_safe():
    """Slack 0.075 m now; 0.020 m after two steps of the commanded yaw.

    Nothing in the present scan is alarming: 0.075 m clears the veto margin
    and the front cone is empty, so v9 drove this command at full authority.
    What is alarming is the command itself -- yawing right walks this return
    from the 0.205 m side threshold onto the 0.260 m rear one, and by the time
    the slack alone reports it there are under two control steps left.
    """
    scan = _open()
    scan[15] = 0.28                       # one return 150 deg off the nose
    assert float((scan - THRESH).min()) > C.EXPL_SHIELD_ROT_MARGIN   # safe today
    assert float(rotation_lookahead_slack(scan, THRESH,
                                          -C.EXPL_TURN_MAX).min()) \
        <= C.EXPL_SHIELD_ROT_MARGIN                                  # not tomorrow

    _out, _scale, overridden = explore_shield(_wheels(0.5, -C.EXPL_TURN_MAX),
                                              scan, THRESH)
    assert overridden


def test_the_veto_mirrors_the_turn_rather_than_stopping_the_robot():
    """A veto that stopped the robot would be the passive-survival trap.

    The prediction is direction-aware, so when one turn direction is doomed
    and the other is clear there is no reason to spend the step standing
    still: keep the forward channel, flip the turn, carry on exploring. Only
    a command with no safe rotation left may fall through to the escape.
    """
    scan = _open()
    scan[15] = 0.28
    out, _scale, overridden = explore_shield(_wheels(0.5, -C.EXPL_TURN_MAX),
                                             scan, THRESH)
    assert overridden
    fwd, turn = 0.5 * float(out[0] + out[1]), 0.5 * float(out[1] - out[0])
    assert turn == pytest.approx(C.EXPL_TURN_MAX, abs=1e-5)   # mirrored
    assert fwd == pytest.approx(0.5, abs=1e-5)                # still driving


def test_a_command_with_no_safe_rotation_falls_through_to_the_escape():
    """Mirroring is only an answer while one side is still open.

    With the rear-threshold cliff loaded on BOTH quarters, either turn
    direction sweeps a corner into a wall, so the guard must hand over to the
    escape -- which picks its direction from the clearances and translates off
    the wall -- instead of mirroring into the other one.
    """
    scan = _open()
    scan[15] = 0.28                       # 150 deg
    scan[21] = 0.28                       # 210 deg
    out, scale, overridden = explore_shield(_wheels(0.5, -C.EXPL_TURN_MAX),
                                            scan, THRESH)
    assert overridden
    # The escape is the only branch that turns HARDER than the policy asked
    # (EXPL_ESCAPE_TURN > EXPL_TURN_MAX) and reports the creep speed rather
    # than the commanded one; a mirror would hand back +EXPL_TURN_MAX at
    # scale 1.0. The magnitudes themselves are compressed by the [-1, 1]
    # wheel clip, so compare the branch, not the raw number.
    turn = 0.5 * float(out[1] - out[0])
    assert abs(turn) > C.EXPL_TURN_MAX
    assert scale == pytest.approx(C.EXPL_SHIELD_FLOOR)


def test_the_same_wall_does_not_trip_the_shield_when_turning_away_from_it():
    """A guard that fires on both turn directions is just a smaller robot.

    Yawing left carries the same return from the 0.205 m threshold down to
    0.142 m, i.e. AWAY from the chassis corner, so this command is the
    correct escape and must survive untouched.
    """
    scan = _open()
    scan[15] = 0.28
    out, scale, overridden = explore_shield(_wheels(0.5, C.EXPL_TURN_MAX),
                                            scan, THRESH)
    assert not overridden and scale == 1.0
    assert out == pytest.approx(_wheels(0.5, C.EXPL_TURN_MAX))


# ── regressions the trigger must not break ──────────────────────────────

def test_driving_a_corridor_is_still_untouched():
    scan = _corridor()
    out, scale, overridden = explore_shield(_wheels(1.0, 0.0), scan, THRESH)
    assert scale == 1.0 and not overridden
    assert out == pytest.approx(_wheels(1.0, 0.0))


def test_turning_a_corner_in_a_corridor_is_still_untouched():
    """The failure mode a raised scalar threshold would have caused.

    Flank slack in a 0.75 m corridor is 0.248 m, so any trigger big enough to
    give a spin two steps of margin would pin the robot at walking pace for
    the whole maze. The lookahead does not, because a corridor's walls run
    parallel to the chassis: rotating the threshold curve against them costs
    far less than rotating it against a corner.
    """
    scan = _corridor()
    assert float(rotation_lookahead_slack(scan, THRESH,
                                          C.EXPL_TURN_MAX).min()) \
        > C.EXPL_SHIELD_ROT_MARGIN
    for turn in (C.EXPL_TURN_MAX, -C.EXPL_TURN_MAX):
        _out, _scale, overridden = explore_shield(_wheels(0.6, turn), scan,
                                                  THRESH)
        assert not overridden, f"corridor turn {turn:+.2f} tripped the shield"


def test_spinning_in_open_space_is_untouched():
    for turn in (C.EXPL_TURN_MAX, -C.EXPL_TURN_MAX, 1.0, -1.0):
        _out, scale, overridden = explore_shield(_wheels(0.0, turn), _open(),
                                                 THRESH)
        assert scale == 1.0 and not overridden
