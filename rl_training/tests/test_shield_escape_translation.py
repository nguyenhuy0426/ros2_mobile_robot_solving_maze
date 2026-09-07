#!/usr/bin/env python3
"""
test_shield_escape_translation.py — the shield's own escape must predict the
wall it is about to back into.

MEASURED MOTIVE. The v10 rotation veto worked: over 126 on-policy episodes of
attempt_20260907_002321, against the 121 of the v9 baseline, deaths with the
shield DISENGAGED fell 69% -> 4%, FRONT deaths 8% -> 2%, RIGHT 38% -> 14%,
LEFT 33% -> 15%, and mean coverage rose 0.324 -> 0.381. But the death mass did
not disappear, it MOVED: REAR deaths went 20% -> 69%, and classifying every
collision by the action_scale logged at the moment of death --

    0.45 == EXPL_SHIELD_FLOOR, the escape creeping while it pivots
    0.00 == the escape's boxed-in fallback, a pure in-place spin
    else == a normally driven or throttled command

-- puts 89 of 98 v10 collisions (91%) INSIDE the shield's own escape, against
34 of 108 (31%) under v9. Of the 68 rear deaths, 45 were creeping at 0.45,
16 were spinning at 0.00, and 66 carried safety_override = 1. The breaching
ray was 160 deg on 41 of them: the bearing where the threshold steps 0.0685 m
as a ray leaves the side of the chassis rectangle and strikes its rear face.

ROOT CAUSE. `rotation_lookahead_slack` models YAW only -- it returns the raw
slack unchanged when `turn == 0`, and its docstring says so: "Range is left
alone". The escape, however, TRANSLATES: it commands `base = +-floor` and
holds that direction across steps through the override hysteresis. It picks
the direction from an instantaneous cone minimum (`rear_slack > turn_at`,
0.18 m) while the threshold astern is 0.260 m, and never re-predicts. So a
wall 0.45 m behind passes the gate and the robot reverses into it at
0.45 * EXPL_W_MAX * WHEEL_RADIUS = 0.216 m/s with nothing watching.

THE FIX. Generalise the veto's prediction from a rotation to an ARC: a body
translation of d along +x moves a return at bearing b to r - d*cos(b), on top
of the threshold rotation the veto already applies. Then let the escape CHOOSE
among its candidate commands by predicted slack instead of falling into a
default. Ranking requires the un-clamped prediction, hence `pessimistic`.
"""

import numpy as np
import pytest

from rl_training import config as C
from rl_training.explore_env import (arc_lookahead_slack, explore_shield,
                                     rotation_lookahead_slack)
from rl_training.wheel_env import compute_collision_thresholds

THRESH = compute_collision_thresholds()
N = THRESH.size
DEG = np.arange(N) * (360.0 / N)


def _open(default: float = 3.0) -> np.ndarray:
    return np.full(N, default, dtype=np.float32)


def _ray(deg: float) -> int:
    return int(round(deg / (360.0 / N))) % N


def _wall_at(scan: np.ndarray, bearing: float, dist: float,
             spread: float = 40.0) -> np.ndarray:
    """Put a flat wall perpendicular to `bearing` at `dist` from the lidar."""
    out = scan.copy()
    for i, d in enumerate(DEG):
        off = abs((d - bearing + 180.0) % 360.0 - 180.0)
        if off < spread:
            out[i] = min(out[i], dist / np.cos(np.radians(off)))
    return out


def _fwd_turn(wheels: np.ndarray) -> tuple:
    return (0.5 * float(wheels[0] + wheels[1]),
            0.5 * float(wheels[1] - wheels[0]))


# ── the arc prediction ──────────────────────────────────────────────────

def test_arc_with_no_translation_is_the_rotation_lookahead():
    """v10's veto must keep behaving byte-for-byte as it does today."""
    scan = _wall_at(_open(), 90.0, 0.375)
    for turn in (-0.65, -0.2, 0.0, 0.2, 0.65):
        assert arc_lookahead_slack(scan, THRESH, 0.0, turn) == pytest.approx(
            rotation_lookahead_slack(scan, THRESH, turn), abs=1e-6)


def test_creeping_forward_at_a_wall_ahead_predicts_less_slack():
    """The case the rotation-only guard is blind to by construction."""
    scan = _wall_at(_open(), 0.0, 0.30)
    still = arc_lookahead_slack(scan, THRESH, 0.0, 0.0, pessimistic=False)
    creep = arc_lookahead_slack(scan, THRESH, C.EXPL_SHIELD_FLOOR, 0.0,
                                pessimistic=False)
    assert creep[_ray(0.0)] < still[_ray(0.0)]


def test_reversing_at_a_wall_astern_predicts_less_slack():
    """The measured killer: 45 of 68 rear deaths were creeping backwards."""
    scan = _wall_at(_open(), 180.0, 0.40)
    still = arc_lookahead_slack(scan, THRESH, 0.0, 0.0, pessimistic=False)
    back = arc_lookahead_slack(scan, THRESH, -C.EXPL_SHIELD_FLOOR, 0.0,
                               pessimistic=False)
    assert back[_ray(180.0)] < still[_ray(180.0)]


def test_ranking_needs_the_unclamped_prediction():
    """Clamping to current slack makes every safe candidate tie.

    The veto must stay pessimistic, but a chooser cannot rank candidates
    whose predictions have all been floored at the same measured minimum.
    """
    scan = _wall_at(_open(), 180.0, 0.40)
    away = arc_lookahead_slack(scan, THRESH, C.EXPL_SHIELD_FLOOR, 0.0,
                               pessimistic=False)
    into = arc_lookahead_slack(scan, THRESH, -C.EXPL_SHIELD_FLOOR, 0.0,
                               pessimistic=False)
    assert away.min() > into.min()
    clamped_away = arc_lookahead_slack(scan, THRESH, C.EXPL_SHIELD_FLOOR, 0.0)
    assert clamped_away.min() <= (scan - THRESH).min() + 1e-6


# ── the escape's choice ─────────────────────────────────────────────────

def test_the_escape_does_not_reverse_into_a_wall_behind_it():
    """Front blocked, rear tight-but-passing: the old gate reverses anyway.

    `rear_slack > turn_at` compares a 0.18 m gate against a 0.260 m rear
    threshold, so a wall well inside braking distance clears it.
    """
    scan = _wall_at(_open(), 0.0, 0.12)            # front blocked -> escape
    scan = _wall_at(scan, 180.0, 0.45)             # wall close astern
    out, _scale, overridden = explore_shield(np.array([1.0, 1.0], np.float32),
                                             scan, THRESH)
    assert overridden
    fwd, _turn = _fwd_turn(out)
    assert fwd >= 0.0, f"escape reversed into a wall 0.45 m astern (fwd={fwd})"


def test_the_escape_prefers_the_direction_with_more_predicted_slack():
    """Given an open side and a blocked one, it must not pick the blocked."""
    scan = _wall_at(_open(), 0.0, 0.12)            # front blocked -> escape
    scan = _wall_at(scan, 180.0, 3.0)              # rear wide open
    out, _scale, overridden = explore_shield(np.array([1.0, 1.0], np.float32),
                                             scan, THRESH)
    assert overridden
    fwd, _turn = _fwd_turn(out)
    assert fwd < 0.0, f"escape refused a wide-open rear (fwd={fwd})"


def test_the_escape_still_moves_when_every_candidate_is_tight():
    """Boxed in, it must still pick the least-bad arc, not freeze."""
    scan = _wall_at(_open(), 0.0, 0.13)
    scan = _wall_at(scan, 180.0, 0.32)
    out, _scale, overridden = explore_shield(np.array([1.0, 1.0], np.float32),
                                             scan, THRESH)
    assert overridden
    assert np.any(np.abs(out) > 1e-6), "escape emitted a dead stop"
