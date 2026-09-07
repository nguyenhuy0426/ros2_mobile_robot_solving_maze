#!/usr/bin/env python3
"""
explore_env.py — Gymnasium env: explore-then-exit task (v5, multi-maze).

v5: ALL 13 decoded mazes are hosted side by side in the combined Gazebo world
``nhom8_maze_multi`` (config.MAZE_MULTI_WORLD_NAME). Since gz cannot hot-swap
worlds, the maze is switched PER EPISODE by teleporting the robot to the
selected maze's start pose. Maze geometry, start poses, exit openings and
coverage zones all come from the maze registry (rl_training.maze_registry):
each maze spec is re-placed at its combined-world grid slot
(``MazeSpec.place_at(multi_placement()[name])``). No v4 hardcoded geometry is
used for maze shape.

Task (mirrors the free-roaming behaviour of the two reference repos, but the
exploration policy itself is learned by RL — no BFS/DFS planner):

  Phase EXPLORE — roam every nook of the CURRENT maze, scan the walls with
    the lidar and build a 2D occupancy map. Reward = one-time bonus per newly
    visited zone; the episode REQUIRES visiting all 25 zones.
  Phase EXIT — once all zones are visited the map is complete: the robot must
    find and take the maze's single exit opening (south border for all 13
    mazes) and leave.

Hard rules:
  * Touching a wall terminates the episode (collision → terminal penalty).
  * Leaving through the exit before the map is complete fails the episode.
  * Approaching a wall ahead is only acceptable while slowing down: a
    clearance-scaled speed penalty r ∝ −v·(1 − clear/thresh) makes fast
    motion near the front wall expensive, slow creep cheap.

Observation (150-dim): 4×36 stacked lidar frames + odometry
[x_norm, y_norm, cos yaw, sin yaw] (x/y normalized to the CURRENT maze's
world-frame bbox → [−1, 1]) + [coverage fraction, exit-phase flag].
Action (2-dim): differential-drive wheel velocities [left, right] ∈ (−1, 1).

Training-only signals: geodesic distance-to-exit field (potential shaping in
the EXIT phase only), pose, coverage counts — never observed by the policy.
"""

import math
import random
import subprocess
import threading
import time
from collections import deque
from typing import Any, Dict, FrozenSet, List, Optional, Tuple

import gymnasium as gym
import numpy as np
import rclpy
from gymnasium import spaces
from nav_msgs.msg import OccupancyGrid
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from geometry_msgs.msg import Pose
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Float64

from rl_training import config as C
from rl_training import maze_registry as MR
from rl_training.reward_shaping import (
    ShieldLockMonitor,
    StallMonitor,
    StuckTracker,
    action_rate_penalty,
    zone_approach_shaping,
)
from rl_training.zone_field import (
    ZoneDistanceFields,
    descent_direction,
)
from rl_training.wheel_env import (
    WHEEL_ORDER,
    compute_collision_thresholds,
    map_wheel_action,
)

PHASE_EXPLORE = 0
PHASE_EXIT = 1


def wrap_angle(a: float) -> float:
    """Wrap an angle to (−π, π]."""
    return (a + math.pi) % (2.0 * math.pi) - math.pi


def front_clearance(scan: np.ndarray, ray_angles: np.ndarray,
                    half_fov: float = math.pi / 4.0) -> float:
    """Minimum range among body-frame rays within ±``half_fov`` of heading."""
    rel = np.abs(np.arctan2(np.sin(ray_angles), np.cos(ray_angles)))
    mask = rel <= half_fov
    if not mask.any():
        return float(scan.min())
    return float(scan[mask].min())


def safe_speed_penalty(front_clear: float, speed: float,
                       thresh: float, scale: float) -> float:
    """Clearance-scaled speed penalty: 0 unless front clearance < ``thresh``.

    r = scale · v · (1 − clear/thresh): full strength at contact, fading to
    zero at the threshold — fast near-wall motion is expensive, slow creep
    nearly free ("đi chậm lại để kiểm soát tốc độ").
    """
    if front_clear >= thresh:
        return 0.0
    return float(scale) * abs(speed) * (1.0 - front_clear / thresh)


def safety_action_scale(min_slack: float, margin: float,
                        floor: float = 0.35) -> float:
    """Throttle wheel commands near any chassis collision ray.

    Open space keeps the new higher speed. Near a wall, the command is reduced
    before integration so the safety reward does not have to learn collision
    avoidance entirely from terminal failures. ``floor`` preserves turning
    authority and prevents a zero-action deadlock.
    """
    if margin <= 0.0 or min_slack >= margin:
        return 1.0
    ratio = max(0.0, float(min_slack) / float(margin))
    return float(floor + (1.0 - floor) * ratio)


def map_explore_action(action: np.ndarray,
                       mode: Optional[str] = None) -> np.ndarray:
    """Map the 2-D policy action to a normalized ``[left, right]`` command.

    ``"wheels"`` is the v4-v6 mapping: the action IS the pair of wheel
    commands, symmetric on [-1, 1].

    ``"twist"`` (v7 default) splits the action into (forward, turn) and
    squashes the forward channel onto ``[-EXPL_REVERSE_FRAC, 1]``. The
    motivation is measured, not stylistic: 79.6% of the terminal contacts in
    the 2026-09-06 run were rear rays, because the lidar sits 0.08 m ahead of
    a 0.26 m chassis and so the rear collision threshold is 0.26 m against
    0.10 m at the front. A high-entropy SAC policy on a symmetric wheel space
    commands full reverse about a quarter of the time and dies in ~4 control
    steps. Under "twist" a zero action drifts FORWARD, full forward authority
    is retained, and reverse is capped at what a dead-end back-out needs.
    """
    a = np.clip(np.asarray(action, dtype=np.float32), -1.0, 1.0)
    mode = C.EXPL_ACTION_MODE if mode is None else mode
    if mode == "wheels":
        return a
    if mode != "twist":
        raise ValueError(f"unknown EXPL_ACTION_MODE {mode!r}")
    rev = float(C.EXPL_REVERSE_FRAC)
    fwd = 0.5 * (1.0 + rev) * float(a[0]) + 0.5 * (1.0 - rev)
    turn = float(C.EXPL_TURN_MAX) * float(a[1])
    out = np.array([fwd - turn, fwd + turn], dtype=np.float32)
    # Saturate by scaling, not clipping. Clipping only the wheel that
    # overflows changes the differential, so a commanded pivot silently
    # becomes a forward lurch; dividing both wheels preserves the arc the
    # policy asked for and only slows it down.
    peak = float(np.abs(out).max())
    if peak > 1.0:
        out /= peak
    return out


def unmap_explore_action(fwd: float, turn: float,
                         mode: Optional[str] = None) -> np.ndarray:
    """Exact inverse of :func:`map_explore_action` — (forward, turn) → action.

    The scripted demonstrator thinks in "drive forward this hard, arc this
    hard", but every transition it writes into the replay buffer has to carry
    the action the POLICY would have emitted. Under "twist" the forward
    channel is squashed onto [-EXPL_REVERSE_FRAC, 1], so writing a raw
    forward command straight into the action slot would train the critic on a
    different robot than the one that moved.

    ``fwd`` outside [-EXPL_REVERSE_FRAC, 1] (or ``turn`` beyond
    ±EXPL_TURN_MAX) clips, so the demonstrator can ask for full speed without
    knowing the mapping's limits.
    """
    mode = C.EXPL_ACTION_MODE if mode is None else mode
    if mode == "wheels":
        return np.clip(np.array([fwd, turn], dtype=np.float32), -1.0, 1.0)
    if mode != "twist":
        raise ValueError(f"unknown EXPL_ACTION_MODE {mode!r}")
    rev = float(C.EXPL_REVERSE_FRAC)
    a0 = (float(fwd) - 0.5 * (1.0 - rev)) / (0.5 * (1.0 + rev))
    a1 = float(turn) / float(C.EXPL_TURN_MAX)
    return np.clip(np.array([a0, a1], dtype=np.float32), -1.0, 1.0)


def steer_to_heading(ux: float, uy: float, yaw: float,
                     mode: Optional[str] = None) -> np.ndarray:
    """Turn a desired world heading into a policy action.

    Proportional on the wrapped bearing error, saturating at a quarter turn,
    with the forward channel gated by ``cos(err)`` clamped at zero: aligned →
    full speed, broadside → pure pivot, behind → pivot the short way round
    rather than reversing (reverse is what killed 79.6% of the v6 episodes,
    since the rear collision threshold is 0.26 m against 0.10 m at the front).
    """
    err = math.atan2(uy, ux) - yaw
    err = math.atan2(math.sin(err), math.cos(err))     # wrap to [-pi, pi]
    turn = float(np.clip(err / (math.pi / 2.0), -1.0, 1.0)) * C.EXPL_TURN_MAX
    fwd = max(0.0, math.cos(err))
    return unmap_explore_action(fwd, turn, mode)


class SimUnresponsiveError(RuntimeError):
    """The simulator stopped honouring teleports and cannot be trained on.

    Raised by :meth:`ExploreEnv.reset` when the robot is still nowhere near
    its requested start after ``EXPL_TELEPORT_TRIES`` attempts. It is a hard
    failure ON PURPOSE: the previous behaviour — warn and continue — fed SB3
    a stale pose that scored an instant -30 out-of-bounds terminal, and 1059
    of those in one campaign drove mean coverage from 0.35 to 0.00. Crashing
    costs one supervisor restart; continuing costs the policy.
    """


def teleport_landed(target_xy: Tuple[float, float],
                    observed_xy: Tuple[float, float],
                    tol: float = C.EXPL_TELEPORT_TOL) -> bool:
    """Did the robot actually arrive where ``set_pose`` was asked to put it?

    The service returns success as soon as the request is *accepted*, which
    tells us nothing about whether the server ever applied it — a hung server
    accepts and never moves. Comparing the pose we read back against the pose
    we asked for is the only check that distinguishes the two.
    """
    dx = float(observed_xy[0]) - float(target_xy[0])
    dy = float(observed_xy[1]) - float(target_xy[1])
    return math.hypot(dx, dy) <= float(tol)


def episode_step_limit(exit_entered_at: Optional[int],
                       max_steps: int = C.EXPL_MAX_STEPS,
                       exit_budget: int = C.EXPL_EXIT_BUDGET) -> int:
    """Step cap for this episode, given when (if ever) PHASE_EXIT began.

    ``max`` rather than a plain sum: an episode that maps the maze in 200
    steps already has budget in hand, and ``exit_entered_at + exit_budget``
    would truncate it early — the opposite of the intent. The extension only
    ever helps the episode that spent its whole budget exploring.
    """
    if exit_entered_at is None:
        return int(max_steps)
    return int(max(max_steps, exit_entered_at + exit_budget))


def twist_of_wheels(wheels: np.ndarray) -> Tuple[float, float]:
    """``[left, right]`` -> ``(forward, turn)``; inverse of the pairing above."""
    w = np.asarray(wheels, dtype=np.float32)
    return 0.5 * float(w[0] + w[1]), 0.5 * float(w[1] - w[0])


def wheels_of_twist(fwd: float, turn: float) -> np.ndarray:
    """``(forward, turn)`` -> ``[left, right]``, saturated by SCALING.

    Clipping only the wheel that overflows changes the differential, so a
    commanded pivot silently becomes a forward lurch; dividing both wheels
    preserves the arc and only slows it down.
    """
    out = np.array([fwd - turn, fwd + turn], dtype=np.float32)
    peak = float(np.abs(out).max())
    if peak > 1.0:
        out /= peak
    return out


def slew_limit(prev: Tuple[float, float], cmd: Tuple[float, float],
               fwd_rate: float = C.EXPL_FWD_RATE,
               turn_rate: float = C.EXPL_TURN_RATE) -> Tuple[float, float]:
    """Cap how far the (forward, turn) command may move in one control step.

    SAC draws an INDEPENDENT sample every 100 ms, so on an identical
    observation two consecutive turn commands measured 0.33 apart on average
    (p90 1.10) and flipped sign on 17% of steps — 9 deg of heading per step
    from noise alone at the v8 EXPL_TURN_MAX, 30 deg at the p90. That is the
    weave the operator sees on a straight corridor, and no amount of training
    removes it: it is the policy's entropy, which EXPL_TARGET_ENTROPY
    deliberately keeps high.

    A rate cap is not a low-pass filter and does not fight a decisive policy:
    the full turn range is still reachable in ``1/turn_rate`` steps (5 at the
    default), which is all a junction needs. What it cannot do is reverse the
    heading every 100 ms, because that requires the policy to hold the same
    opinion for consecutive steps — exactly what uncorrelated noise cannot.

    The limiter state (``prev``) is deliberately NOT added to the
    observation: doing so would change EXPL_STATE_DIM and orphan every
    existing checkpoint. The residual partial observability is small — the
    4-frame lidar stack already carries the recent motion — and is paid for
    by the action-rate penalty, which is what actually teaches the policy to
    stop asking for reversals.
    """
    pf, pt = float(prev[0]), float(prev[1])
    cf, ct = float(cmd[0]), float(cmd[1])
    fwd = pf + float(np.clip(cf - pf, -fwd_rate, fwd_rate))
    turn = pt + float(np.clip(ct - pt, -turn_rate, turn_rate))
    return fwd, turn


def rotation_lookahead_slack(scan: np.ndarray, threshold: np.ndarray,
                            turn: float,
                            steps: float = C.EXPL_SHIELD_LOOKAHEAD
                            ) -> np.ndarray:
    """Slack each return will have after ``steps`` of the commanded yaw.

    The collision threshold is a function of BEARING, not a radius: 0.100 m
    at the nose, 0.127 m at the flank, 0.260 m astern, because the lidar sits
    LIDAR_X_OFF forward of the chassis centre. It steps by 0.0685 m between
    the 150 deg and 160 deg rays, where a ray stops striking the side of the
    chassis rectangle and starts striking its rear face. A stationary
    obstacle therefore loses 0.0685 m of slack for every 10 deg the robot
    yaws, having never moved -- and at EXPL_TURN_MAX the chassis yaws
    13.8 deg per control step, so a spin burns 0.094 m of slack per step
    against a 0.18 m trigger. Measured on the scan alone, the danger arrives
    1.9 steps before contact; predicted from the command, it arrives in time
    to be acted on.

    A body yaw of psi carries a fixed return from bearing b to b - psi, so
    the threshold it will be judged against is tau(b - psi). Range is left
    alone: for a lidar at the centre of rotation a pure yaw does not change
    it, and the translation component is already covered by the cone bands.

    Returned elementwise with the CURRENT slack so the guard can only ever be
    pessimistic -- a command whose rotation opens the scan up must not be
    credited with slack the robot does not have yet.
    """
    return arc_lookahead_slack(scan, threshold, 0.0, turn, steps)


def arc_lookahead_slack(scan: np.ndarray, threshold: np.ndarray,
                        fwd: float, turn: float,
                        steps: float = C.EXPL_SHIELD_LOOKAHEAD,
                        pessimistic: bool = True) -> np.ndarray:
    """Slack each return will have after ``steps`` of the commanded ARC.

    ``rotation_lookahead_slack`` is the ``fwd == 0`` case of this, and the
    v10 veto still uses it unchanged. The generalisation exists because the
    shield's own ESCAPE translates, and a yaw-only prediction is blind to
    that by construction -- its docstring says "Range is left alone".

    That blindness is where the robot now dies. Over the 126 on-policy
    episodes of attempt_20260907_002321 the veto did what it was built for:
    deaths with the shield disengaged fell 69% -> 4% and FRONT/LEFT/RIGHT
    deaths all roughly halved. But 89 of 98 collisions (91%, against 31%
    under v9) happened INSIDE the escape, 45 of them creeping backwards at
    EXPL_SHIELD_FLOOR, because the escape picked its direction from an
    instantaneous 0.18 m cone gate against a 0.260 m rear threshold and then
    held it across steps.

    Yaw carries a fixed return from bearing b to b - psi, so it is judged
    against tau(b - psi) -- unchanged from the rotation case. A body
    translation of d along +x additionally moves a return at (r cos b,
    r sin b) to (r cos b - d, r sin b), i.e. r -> r - d cos(b) to first
    order in d/r, which is exact for a surface normal to the ray and the
    same order of approximation the yaw term already makes.

    ``pessimistic`` clamps the result elementwise to the CURRENT slack so a
    guard can never be credited with room the robot does not have yet. A
    guard wants that; a CHOOSER cannot use it, because clamping collapses
    every safe candidate onto the same measured minimum and destroys the
    ranking. So the veto keeps it and the escape's selection turns it off.
    """
    scan = np.asarray(scan, dtype=np.float32)
    threshold = np.asarray(threshold, dtype=np.float32)
    n = scan.size
    if n == 0 or threshold.size != n:
        return np.zeros(0, dtype=np.float32)
    slack = scan - threshold
    if float(steps) <= 0.0 or (float(turn) == 0.0 and float(fwd) == 0.0):
        return slack
    rim = C.EXPL_W_MAX * C.WHEEL_RADIUS
    horizon = C.EXPL_DT * float(steps)
    psi = math.degrees(2.0 * float(turn) * rim / C.WHEEL_SEP) * horizon
    dist = float(fwd) * rim * horizon
    deg = np.arange(n, dtype=np.float32) * (360.0 / n)
    tau_next = np.interp((deg - psi) % 360.0,
                         np.append(deg, 360.0),
                         np.append(threshold, threshold[0]))
    rng_next = scan - dist * np.cos(np.radians(deg))
    pred = (rng_next - tau_next).astype(np.float32)
    return np.minimum(slack, pred) if pessimistic else pred


def explore_shield(wheels: np.ndarray, scan: np.ndarray,
                   collision_thresh: np.ndarray,
                   cone_deg: float = C.EXPL_SHIELD_CONE,
                   slow: float = C.EXPL_SHIELD_SLOW,
                   turn_at: float = C.EXPL_SHIELD_TURN,
                   floor: float = C.EXPL_SHIELD_FLOOR,
                   turn_strength: float = C.EXPL_ESCAPE_TURN,
                   engaged: bool = False,
                   release_at: float = C.EXPL_SHIELD_RELEASE,
                   rot_margin: float = C.EXPL_SHIELD_ROT_MARGIN,
                   escape_sign: int = 0,
                   sign_margin: float = C.EXPL_ESCAPE_SIGN_MARGIN
                   ) -> Tuple[np.ndarray, float, bool]:
    """Throttle, then override, a wheel command that is driving into a wall.

    Returns ``(wheels, scale, overridden)``.

    Two bands on the slack of the cone the chassis is actually moving into
    (front when the command drives forward, rear when it reverses):

      * ``slack < slow``    — the forward component is throttled, the turn
        component is preserved so the policy keeps steering authority;
      * ``slack < turn_at`` — the command is replaced by an in-place turn
        toward the side with more clearance. Once that override is
        ``engaged`` it holds until the slack recovers past ``release_at``
        (v9 hysteresis): a single threshold makes the command chatter
        between full forward and a pivot at 10 Hz on the lidar noise alone,
        which is half of the stutter the operator reported. ``engaged`` is
        the ``overridden`` flag returned by the previous call.

    The THROTTLE band watches a cone rather than the global minimum slack,
    and that still matters: centred in a 0.75 m corridor the flank slack is
    0.248 m, so throttling on the global minimum would hold the robot at
    walking pace down every corridor in the maze.

    Both bands are blind where the robot actually dies, though. Over the 355
    collisions of the v9 run that survived the teleport bug the breaching ray
    was FRONT on 1% (REAR 34%, LEFT 34%, RIGHT 30%), so a cone watches the one
    sector the robot almost never hits. Widening it is not the answer: a
    global slack threshold big enough to give a spin room to stop fires on 84%
    of collision-free maze poses even with a straight command, because the
    0.260 m rear threshold puts an ordinary wall astern permanently inside it.

    So the third rule (v10) is a VETO on the ROTATION, not a wider band. The
    threshold is a function of bearing and steps by 0.0685 m between the
    150 deg and 160 deg rays, so a stationary obstacle loses that much slack
    for every 10 deg the robot yaws; at EXPL_TURN_MAX the chassis yaws
    13.8 deg per step, i.e. it burns 0.094 m per step while the scan alone
    reports the danger 1.9 steps before contact. ``rotation_lookahead_slack``
    rotates the threshold curve by the yaw the command is ASKING for, and if
    that leaves less than ``rot_margin`` anywhere on the scan the command is
    refused. Because the prediction is direction-aware the refusal is cheap:
    when the opposite turn is clear the shield MIRRORS the turn and the robot
    keeps driving at its commanded speed -- 9.7 of every 9.8 points of added
    firing, measured across the 13 mazes -- and only a command with no safe
    rotation left reaches the escape.

    The cone band edges are the ones a scripted probe survived 500 steps on
    delta_1 with,
    reaching 6/25 zones at R = +44.8 where the learned policy managed 39
    steps, 1.5/25 and R = -27. The v6 trigger (0.06 m global slack) fired
    1.2 control steps before contact at 0.48 m/s — too late to change the
    outcome, which is why it never appeared in the outcome mix.
    """
    wheels = np.clip(np.asarray(wheels, dtype=np.float32), -1.0, 1.0)
    scan = np.asarray(scan, dtype=np.float32)
    threshold = np.asarray(collision_thresh, dtype=np.float32)
    if scan.size == 0 or threshold.size != scan.size or wheels.size != 2:
        return wheels, 1.0, False

    slack = scan - threshold
    n = scan.size
    deg = np.arange(n, dtype=np.float32) * (360.0 / n)
    rel = (deg + 180.0) % 360.0 - 180.0          # (-180, 180]
    fwd = 0.5 * float(wheels[0] + wheels[1])
    diff = 0.5 * float(wheels[1] - wheels[0])

    centre = 0.0 if fwd >= 0.0 else 180.0
    cone = np.abs((rel - centre + 180.0) % 360.0 - 180.0) <= float(cone_deg)
    cone_slack = float(slack[cone].min()) if cone.any() else float(slack.min())

    trip = release_at if engaged else turn_at
    # The veto is evaluated first but applied last: a front wall inside the
    # cone band still owns the step, because mirroring the turn would leave
    # the forward channel driving into it.
    vetoed = float(rotation_lookahead_slack(scan, threshold, diff).min()) \
        <= float(rot_margin)
    mirrorable = vetoed and diff != 0.0 and float(
        rotation_lookahead_slack(scan, threshold, -diff).min()) > \
        float(rot_margin)

    if cone_slack > trip and not vetoed:
        if cone_slack >= slow:
            return wheels, 1.0, False
        scale = safety_action_scale(cone_slack - turn_at, slow - turn_at,
                                    floor)
        out = np.array([fwd * scale - diff, fwd * scale + diff],
                       dtype=np.float32)
        return np.clip(out, -1.0, 1.0), scale, False

    if cone_slack > trip and mirrorable:
        scale = 1.0 if cone_slack >= slow else safety_action_scale(
            cone_slack - turn_at, slow - turn_at, floor)
        out = np.array([fwd * scale + diff, fwd * scale - diff],
                       dtype=np.float32)
        return np.clip(out, -1.0, 1.0), scale, True

    left = np.abs(rel - 90.0) <= 60.0
    right = np.abs(rel + 90.0) <= 60.0
    left_clear = float(slack[left].min()) if left.any() else float("inf")
    right_clear = float(slack[right].min()) if right.any() else float("inf")
    turn = float(np.clip(turn_strength, 0.2, 1.0))
    if left_clear < right_clear:
        turn = -turn

    # v12: hold the escape's steering sign across steps. turn > 0 means the
    # left 60 deg cone was the roomier one; in a staircase-diagonal corridor
    # that choice alternates tooth to tooth and the robot rocks in place,
    # never completing the sweep that would bring a side opening into the
    # front cone and let `base` release below. Once engaged the sign flips
    # only when the other side is clearer by more than sign_margin -- the same
    # hysteresis the throttle band applies to `engaged`.
    if escape_sign and (turn > 0.0) != (escape_sign > 0):
        held = left_clear if escape_sign > 0 else right_clear
        other = right_clear if escape_sign > 0 else left_clear
        if not (other - held > float(sign_margin)):
            turn = math.copysign(abs(turn), float(escape_sign))

    # Arc away rather than spin in place when there is room ahead. The lidar
    # is mounted 0.08 m off the rotation centre, so a pure in-place turn
    # ORBITS it by up to 0.16 m without moving the chassis: with a wall
    # behind, that drives a rear ray further in. Every scripted-probe death
    # on ortho_1 and sigma_1 was a 160-200 deg ray for exactly this reason.
    # Creeping forward while turning moves the whole chassis off the wall.
    # ...but when the front is ALSO tight the old fallback was base = 0.0,
    # i.e. exactly the in-place spin the paragraph above warns about. That
    # fallback is where the deaths are: over the 446 v9 episodes that
    # survived the teleport bug, the ray that breached at the moment of
    # death was REAR on 44% of sigma collisions, 27% of ortho and 16% of
    # delta, against 1-2% FRONT everywhere. Sigma is worst because only 19
    # of its 69 walls are axis-aligned (the rest sit at 52-65 deg), so an
    # oblique surface behind the robot is the normal case rather than the
    # exception.
    #
    # Translating off the wall beats rotating on it whichever end is open,
    # so the direction comes from the clearances rather than from assuming
    # forward, with the pure rotation kept for the genuinely boxed-in case.
    #
    # v11: judge that choice on the PREDICTED cone, not the measured one.
    # The gate below used to read the instantaneous slack and compare it
    # against turn_at = 0.18 m -- but the rear collision threshold is
    # 0.260 m, so a wall 0.45 m astern scores 0.19 m, clears the gate, and
    # the escape then reverses into it at floor * EXPL_W_MAX * WHEEL_RADIUS
    # = 0.216 m/s while the override hysteresis holds the direction across
    # steps with nothing re-checking. That is where the deaths moved to once
    # the v10 rotation veto closed the driven path: over the 126 on-policy
    # episodes of attempt_20260907_002321, REAR deaths went 20% -> 69% and
    # 89 of 98 collisions (91%, against 31% under v9) happened inside this
    # branch -- 45 of them creeping at exactly EXPL_SHIELD_FLOOR.
    #
    # Scoring the same cone through arc_lookahead_slack costs the candidate
    # the distance it is about to cover, so the 0.45 m wall now scores
    # 0.19 - 0.45 * 0.48 * EXPL_DT * EXPL_SHIELD_LOOKAHEAD = 0.147 m and is
    # refused. Comparing the two predictions instead of testing front first
    # also lets the escape CHOOSE the roomier end rather than fall into the
    # first one that happens to pass.
    front = np.abs(rel) <= float(cone_deg)
    rear = np.abs(np.abs(rel) - 180.0) <= float(cone_deg)

    def _cone_score(base: float, cone: np.ndarray) -> float:
        if not cone.any():
            return float("-inf")
        pred = arc_lookahead_slack(scan, threshold, base, turn,
                                   pessimistic=False)
        return float(pred[cone].min())

    fwd_score = _cone_score(floor, front)
    rev_score = _cone_score(-floor, rear)
    if max(fwd_score, rev_score) <= float(turn_at):
        base = 0.0                      # boxed in: rotating is all that is left
    else:
        base = floor if fwd_score >= rev_score else -floor
    out = np.array([base - turn, base + turn], dtype=np.float32)
    return np.clip(out, -1.0, 1.0), abs(base), True


def build_explore_obs(frames: deque, odom: np.ndarray,
                      coverage_frac: float, phase: int) -> np.ndarray:
    """Stacked lidar + odometry + [coverage fraction, phase flag]."""
    parts = list(frames)
    parts.append(np.asarray(odom, dtype=np.float32))
    parts.append(np.array([coverage_frac, float(phase)], dtype=np.float32))
    return np.concatenate(parts)


def _next_maze_name(names: List[str], cursor: int, rng: random.Random,
                    mode: str) -> Tuple[str, int]:
    """Pure maze-selection helper: returns ``(name, new_cursor)``.

    ``round_robin`` walks ``names`` in order (``cursor`` increments mod n);
    ``random`` draws uniformly via ``rng`` and leaves the cursor untouched.
    Raises ValueError for any other ``mode``.
    """
    if mode == "random":
        return names[rng.randrange(len(names))], cursor
    if mode == "round_robin":
        name = names[cursor % len(names)]
        return name, (cursor + 1) % len(names)
    raise ValueError(f"unknown maze selection mode: {mode!r}")


class GazeboExploreEnv(gym.Env):
    """Multi-maze explore-then-exit env (v5) in the combined Gazebo world.

    Per-episode maze selection (``selection``):
      * ``"round_robin"`` — episodes walk ``maze_names`` in order, wrapping
        mod n (default, ``config.EXPL_V5_SELECTION``).
      * ``"random"`` — uniform draw each episode.

    ``reset(options={"maze_name": "sigma_1"})`` forces one specific maze:
    the name is validated against ``maze_names`` and the round-robin cursor
    is NOT advanced (overrides are out-of-band and must not disturb the
    training cadence). ``reset(seed=s)`` re-seeds the RNG used by ``random``
    selection.
    """

    metadata = {"render_modes": []}

    def __init__(self, robot_id: int = 1, seed: Optional[int] = None,
                 maze_names: Optional[List[str]] = None,
                 selection: Optional[str] = None,
                 world_name: Optional[str] = None,
                 prebuild: bool = False,
                 stall_limit: Optional[int] = None,
                 roam_bonus: Optional[float] = None,
                 action_mode: Optional[str] = None):
        super().__init__()
        self.robot_id = robot_id

        self.observation_space = spaces.Box(
            low=-1.0, high=1.0, shape=(C.EXPL_STATE_DIM,), dtype=np.float32)
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(2,), dtype=np.float32)

        # ── Maze registry (v5): which mazes, where, how to pick ─────
        self._maze_names: List[str] = (
            list(maze_names) if maze_names is not None
            else MR.sorted_registry_names())
        if not self._maze_names:
            raise ValueError("maze_names is empty — nothing to train on")
        self._selection = (selection if selection is not None
                           else C.EXPL_V5_SELECTION)
        if self._selection not in ("round_robin", "random"):
            raise ValueError(
                f"selection must be 'round_robin' or 'random', "
                f"got {self._selection!r}")
        self._world_name = (world_name if world_name is not None
                            else C.MAZE_MULTI_WORLD_NAME)

        self._registry = MR.load_registry()
        unknown = [n for n in self._maze_names if n not in self._registry]
        if unknown:
            raise ValueError(
                f"unknown maze(s) {unknown} — known: "
                f"{', '.join(sorted(self._registry))}")
        # Grid slots must match the combined-world SDF layout, which is
        # generated from the FULL sorted registry (generate_maze_sdf_multi).
        # Lookup is therefore done on the full grid even when maze_names is
        # a subset; identical to multi_placement(maze_names) by default.
        self._placements = MR.multi_placement()
        self._bundles: Dict[str, Dict[str, Any]] = {}
        self._maze_cursor = 0
        self._rng = random.Random(seed)

        # Pre-select the first maze so _maze_name/_spec are never None (the
        # first bundle is therefore built eagerly; the rest lazily on first
        # use unless prebuild=True).
        self._maze_name: str = self._maze_names[0]
        self._bundle = self._bundle_for(self._maze_name)
        self._spec = self._bundle["spec"]
        if prebuild:
            for name in self._maze_names:
                self._bundle_for(name)

        # Training-only helpers
        self._collision_thresh = compute_collision_thresholds()
        self._stuck = StuckTracker(
            C.EXPL_STUCK_WINDOW, C.EXPL_STUCK_MIN_DISP, C.EXPL_STUCK_PENALTY)
        self._stall_limit = (int(stall_limit) if stall_limit is not None
                             else C.EXPL_STALL_LIMIT)
        self._roam_bonus = (float(roam_bonus) if roam_bonus is not None
                            else C.EXPL_ROAM_BONUS)
        self._stall = StallMonitor(
            C.EXPL_STUCK_WINDOW, C.EXPL_STUCK_MIN_DISP, self._stall_limit)
        self._action_mode = (str(action_mode) if action_mode is not None
                             else C.EXPL_ACTION_MODE)
        self._shield_lock = ShieldLockMonitor(
            C.EXPL_SHIELD_STREAK, C.EXPL_SHIELD_LOCK_FRAC)
        self._prev_override = False
        self._last_escape_sign = 0
        self._frames: deque = deque(maxlen=4)

        # Sensor state (guarded by _lock)
        self._lock = threading.Lock()
        self._scan: Optional[np.ndarray] = None
        self._pending_action: Optional[np.ndarray] = None
        self._last_action_scale = 1.0
        self._last_safety_override = False
        # v9 slew-limiter state: the (forward, turn) actually commanded last
        # step, and the change the limiter allowed (priced by the reward).
        # Step on which PHASE_EXIT began, or None while still exploring;
        # episode_step_limit turns it into this episode's step cap.
        self._exit_entered_at: Optional[int] = None
        self._prev_twist: Tuple[float, float] = (0.0, 0.0)
        self._twist_delta: Tuple[float, float] = (0.0, 0.0)
        self._send_seq0: Optional[Tuple[int, int]] = None
        self._send_t = 0.0
        self._scan_seq = 0
        self._ray_angles: Optional[np.ndarray] = None
        self._pose_xy = self._spec.start_xy_world
        self._pose_yaw = self._spec.start_yaw
        self._pose_seq = 0

        # Episode state
        self._step_count = 0
        self._stale_steps = 0
        self._phase = PHASE_EXPLORE
        self._prev_xy = self._spec.start_xy_world
        self._prev_exit_d: Optional[float] = None
        self._prev_zone_d: Optional[float] = None
        self._zone_target = -1
        self._ep_idx = 0

        if not rclpy.ok():
            rclpy.init()
        self.node = Node(f"explore_env_node_{robot_id}")

        qos = rclpy.qos.QoSProfile(
            reliability=rclpy.qos.ReliabilityPolicy.BEST_EFFORT,
            history=rclpy.qos.HistoryPolicy.KEEP_LAST, depth=1)
        self.node.create_subscription(
            LaserScan, f"/scan{robot_id}", self._cb_scan, qos)
        self.node.create_subscription(
            Pose, f"/model/robot_{robot_id}/pose", self._cb_pose, qos)

        self._wheel_pubs = [
            self.node.create_publisher(
                Float64, f"/wheel_{w}_{robot_id}", 10)
            for w in WHEEL_ORDER
        ]

        # Latched 2D map for RViz (transient-local, like map_server)
        map_qos = rclpy.qos.QoSProfile(
            reliability=rclpy.qos.ReliabilityPolicy.RELIABLE,
            durability=rclpy.qos.DurabilityPolicy.TRANSIENT_LOCAL,
            history=rclpy.qos.HistoryPolicy.KEEP_LAST, depth=1)
        self._map_pub = self.node.create_publisher(
            OccupancyGrid, "/map", map_qos)

        # One callback worker per robot is enough: callbacks only update a
        # small locked sensor snapshot. Two workers per env created 26 DDS
        # executor threads for the 13-robot trainer and collapsed Gazebo RTF.
        self.executor = MultiThreadedExecutor(num_threads=1)
        self.executor.add_node(self.node)
        self._spin_thread = threading.Thread(
            target=self.executor.spin, daemon=True)
        self._spin_thread.start()

    # ── Per-maze artifacts ───────────────────────────────────────

    def _bundle_for(self, name: str) -> Dict[str, Any]:
        """Lazily build (and cache) the artifact bundle of maze ``name``.

        The spec is re-placed at its combined-world grid slot so all
        artifacts (distance field, mapper, zone coverage) are
        placement-consistent with the ``nhom8_maze_multi`` world.
        """
        bundle = self._bundles.get(name)
        if bundle is None:
            spec = self._registry[name].place_at(self._placements[name])
            bundle = {
                "spec": spec,
                "field": MR.build_distance_field(spec),
                "mapper": MR.build_mapper(spec),
                "zone": MR.build_zone_coverage(spec),
                # "zone_fields" is filled in on first use: 25 Dijkstras cost
                # ~1 s and 3 MB per maze, and prebuild=True would otherwise
                # pay that for every maze in the group before the first step.
                "zone_fields": None,
            }
            self._bundles[name] = bundle
        return bundle

    def _set_maze(self, name: str) -> None:
        """Activate maze ``name`` for the upcoming episode."""
        self._maze_name = name
        self._bundle = self._bundle_for(name)
        self._spec = self._bundle["spec"]

    @property
    def current_maze(self) -> str:
        """Name of the maze selected for the current (or next) episode."""
        return self._maze_name

    @property
    def _mapper(self):
        """Occupancy mapper of the current maze (v4 attribute compat)."""
        return self._bundle["mapper"]

    @property
    def _zone(self):
        """Zone-coverage tracker of the current maze."""
        return self._bundle["zone"]

    @property
    def _field(self):
        """Geodesic distance-to-exit field of the current maze."""
        return self._bundle["field"]

    @property
    def _zone_fields(self) -> ZoneDistanceFields:
        """Per-zone geodesic fields of the current maze (built on first use)."""
        fields = self._bundle["zone_fields"]
        if fields is None:
            fields = ZoneDistanceFields.from_spec(self._bundle["spec"])
            self._bundle["zone_fields"] = fields
        return fields

    def _unvisited(self) -> FrozenSet[int]:
        return frozenset(range(self._zone.n_zones)) - self._zone.visited

    # ── ROS callbacks ────────────────────────────────────────────

    def _cb_scan(self, msg: LaserScan) -> None:
        ranges = np.asarray(msg.ranges, dtype=np.float32)
        ranges = np.where(np.isfinite(ranges) & (ranges > 0.0),
                          ranges, C.LIDAR_MAX)
        ranges = np.clip(ranges, C.LIDAR_MIN, C.LIDAR_MAX)
        if ranges.shape[0] != C.N_RAYS:  # defensive resampling
            idx = (np.arange(C.N_RAYS) * ranges.shape[0]) // C.N_RAYS
            ranges = ranges[idx]
        with self._lock:
            self._scan = ranges
            self._scan_seq += 1
            if self._ray_angles is None and msg.angle_increment != 0.0:
                self._ray_angles = (
                    msg.angle_min
                    + np.arange(ranges.shape[0]) * msg.angle_increment
                ).astype(np.float32)

    def _cb_pose(self, msg: Pose) -> None:
        with self._lock:
            self._pose_xy = (msg.position.x, msg.position.y)
            self._pose_yaw = 2.0 * math.atan2(msg.orientation.z,
                                              msg.orientation.w)
            self._pose_seq += 1

    # ── Sensor access ────────────────────────────────────────────

    def _wait_fresh(self, timeout: float = 5.0) -> None:
        with self._lock:
            scan0, pose0 = self._scan_seq, self._pose_seq
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            with self._lock:
                if self._scan_seq > scan0 and self._pose_seq > pose0:
                    return
            time.sleep(0.02)
        raise RuntimeError(
            f"No fresh /scan{self.robot_id} + pose within {timeout}s. "
            "Is Gazebo running (unpaused) and spawn_robot_explore.sh active?")

    def _wait_fresh_step(self, timeout: Optional[float] = None) -> None:
        """Wait for a NEW scan+pose pair, at least ``C.EXPL_DT`` wall time.

        Gazebo's low real-time factor means the 10 Hz publishers lag the
        wall clock; a fixed sleep would consume stale sensor data. This
        polls for genuinely new data (seq-based) instead, but never waits
        longer than ``timeout``. A timeout stops this robot and aborts the
        step so stale observations never enter the replay buffer.
        """
        timeout = (C.EXPL_SENSOR_TIMEOUT if timeout is None
                   else float(timeout))
        if self._send_seq0 is not None:
            scan0, pose0 = self._send_seq0
            start = self._send_t
            self._send_seq0 = None
        else:
            with self._lock:
                scan0, pose0 = self._scan_seq, self._pose_seq
            start = time.monotonic()
        deadline = start + timeout
        while time.monotonic() < deadline:
            with self._lock:
                fresh = (self._scan_seq > scan0
                         and self._pose_seq > pose0)
            if fresh and time.monotonic() - start >= C.EXPL_DT:
                return
            time.sleep(0.005)
        self._stale_steps += 1
        self._stop_wheels()
        raise RuntimeError(
            f"robot_{self.robot_id}: no fresh scan+pose within {timeout}s; "
            "stopped instead of training on stale sensors")

    def _snapshot(self) -> Tuple[np.ndarray, Tuple[float, float], float]:
        with self._lock:
            if self._scan is None:
                raise RuntimeError(
                    f"No LaserScan received on /scan{self.robot_id} — "
                    "check the ros_gz bridge.")
            return (self._scan.copy(), self._pose_xy, self._pose_yaw)

    def _ray_angles_or_default(self) -> np.ndarray:
        with self._lock:
            if self._ray_angles is not None:
                return self._ray_angles
        return (np.arange(C.N_RAYS) * (2.0 * math.pi / C.N_RAYS)
                ).astype(np.float32)

    def _odom_vector(self, xy: Tuple[float, float],
                     yaw: float) -> np.ndarray:
        # x/y normalized to the CURRENT maze's world bbox → [−1, 1].
        # norm_bounds_world returns (xmin, ymin, width, height) with
        # width/height == xmax−xmin / ymax−ymin.
        xmin, ymin, w, h = self._spec.norm_bounds_world
        nx = float(np.clip((xy[0] - xmin) / max(w, 1e-9) * 2.0 - 1.0,
                           -1.0, 1.0))
        ny = float(np.clip((xy[1] - ymin) / max(h, 1e-9) * 2.0 - 1.0,
                           -1.0, 1.0))
        return np.array([nx, ny, math.cos(yaw), math.sin(yaw)], dtype=np.float32)

    def _stack_obs(self, scan: np.ndarray, xy: Tuple[float, float],
                   yaw: float) -> np.ndarray:
        frame = (scan / C.LIDAR_MAX).astype(np.float32)
        self._frames.append(frame)
        while len(self._frames) < 4:
            self._frames.appendleft(frame.copy())
        cov = self._zone.n_cells / float(self._zone.n_zones)
        return build_explore_obs(self._frames, self._odom_vector(xy, yaw),
                                 cov, self._phase)

    # ── Actuation / teleport ─────────────────────────────────────

    def _publish_wheels(self, wheel_vels: np.ndarray) -> None:
        for pub, w in zip(self._wheel_pubs, wheel_vels):
            pub.publish(Float64(data=float(w)))

    def _apply_action(self, action: np.ndarray) -> np.ndarray:
        action = np.clip(np.asarray(action, dtype=np.float32), -1.0, 1.0)
        with self._lock:
            scan = self._scan.copy() if self._scan is not None else None
        wheels = map_explore_action(action, self._action_mode)
        # v9: rate-limit the (forward, turn) command before anything else
        # touches it. SAC resamples both channels i.i.d. every control step,
        # which on its own swung the heading 9 deg per step (p90 30 deg) and
        # reversed the turn on 17% of steps — the weave seen in Gazebo.
        prev = self._prev_twist
        twist = slew_limit(prev, twist_of_wheels(wheels))
        self._twist_delta = (twist[0] - prev[0], twist[1] - prev[1])
        wheels = wheels_of_twist(*twist)
        if scan is None:
            self._last_action_scale = 1.0
            self._last_safety_override = False
        else:
            wheels, self._last_action_scale, self._last_safety_override = (
                explore_shield(wheels, scan, self._collision_thresh,
                               engaged=self._last_safety_override,
                               escape_sign=self._last_escape_sign))
            # v12: remember which way the escape steered so the next call can
            # hold it (hysteresis on the steering sign, not just the throttle
            # band). Cleared whenever the shield is not overriding.
            if self._last_safety_override:
                esc_turn = 0.5 * (float(wheels[1]) - float(wheels[0]))
                self._last_escape_sign = (
                    1 if esc_turn > 0.0 else -1 if esc_turn < 0.0 else 0)
            else:
                self._last_escape_sign = 0
        # Track what the WHEELS were actually given, shield included, so the
        # limiter ramps from the real state rather than from a command the
        # shield overrode — otherwise every shield release is a step change.
        self._prev_twist = twist_of_wheels(wheels)
        self._publish_wheels(map_wheel_action(wheels, diff_drive=True)
                             * C.EXPL_W_MAX)
        return wheels

    def pre_send(self, action: np.ndarray) -> None:
        """Publish this step's wheel command WITHOUT waiting for sensors.

        Lets a vec-env issue every robot's command back-to-back so each one
        integrates for a single control period. Stepping N envs sequentially
        instead leaves each command active for N control periods (the other
        envs' blocking sensor waits), so per-step travel scales with N.
        The freshness baseline is latched here, at send time, so the paired
        ``step()`` still consumes a scan taken AFTER the command.
        """
        with self._lock:
            self._send_seq0 = (self._scan_seq, self._pose_seq)
        self._send_t = time.monotonic()
        self._pending_action = self._apply_action(action)

    def hold(self) -> None:
        """Zero the wheels so a latched command stops accumulating travel."""
        self._stop_wheels()

    def _stop_wheels(self) -> None:
        self._publish_wheels(np.zeros(len(self._wheel_pubs)))

    def _teleport_start(self) -> None:
        sx, sy = self._spec.start_xy_world
        yaw = self._spec.start_yaw
        qz = math.sin(yaw / 2)
        qw = math.cos(yaw / 2)
        req = (f"name: 'robot_{self.robot_id}' "
               f"position {{ x: {sx} y: {sy} "
               f"z: 0.024 }} "
               f"orientation {{ x: 0.0 y: 0.0 z: {qz:.6f} w: {qw:.6f} }}")
        try:
            result = subprocess.run(
                ["gz", "service", "-s",
                 f"/world/{self._world_name}/set_pose",
                 "--reqtype", "gz.msgs.Pose", "--reptype", "gz.msgs.Boolean",
                 "--req", req, "--timeout", "2000"],
                capture_output=True, timeout=4.0)
            if result.returncode != 0:
                self.node.get_logger().warn(
                    f"set_pose failed: {result.stderr.decode(errors='ignore')}")
        except FileNotFoundError:
            raise RuntimeError("`gz` CLI not found — source the Gazebo env.")
        except subprocess.TimeoutExpired:
            self.node.get_logger().warn("set_pose service call timed out")

    # ── Map publishing / saving ──────────────────────────────────

    def _publish_map(self) -> None:
        grid = self._mapper.occupancy_array()
        msg = OccupancyGrid()
        msg.header.frame_id = "map"
        msg.header.stamp = self.node.get_clock().now().to_msg()
        msg.info.resolution = self._mapper.res
        msg.info.width = self._mapper.n
        msg.info.height = self._mapper.n
        msg.info.origin.position.x = self._mapper.ox
        msg.info.origin.position.y = self._mapper.oy
        msg.info.origin.orientation.w = 1.0
        msg.data = grid.flatten().tolist()
        self._map_pub.publish(msg)

    def _save_map(self) -> None:
        try:
            C.EXPL_MAP_DIR.mkdir(parents=True, exist_ok=True)
            stem = f"ep_{self._ep_idx:04d}_{self._maze_name}"
            self._mapper.save_png(C.EXPL_MAP_DIR / f"{stem}.png")
            self._mapper.save_png(C.EXPL_MAP_DIR / "map_latest.png")
        except Exception as exc:  # map saving must never kill training
            self.node.get_logger().warn(f"map save failed: {exc}")

    # ── Gymnasium API ────────────────────────────────────────────

    def reset(self, *, seed: Optional[int] = None,
              options: Optional[Dict[str, Any]] = None
              ) -> Tuple[np.ndarray, Dict[str, Any]]:
        super().reset(seed=seed)
        self._pending_action = None
        self._send_seq0 = None
        if seed is not None:
            self._rng = random.Random(seed)

        # Per-episode maze selection (see class docstring for semantics).
        maze_name = options.get("maze_name") if options else None
        if maze_name is not None:
            if maze_name not in self._maze_names:
                raise ValueError(
                    f"unknown maze {maze_name!r} — known: "
                    f"{', '.join(self._maze_names)}")
            self._set_maze(maze_name)   # explicit override: cursor untouched
        else:
            name, self._maze_cursor = _next_maze_name(
                self._maze_names, self._maze_cursor, self._rng,
                self._selection)
            self._set_maze(name)

        # Verify the teleport actually took. `set_pose` reports success on
        # ACCEPTING the request, so a hung server answers OK and never moves
        # the robot; reset would then hand back the stale pose as a legal
        # start and the next step would score an instant -30 out-of-bounds
        # terminal straight into the replay buffer. Measured: 100% of the
        # out_of_bounds deaths in both campaigns were exactly this, never a
        # real escape (ep_len == 1, pose 7-15 m outside every maze).
        target = self._spec.start_xy_world
        for attempt in range(C.EXPL_TELEPORT_TRIES):
            self._stop_wheels()
            self._teleport_start()
            time.sleep(C.EXPL_SETTLE_SEC)
            self._wait_fresh()
            scan, xy, yaw = self._snapshot()
            if teleport_landed(target, xy):
                break
            self.node.get_logger().warn(
                f"robot_{self.robot_id}: teleport to {self._maze_name} "
                f"did not take (asked {target}, got {xy}) — "
                f"attempt {attempt + 1}/{C.EXPL_TELEPORT_TRIES}")
        else:
            raise SimUnresponsiveError(
                f"robot_{self.robot_id}: {C.EXPL_TELEPORT_TRIES} teleports to "
                f"{self._maze_name} at {target} all left the robot at {xy}. "
                f"The Gazebo server is accepting set_pose without applying "
                f"it — restart the simulator rather than training on this.")
        self._frames.clear()
        self._mapper.reset()
        self._zone.reset(xy)
        self._stuck.reset(xy)
        self._stall.reset(xy)
        self._shield_lock.reset()
        self._prev_override = False
        self._last_escape_sign = 0
        # The robot is teleported and stationary, so the limiter must ramp
        # from a standstill; carrying the last episode's command over would
        # charge the first step of the new episode for a phantom reversal.
        self._prev_twist = (0.0, 0.0)
        self._twist_delta = (0.0, 0.0)
        self._last_safety_override = False
        self._step_count = 0
        self._phase = PHASE_EXPLORE
        self._exit_entered_at = None
        self._prev_xy = xy
        self._prev_exit_d = None
        self._prev_zone_d = None
        self._zone_target = -1
        self._ep_idx += 1
        self._publish_map()

        obs = self._stack_obs(scan, xy, yaw)
        info: Dict[str, Any] = {
            "maze": self._maze_name,
            "pos": xy, "phase": self._phase,
            "coverage_cells": self._zone.n_cells,
            "coverage_frac": self._zone.n_cells / float(self._zone.n_zones),
            "map_pct": self._mapper.coverage(),
        }
        return obs, info

    def step(self, action: np.ndarray
             ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        if self._pending_action is not None:
            action, self._pending_action = self._pending_action, None
        else:
            action = self._apply_action(action)
        self._wait_fresh_step()
        scan, xy, yaw = self._snapshot()
        self._step_count += 1

        # Ground-truth linear speed from consecutive poses
        speed = math.dist(xy, self._prev_xy) / C.EXPL_DT

        # 2D map update (the maze walls are the mapped obstacles)
        self._mapper.update(xy[0], xy[1], yaw, scan,
                            self._ray_angles_or_default())

        ray_angles = self._ray_angles_or_default()
        min_scan = float(scan.min())
        front_clear = front_clearance(scan, ray_angles)
        # Per-ray margin to the termination test; min < 0 IS the collision
        # condition (identical to np.any(scan < thresh)). The margin and its
        # arg-min are logged so deaths can be attributed to a ray direction.
        slack = scan - self._collision_thresh
        collided = bool(slack.min() < 0.0)

        # Boundary classification: exit opening vs. elsewhere (registry-
        # driven; per-maze opening + placement).
        exited = MR.exit_crossed(self._spec, self._spec.exit_opening, xy)
        out_of_bounds = MR.out_of_bounds(self._spec, xy)

        info: Dict[str, Any] = {
            "maze": self._maze_name,
            "pos": xy, "phase": self._phase, "min_scan": min_scan,
            "yaw": yaw,
            "action_scale": self._last_action_scale,
            "safety_override": self._last_safety_override,
            "speed": speed, "front_clear": front_clear,
            "coverage_cells": self._zone.n_cells,
            "coverage_frac": self._zone.n_cells / float(self._zone.n_zones),
            "map_pct": self._mapper.coverage(),
            "min_slack": float(slack.min()),
            "danger_ray": int(np.argmin(slack)),
            "ep_len": self._step_count,
        }

        reward = C.EXPL_R_TIME
        terminated = False

        if exited and self._phase == PHASE_EXPLORE:
            reward += C.EXPL_R_EXIT_EARLY
            terminated = True
            info["early_exit"] = True
        elif exited and self._phase == PHASE_EXIT:
            reward += C.EXPL_R_EXIT
            terminated = True
            info["success"] = True
        elif out_of_bounds:
            # Sealed borders make this unreachable except via the gap;
            # treat as a failure (defensive, mirrors v3).
            reward += C.EXPL_R_COLLISION
            terminated = True
            info["out_of_bounds"] = True
        elif collided:
            reward += C.EXPL_R_COLLISION
            terminated = True
            info["collision"] = True
        else:
            # Coverage bonus (tracker returns the bonus for a brand-new zone)
            if self._zone.update(xy) > 0.0:
                reward += C.EXPL_R_CELL
                if (self._zone.n_cells >= self._zone.n_zones
                        and self._phase == PHASE_EXPLORE):
                    self._phase = PHASE_EXIT
                    self._exit_entered_at = self._step_count
                    info["phase"] = self._phase
                    info["map_complete"] = True
                    reward += C.EXPL_R_MAP_DONE
                    self._prev_exit_d = self._field.distance(*xy)
            # EXPLORE phase: potential-based shaping on the geodesic
            # distance to the NEAREST UNVISITED zone. Without it the only
            # signal pointing at an unexplored zone is the +8 collected on
            # arrival, discounted by 0.997^(steps away) and worth nothing at
            # all until the agent blunders into the zone by chance — which is
            # why 240 episodes produced 1.5/25 zones. The target-change guard
            # skips the step on which the distance jumps discontinuously
            # (a zone was ticked off); a mere tie-flip between two equidistant
            # zones is continuous, so skipping it costs nothing either.
            if self._phase == PHASE_EXPLORE:
                target, zone_d = self._zone_fields.nearest(
                    xy[0], xy[1], self._unvisited())
                reward += zone_approach_shaping(
                    self._prev_zone_d, zone_d, C.EXPL_ZONE_POT_SCALE,
                    retargeted=(target != self._zone_target))
                self._prev_zone_d, self._zone_target = zone_d, target
                info["zone_target"] = target
                info["zone_dist"] = zone_d
            # EXIT phase: potential-based shaping on geodesic distance
            if self._phase == PHASE_EXIT and self._prev_exit_d is not None:
                d = self._field.distance(*xy)
                reward += C.EXPL_EXIT_POT_SCALE * (self._prev_exit_d - d)
                self._prev_exit_d = d
            # "Slow down near a wall" rule. v6: the gate is min per-ray
            # slack, not front clearance — 87.4% of deaths come from the
            # side/rear rays that front_clearance cannot see, and it is the
            # SAME quantity the termination test uses (slack.min() < 0), so
            # the dense penalty now warns about the actual failure mode.
            safety = max(float(slack.min()), 0.0)
            reward += safe_speed_penalty(
                safety, speed, C.EXPL_SAFE_MARGIN, C.EXPL_SAFE_SPEED_SCALE)
            # Bold-roaming bonus: brisk motion in open space earns; creeping
            # earns nothing (paired with the stall rule below, standing or
            # crawling can never be the safe choice).
            if safety >= C.EXPL_SAFE_MARGIN:
                reward += self._roam_bonus * speed
            # v9: price the per-step command change. The slew limiter caps
            # how fast the wheels may move, but a policy that saturates the
            # cap every step still weaves at the cap; this is what makes
            # holding a heading strictly better than sawing across it.
            reward += action_rate_penalty((0.0, 0.0), self._twist_delta,
                                          C.EXPL_R_SMOOTH)
            # Anti-stall: immediate penalty, then hard termination — standing
            # still must never be the discounted-safe alternative to acting.
            stuck_pen = self._stuck.update(xy)
            reward += stuck_pen
            info["stuck"] = bool(stuck_pen < 0.0)
            # v7: an in-place shield turn is a legitimate manoeuvre, not a
            # stall. StallMonitor measures NET displacement over a 20-step
            # window, so it kills a robot that is rotating to find an
            # opening: the scripted probe lost a 6/25-zone episode that way
            # at step 333. The lock cap keeps "hide inside the shield
            # forever" from becoming the optimal policy.
            #
            # v12: the cap is a SLIDING window (ShieldLockMonitor), not a
            # consecutive run. A staircase corridor releases the shield for
            # one step every few, which zeroed the old consecutive counter so
            # it never reached EXPL_SHIELD_STREAK -- the sigma_3 episode at
            # ts 419672 rode the full 1500-step timeout with the shield
            # engaged 88% of the time. The window survives those blips.
            lock = self._shield_lock.update(self._last_safety_override)
            if self._last_safety_override:
                stalled = lock
            else:
                # Re-seed the displacement window when the policy takes back
                # over so a stale pre-override position cannot fire a spurious
                # stall on the very next step.
                if self._prev_override:
                    self._stall.reset(xy)
                stalled = self._stall.update(xy) or lock
            self._prev_override = self._last_safety_override
            if stalled:
                reward += C.EXPL_R_COLLISION
                terminated = True
                info["stall"] = True

        truncated = False
        limit = episode_step_limit(self._exit_entered_at)
        if not terminated and self._step_count >= limit:
            truncated = True
            info["timeout"] = True

        if self._step_count % C.EXPL_MAP_EVERY == 0:
            self._publish_map()

        obs = self._stack_obs(scan, xy, yaw)
        info["coverage_cells"] = self._zone.n_cells
        info["coverage_frac"] = self._zone.n_cells / float(self._zone.n_zones)
        self._prev_xy = xy
        if terminated or truncated:
            self._stop_wheels()
            self._save_map()
        return obs, float(reward), terminated, truncated, info

    def scripted_action(self) -> np.ndarray:
        """Demonstrator action: steer down the ACTIVE geodesic field.

        EXPLORE heads for the nearest unvisited zone, EXIT for the opening.
        Nothing here runs inside the learning loop — the CLAUDE.md pure-RL
        mandate keeps planners opt-in, and this is reached only through
        ``--bootstrap-scripted``, which seeds the replay buffer before
        training starts (SACfD) and is never consulted again. Safety is not
        this function's job: the returned action goes through
        ``_apply_action`` -> ``explore_shield`` like any policy action.
        """
        with self._lock:
            (x, y), yaw = self._pose_xy, self._pose_yaw
        if self._phase == PHASE_EXIT:
            ux, uy = descent_direction(self._field.dist, self._field.origin,
                                       self._field.res, x, y)
        else:
            target, _ = self._zone_fields.nearest(x, y, self._unvisited())
            ux, uy = self._zone_fields.descent_direction(target, x, y)
        if ux == 0.0 and uy == 0.0:
            # No finite cell on the search rim (deep inside the inflation
            # band). Drive straight and let the shield sort it out.
            return unmap_explore_action(1.0, 0.0, self._action_mode)
        return steer_to_heading(ux, uy, yaw, self._action_mode)

    def close(self) -> None:
        try:
            self._stop_wheels()
        finally:
            self.executor.shutdown()
            self.node.destroy_node()


def make_explore_env(robot_id: int = 1, seed: Optional[int] = None,
                     maze_names: Optional[List[str]] = None,
                     selection: Optional[str] = None,
                     world_name: Optional[str] = None,
                     prebuild: bool = False,
                     stall_limit: Optional[int] = None,
                     roam_bonus: Optional[float] = None,
                     action_mode: Optional[str] = None) -> GazeboExploreEnv:
    """Factory shared by training and evaluation scripts."""
    return GazeboExploreEnv(robot_id=robot_id, seed=seed,
                            maze_names=maze_names, selection=selection,
                            world_name=world_name, prebuild=prebuild,
                            stall_limit=stall_limit, roam_bonus=roam_bonus,
                            action_mode=action_mode)
