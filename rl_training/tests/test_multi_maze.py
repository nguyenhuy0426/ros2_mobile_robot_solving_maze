#!/usr/bin/env python3
"""
test_multi_maze.py — Tests for the multi-maze registry stack (v5).

Covers the 13 mazes decoded from mazes_png/*.png: the JSON definitions in
worlds/maze_defs/, the MazeSpec registry invariants, standalone SDFs
(worlds/nhom8_maze_<name>.sdf), the combined world (worlds/nhom8_maze_multi.sdf),
the OBB wall geometry, zone coverage, geometric exit/bounds predicates and the
geodesic distance fields. No ROS/Gazebo required.
"""

import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest

WS_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(WS_ROOT / "scripts"))

from generate_maze_sdf_multi import verify_maze_seal, verify_sdf_file  # noqa: E402
import rl_training.maze_registry as MR  # noqa: E402
from rl_training import config as C  # noqa: E402
from rl_training.maze_field import wall_from_segment  # noqa: E402

WORLDS_DIR = Path(C.WS_ROOT) / "worlds"
MULTI_SDF = Path(C.MAZE_MULTI_SDF)
REQUIRED_JSON_KEYS = {
    "name", "family", "walls", "openings", "start_xy", "start_yaw_deg",
    "zone_labels_shape", "zone_labels_b64", "zone_res", "zone_centroids",
    "zone_start",
}


@pytest.fixture(scope="module")
def registry():
    return MR.load_registry()


def _names(registry):
    return MR.sorted_registry_names(registry)


# ══════════════════════════════════════════════════════════════════════════
# JSON definitions on disk
# ══════════════════════════════════════════════════════════════════════════

def test_maze_defs_complete(registry):
    names = _names(registry)
    assert len(names) == 13
    jsons = sorted((Path(C.MAZE_DEFS_DIR)).glob("*.json"))
    assert [p.stem for p in jsons] == names
    for path in jsons:
        data = json.loads(path.read_text())
        assert REQUIRED_JSON_KEYS <= set(data), f"{path.name}: missing keys"
        assert data["name"] == path.stem
        assert isinstance(data["walls"], list) and data["walls"]
        for w in data["walls"]:
            assert isinstance(w, list) and len(w) == 4
            assert all(isinstance(v, (int, float)) for v in w), \
                f"{path.name}: wall {w} is not 4 floats"
        ops = data["openings"]
        assert isinstance(ops, list) and len(ops) == 2
        assert ops[0]["role"] == "entrance" and ops[1]["role"] == "exit"


# ══════════════════════════════════════════════════════════════════════════
# Registry spec metadata + placement
# ══════════════════════════════════════════════════════════════════════════

def test_registry_spec_metadata(registry):
    for name in _names(registry):
        spec = registry[name]
        data = json.loads(
            (Path(C.MAZE_DEFS_DIR) / f"{name}.json").read_text())
        assert spec.name == data["name"] == name
        assert spec.family in {"delta", "ortho", "sigma"}
        bw = spec.wall_bbox[2] - spec.wall_bbox[0]
        bh = spec.wall_bbox[3] - spec.wall_bbox[1]
        assert 0 < bw <= 3.93 + 0.05, f"{name}: bbox_w={bw:.3f}"
        assert 0 < bh <= 3.93 + 0.05, f"{name}: bbox_h={bh:.3f}"
        assert len(spec.walls_local) > 0
        assert spec.placement_offset == tuple(C.MAZE_CENTER_XY)


def test_openings_invariants(registry):
    for name in _names(registry):
        spec = registry[name]
        assert len(spec.openings) == 2
        entrance, exit_ = spec.openings
        assert entrance.role == "entrance"
        assert exit_.role == "exit"
        assert entrance.border == "north"
        assert exit_.border == "south"
        assert spec.exit_opening == exit_
        for op in spec.openings:
            assert 0.15 <= op.half_width <= 0.5, \
                f"{name}: {op.role} half_width={op.half_width}"
        bx0, by0, bx1, by1 = spec.wall_bbox
        assert abs(exit_.xy_local[1] - by0) <= 0.02, \
            f"{name}: exit not flush with ymin"
        assert bx0 <= entrance.xy_local[0] <= bx1, \
            f"{name}: entrance x outside bbox"
        assert entrance.xy_local[1] > exit_.xy_local[1]


def test_zone_labels_25(registry):
    for name in _names(registry):
        spec = registry[name]
        labels = spec.zone_labels
        assert labels.dtype == np.int16
        assert labels.ndim == 2
        uniq = set(int(v) for v in np.unique(labels))
        assert -1 in uniq
        assert uniq - {-1} == set(range(MR.N_ZONES)), \
            f"{name}: zone ids {sorted(uniq - {-1})}"
        assert isinstance(spec.zone_start, int)
        assert 0 <= spec.zone_start < MR.N_ZONES


def test_start_free_space(registry):
    for name in _names(registry):
        spec = registry[name]
        sx_l, sy_l = spec.start_xy_local
        bx0, by0, bx1, by1 = spec.wall_bbox
        assert bx0 <= sx_l <= bx1 and by0 <= sy_l <= by1, \
            f"{name}: local start outside wall bbox"
        sx, sy = spec.start_xy_world
        for w in spec.walls_world:
            assert not w.contains(sx, sy, C.ROBOT_RADIUS), \
                f"{name}: start inside an inflated wall"
        x0, y0, wd, ht = spec.norm_bounds_world
        assert x0 <= sx <= x0 + wd and y0 <= sy <= y0 + ht, \
            f"{name}: start outside norm bounds"


# ══════════════════════════════════════════════════════════════════════════
# OBB wall geometry
# ══════════════════════════════════════════════════════════════════════════

def test_obb_contains_rotated(registry):
    t = C.EXPL_WALL_T
    sqrt3 = math.sqrt(3.0)
    w60 = wall_from_segment(0.0, 0.0, 1.0, sqrt3, t)
    w60n = wall_from_segment(0.0, 0.0, 1.0, -sqrt3, t)
    assert w60.yaw == pytest.approx(math.radians(60.0))
    assert w60n.yaw == pytest.approx(math.radians(-60.0))
    assert w60.half_length == pytest.approx(1.0)
    assert w60.half_thickness == pytest.approx(t / 2.0)

    mid = (0.5, sqrt3 / 2.0)
    mid_n = (0.5, -sqrt3 / 2.0)
    assert w60.contains(*mid)
    assert w60n.contains(*mid_n)

    c, s = math.cos(math.radians(60.0)), math.sin(math.radians(60.0))
    ux, uy = c, s
    vx, vy = -s, c

    def at(along, across):
        return (mid[0] + along * ux + across * vx,
                mid[1] + along * uy + across * vy)

    assert not w60.contains(*at(1.10, 0.0))
    assert not w60.contains(*at(1.10, 0.0), 0.06)
    assert not w60.contains(*at(1.03, 0.0))
    assert w60.contains(*at(1.03, 0.0), 0.06)
    assert not w60.contains(*at(0.0, 0.065))
    assert w60.contains(*at(0.0, 0.065), 0.06)
    assert not w60.contains(*at(0.0, -0.065))
    assert w60.contains(*at(0.0, -0.065), 0.06)

    moved = w60.translated(2.0, -1.0)
    assert (moved.cx, moved.cy) == pytest.approx(
        (w60.cx + 2.0, w60.cy - 1.0))
    assert moved.yaw == pytest.approx(w60.yaw)
    assert (moved.half_length, moved.half_thickness) == \
        pytest.approx((w60.half_length, w60.half_thickness))
    assert not moved.contains(*mid)
    assert moved.contains(mid[0] + 2.0, mid[1] - 1.0)

    xs = np.array([p[0] for p in
                   (at(0.0, 0.0), at(0.9, 0.0), at(1.2, 0.0),
                    at(0.0, 0.05), at(-1.05, 0.0), at(0.5, 0.5))])
    ys = np.array([p[1] for p in
                   (at(0.0, 0.0), at(0.9, 0.0), at(1.2, 0.0),
                    at(0.0, 0.05), at(-1.05, 0.0), at(0.5, 0.5))])
    arr = w60.contains_array(xs, ys)
    expected = np.array([w60.contains(x, y) for x, y in zip(xs, ys)])
    assert np.array_equal(arr, expected)
    assert arr[0] and arr[1] and not arr[2] and not arr[3] and not arr[4]

    yaws = [math.degrees(w.yaw) for w in registry["sigma_1"].walls_local]

    def mod180_dist(a, b):
        d = abs((a - b) % 180.0)
        return min(d, 180.0 - d)

    rotated = [a for a in yaws
               if min(mod180_dist(a, 60.0), mod180_dist(a, 120.0)) <= 10.0]
    assert rotated, "sigma_1 has no ~60°/120° walls"
    assert any(min(mod180_dist(a, 0.0), mod180_dist(a, 180.0)) > 10.0
               for a in yaws), "sigma_1 walls are all axis-aligned"


# ══════════════════════════════════════════════════════════════════════════
# Exit / bounds predicates
# ══════════════════════════════════════════════════════════════════════════

@pytest.mark.parametrize("name", ["delta_1", "ortho_1", "sigma_1"])
def test_exit_crossed_and_oob(registry, name):
    spec = registry[name]
    exit_ = spec.exit_opening
    bx0, by0, bx1, by1 = spec.wall_bbox_world
    ox, oy = spec.placement_offset
    gx = exit_.xy_local[0] + ox

    assert MR.exit_crossed(spec, exit_, (gx, by0 - 0.10))
    assert not MR.exit_crossed(spec, exit_, (gx, by0 + 0.10))
    assert not MR.exit_crossed(
        spec, exit_, (gx + exit_.half_width + 0.10, by0 - 0.10))

    cx, cy = (bx0 + bx1) / 2.0, (by0 + by1) / 2.0
    assert MR.out_of_bounds(spec, (bx0 - 0.10, cy))
    assert MR.out_of_bounds(spec, (cx, by1 + 0.10))
    assert not MR.out_of_bounds(spec, (cx, cy))


@pytest.mark.parametrize("name", ["delta_1", "ortho_1", "sigma_1"])
def test_leaving_through_the_exit_is_never_scored_out_of_bounds(registry, name):
    """A winning exit move must not be reachable as a -30 death.

    explore_env tests exit_crossed before out_of_bounds, so the two
    thresholds must not straddle: any point that is already out of bounds
    inside the exit gap has to count as crossed. When out_of_bounds used a
    tighter eps than exit_crossed, the 3 cm band between them turned the
    final step of a successful run into EXPL_R_COLLISION.
    """
    spec = registry[name]
    exit_ = spec.exit_opening
    _bx0, by0, _bx1, _by1 = spec.wall_bbox_world
    gx = exit_.xy_local[0] + spec.placement_offset[0]
    for depth in np.arange(0.0, 0.31, 0.005):
        xy = (gx, by0 - float(depth))
        if MR.out_of_bounds(spec, xy):
            assert MR.exit_crossed(spec, exit_, xy), \
                f"{name}: y={by0 - depth:.3f} is OOB but not yet exited"


# ══════════════════════════════════════════════════════════════════════════
# Zone coverage tracker
# ══════════════════════════════════════════════════════════════════════════

@pytest.mark.parametrize("name", ["delta_1", "sigma_1"])
def test_zone_coverage_class(registry, name):
    spec = registry[name]
    zc = MR.build_zone_coverage(spec)
    assert zc.n_zones == MR.N_ZONES == 25
    assert zc.n_cells == 0

    labels = spec.zone_labels
    res = spec.zone_res
    ox, oy = spec.placement_offset
    bx0, by0 = spec.bounds[0], spec.bounds[1]
    start_lbl = spec.zone_label_at_world(*spec.start_xy_world)
    assert start_lbl >= 0

    k = next(int(labels[iy, ix]) for iy, ix in np.argwhere(labels >= 0)
             if int(labels[iy, ix]) != start_lbl)
    iy, ix = next(map(tuple, np.argwhere(labels == k)))
    wx = bx0 + ix * res + ox
    wy = by0 + iy * res + oy
    assert spec.zone_label_at_world(wx, wy) == k
    assert zc.label_at(wx, wy) == k

    assert zc.update((wx, wy)) == pytest.approx(1.0)
    assert zc.update((wx, wy)) == pytest.approx(0.0)
    assert zc.n_cells == 1

    iy_n, ix_n = next(map(tuple, np.argwhere(labels < 0)))
    wxn = bx0 + ix_n * res + ox
    wyn = by0 + iy_n * res + oy
    assert spec.zone_label_at_world(wxn, wyn) == -1
    assert zc.update((wxn, wyn)) == pytest.approx(0.0)
    assert zc.update((wxn - 1.0, wyn - 1.0)) == pytest.approx(0.0)
    assert zc.n_cells == 1
    assert zc.n_cells / zc.n_zones == pytest.approx(1.0 / 25.0)

    zc.reset(spec.start_xy_world)
    assert zc.n_cells == 1
    assert zc.update((wx, wy)) == pytest.approx(1.0)
    assert zc.n_cells == 2


# ══════════════════════════════════════════════════════════════════════════
# Multi-maze placement grid
# ══════════════════════════════════════════════════════════════════════════

def test_multi_placement_grid(registry):
    names = _names(registry)
    placements = MR.multi_placement(names)
    assert sorted(placements) == names
    assert len(set(placements.values())) == len(names)
    half = (C.MAZE_MULTI_COLS - 1) / 2.0
    for i, name in enumerate(names):
        expected = ((i % C.MAZE_MULTI_COLS - half) * C.MAZE_MULTI_SPACING,
                    (i // C.MAZE_MULTI_COLS - half) * C.MAZE_MULTI_SPACING)
        assert placements[name] == pytest.approx(expected), name

    placed = {n: registry[n].place_at(placements[n]) for n in names}
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            ax0, ay0, ax1, ay1 = placed[a].wall_bbox_world
            bx0, by0, bx1, by1 = placed[b].wall_bbox_world
            dx = max(ax0 - bx1, bx0 - ax1)
            dy = max(ay0 - by1, by0 - ay1)
            sep = math.hypot(max(dx, 0.0), max(dy, 0.0))
            assert sep > 1.0, f"{a}/{b}: bbox separation {sep:.3f} m"

    single = MR.multi_placement(["sigma_1"])
    assert single == {"sigma_1": (-12.0, -12.0)}


# ══════════════════════════════════════════════════════════════════════════
# Seal proofs
# ══════════════════════════════════════════════════════════════════════════

def test_seal_all_mazes(registry):
    for name in _names(registry):
        spec = registry[name]
        seal = verify_maze_seal(spec)
        assert seal["ok"], f"{name}: leaks={seal['leak_samples']}"
        assert seal["leak_count"] == 0
        if spec.family == "ortho":
            assert seal["ortho_violations"] == 0, \
                f"{name}: {seal['ortho_violation_points']}"


# ══════════════════════════════════════════════════════════════════════════
# SDF round-trip
# ══════════════════════════════════════════════════════════════════════════

def test_sdf_roundtrip_standalone(registry):
    for name in _names(registry):
        spec = registry[name]
        sdf = WORLDS_DIR / f"{MR.standalone_world_name(name)}.sdf"
        assert sdf.is_file(), f"missing standalone world: {sdf}"
        result = verify_sdf_file(sdf, spec, spec.placement_offset)
        assert result["ok"], f"{name}: leaks={result['leak_samples']}"


def test_sdf_roundtrip_combined_representative(registry):
    assert MULTI_SDF.is_file(), f"missing combined world: {MULTI_SDF}"
    placements = MR.multi_placement(_names(registry))
    for name in ("delta_1", "ortho_1", "sigma_1"):
        offset = placements[name]
        result = verify_sdf_file(MULTI_SDF, registry[name], offset)
        assert result["ok"], \
            f"{name}@{offset}: leaks={result['leak_samples']}"


# ══════════════════════════════════════════════════════════════════════════
# Distance fields (slowest — keep last)
# ══════════════════════════════════════════════════════════════════════════

def test_distance_fields_reachable(registry):
    for name in _names(registry):
        spec = registry[name]
        field = MR.build_distance_field(spec)
        d = field.distance(*spec.start_xy_world)
        assert math.isfinite(d), f"{name}: d(start) not finite"
        assert d >= 1.0, f"{name}: d(start)={d:.3f} < 1.0"


# ══════════════════════════════════════════════════════════════════════════
# Entrance seal + spawn clearance
# ══════════════════════════════════════════════════════════════════════════

def _simulated_scan(walls, px, py, yaw):
    """Analytic 36-ray lidar scan of an OBB wall set (slab ray/OBB test).

    (px, py) is the ROBOT pose; rays leave from the lidar mount LIDAR_X_OFF
    ahead of it, because that is where compute_collision_thresholds() measures
    its ray-to-chassis distances from. Ray i points at yaw + i*(2pi/36) and
    ranges are clipped to [LIDAR_MIN, LIDAR_MAX], so the result can be
    compared against those thresholds exactly like a live scan is.
    """
    px, py = px + C.LIDAR_X_OFF * math.cos(yaw), py + C.LIDAR_X_OFF * math.sin(yaw)
    n = C.N_RAYS
    out = np.full(n, C.LIDAR_MAX, dtype=np.float64)
    for i in range(n):
        th = yaw + i * (2.0 * math.pi / n)
        dx, dy = math.cos(th), math.sin(th)
        best = C.LIDAR_MAX
        for w in walls:
            c, s = math.cos(w.yaw), math.sin(w.yaw)
            ox, oy = px - w.cx, py - w.cy
            lx, ly = c * ox + s * oy, -s * ox + c * oy
            rx, ry = c * dx + s * dy, -s * dx + c * dy
            t0, t1, hit = 0.0, best, True
            for o, r, h in ((lx, rx, w.half_length),
                            (ly, ry, w.half_thickness)):
                if abs(r) < 1e-12:
                    if abs(o) > h:
                        hit = False
                        break
                    continue
                a, b = (-h - o) / r, (h - o) / r
                t0, t1 = max(t0, min(a, b)), min(t1, max(a, b))
                if t0 > t1:
                    hit = False
                    break
            if hit:
                best = t0
        out[i] = max(best, C.LIDAR_MIN)
    return out


def test_entrance_gap_is_walled_shut(registry):
    """The only border opening is the exit gap (CLAUDE.md geometry invariant).

    An open entrance is not a harmless hole. The robot spawns facing it, no
    lidar ray ever fires across it (there is nothing there to reflect), and
    driving out through it terminated the episode at -30 as out_of_bounds:
    11 of 17 logged episodes (65%) died that way, after a median 72 steps
    versus 368 for a genuine wall collision.
    """
    for name in _names(registry):
        spec = registry[name]
        entrance = next(op for op in spec.openings if op.role == "entrance")
        ex, ey = entrance.xy_local
        span = np.linspace(-entrance.half_width, entrance.half_width, 41)
        xs, ys = ex + span, np.full(span.shape, ey)
        covered = np.zeros(span.shape, dtype=bool)
        for w in spec.walls_local:
            covered |= w.contains_array(xs, ys)
        assert covered.all(), \
            f"{name}: entrance mouth at ({ex:.3f}, {ey:.3f}) is not walled shut"


def test_spawn_pose_clears_every_wall_by_the_collision_model(registry):
    """Sealing the entrance moves a wall right in front of some spawns.

    delta_1/delta_2 start only 0.147 m from the north border, and the rear
    collision threshold is 0.26 m, so an unnudged spawn scores an instant
    -30 collision on step 0. The spawn pose must clear every wall under the
    SAME per-ray test the env terminates on.
    """
    from rl_training.wheel_env import compute_collision_thresholds
    thresh = compute_collision_thresholds()
    for name in _names(registry):
        spec = registry[name]
        sx, sy = spec.start_xy_world
        scan = _simulated_scan(spec.walls_world, sx, sy, spec.start_yaw)
        slack = (scan - thresh).min()
        assert slack > 0.0, \
            f"{name}: spawn slack {slack:+.3f} m — collides on step 0"


def test_zone_start_matches_the_spawn_pose(registry):
    """zone_start is the zone the robot actually spawns in.

    It is decoded from the PNG before any spawn adjustment, so a moved start
    silently desynchronizes it from reality and the coverage tracker credits
    the wrong first zone.
    """
    for name in _names(registry):
        spec = registry[name]
        assert spec.zone_label_at_world(*spec.start_xy_world) == \
            spec.zone_start, f"{name}: zone_start out of sync with the start"
