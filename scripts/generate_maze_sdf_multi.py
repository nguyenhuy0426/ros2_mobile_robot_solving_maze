#!/usr/bin/env python3
"""
generate_maze_sdf_multi.py — Generate standalone + combined maze world SDFs.

Pipeline (v5 maze registry → Gazebo worlds):

  1. Load the 13 decoded maze definitions from ``worlds/maze_defs/*.json``
     via ``rl_training.maze_registry.load_registry()``.
  2. For each maze build a STANDALONE world SDF
     (``worlds/nhom8_maze_<name>.sdf``): physics plugins, directional light,
     50 m ground plane, one wall model per ``spec.walls_world`` entry (box
     long axis = local X rotated by pose yaw, WALL_H tall), a beige floor
     slab, and a comment banner documenting the maze.
  3. Build the COMBINED world (``worlds/nhom8_maze_multi.sdf``): all 13
     mazes on the CENTERED ``multi_placement()`` grid (4 columns, 8 m
     spacing ⇒ mazes span roughly [-13.97, +13.97]), each maze's walls
     recomputed at its grid placement, plugin set emitted exactly once,
     80 m ground plane.
  4. Verify EVERY emitted world by re-parsing it with
     ``rl_training.maze_field.parse_walls`` and checking each parsed wall
     against the registry geometry (positions/sizes within 2 mm, yaw mod π),
     then prove the seal at 1 cm resolution (robot-radius dilation, flood
     fill from the start and from the moat — leaks only where the floods
     meet OUTSIDE every opening's gate box) plus the ortho perimeter walk.

Protected originals — ``nhom8_maze75.sdf``, ``nhom8_maze.sdf`` and
``nhom8_multi5.sdf`` — are never written; an explicit guard asserts the
output path is not one of them before every write.

Run:  ./rl_venv/bin/python scripts/generate_maze_sdf_multi.py
      [--only NAME] [--skip-combined] [--combined-only]
"""

from __future__ import annotations

import argparse
import math
import sys
from collections import deque
from pathlib import Path

import numpy as np

WS_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WS_ROOT))

from rl_training import config as C          # noqa: E402
import rl_training.maze_registry as MR       # noqa: E402
from rl_training.maze_field import parse_walls  # noqa: E402
from rl_training.maze_registry import MazeSpec  # noqa: E402
from rl_training.maze_field import Wall         # noqa: E402

WALL_H = 0.40                # wall height (m) — same as generate_maze_sdf
FLOOR_H = 0.03               # floor slab thickness (m)
FLOOR_MARGIN = 0.10          # floor oversize vs the wall bbox (m)

PROTECTED_SDFS = {
    WS_ROOT / "worlds" / "nhom8_maze75.sdf",
    WS_ROOT / "worlds" / "nhom8_maze.sdf",
    WS_ROOT / "worlds" / "nhom8_multi5.sdf",
}


# ══════════════════════════════════════════════════════════════════════════
# SDF emission
# ══════════════════════════════════════════════════════════════════════════

def wall_model_xml(name: str, wall: Wall, wall_h: float = WALL_H) -> str:
    """One static wall model: OBB box (long axis = local X, pose yaw)."""
    return f"""    <model name="{name}">
      <static>true</static>
      <pose>{wall.cx:.6f} {wall.cy:.6f} {wall_h / 2:.4f} 0 0 {wall.yaw:.6f}</pose>
      <link name="link">
        <collision name="col">
          <geometry><box><size>{2 * wall.half_length:.4f} {2 * wall.half_thickness:.4f} {wall_h:.4f}</size></box></geometry>
        </collision>
        <visual name="vis">
          <geometry><box><size>{2 * wall.half_length:.4f} {2 * wall.half_thickness:.4f} {wall_h:.4f}</size></box></geometry>
          <material><ambient>0.15 0.45 0.85 1</ambient><diffuse>0.15 0.45 0.85 1</diffuse></material>
        </visual>
      </link>
    </model>
"""


def floor_model_xml(name: str, cx: float, cy: float, sx: float,
                    sy: float) -> str:
    """Beige floor slab under one maze (same look as generate_maze_sdf)."""
    return f"""    <model name="{name}">
      <static>true</static>
      <pose>{cx:.6f} {cy:.6f} -0.02 0 0 0</pose>
      <link name="link">
        <collision name="col">
          <geometry><box><size>{sx:.4f} {sy:.4f} {FLOOR_H:.2f}</size></box></geometry>
        </collision>
        <visual name="vis">
          <geometry><box><size>{sx:.4f} {sy:.4f} {FLOOR_H:.2f}</size></box></geometry>
          <material><ambient>0.87 0.87 0.82 1</ambient><diffuse>0.87 0.87 0.82 1</diffuse></material>
        </visual>
      </link>
    </model>
"""


def world_header(world_name: str, ground_size: float = 50.0) -> str:
    """Physics plugins + light + ground plane (template from
    generate_maze_sdf.build_sdf_text, parameterized ground size)."""
    return f"""<?xml version="1.0" ?>
<sdf version="1.10">
<world name="{world_name}">
  <physics name="1ms" type="ignored">
    <max_step_size>0.001</max_step_size>
    <real_time_factor>1.0</real_time_factor>
  </physics>
  <plugin filename="gz-sim-physics-system"   name="gz::sim::systems::Physics"/>
  <plugin filename="gz-sim-user-commands-system" name="gz::sim::systems::UserCommands"/>
  <plugin filename="gz-sim-scene-broadcaster-system" name="gz::sim::systems::SceneBroadcaster"/>
  <plugin filename="gz-sim-sensors-system" name="gz::sim::systems::Sensors">
    <render_engine>ogre2</render_engine>
  </plugin>

  <light type="directional" name="sun">
    <cast_shadows>true</cast_shadows>
    <pose>0 0 10 0 0 0</pose>
    <diffuse>1 1 1 1</diffuse>
    <specular>0.5 0.5 0.5 1</specular>
    <attenuation><range>1000</range><constant>0.9</constant>
      <linear>0.01</linear><quadratic>0.001</quadratic></attenuation>
    <direction>-0.5 0.1 -0.9</direction>
  </light>

  <model name="ground_plane">
    <static>true</static>
    <link name="link">
      <collision name="collision">
        <geometry><plane><normal>0 0 1</normal><size>{ground_size} {ground_size}</size></plane></geometry>
        <surface><friction><ode><mu>40</mu></ode></friction></surface>
      </collision>
      <visual name="visual">
        <geometry><plane><normal>0 0 1</normal><size>{ground_size} {ground_size}</size></plane></geometry>
        <material>
          <ambient>0.8 0.8 0.8 1</ambient><diffuse>0.8 0.8 0.8 1</diffuse>
          <specular>0.8 0.8 0.8 1</specular>
        </material>
      </visual>
    </link>
  </model>
"""


def _maze_comment(spec: MazeSpec, placement: tuple) -> str:
    """Comment banner: maze name, family, walls, bbox, openings, start."""
    ops = "  ".join(
        f"{op.role}={op.border}@({op.xy_local[0]:.3f},{op.xy_local[1]:.3f})"
        f" hw={op.half_width:.3f}" for op in spec.openings)
    return f"""  <!-- ═══════════════════════════════════════════════════════ -->
  <!--  MAZE {spec.name} (family {spec.family})                -->
  <!--  walls: {len(spec.walls_local)}   wall bbox (local): {tuple(round(v, 4) for v in spec.wall_bbox)}   -->
  <!--  openings: {ops}   -->
  <!--  start (local): ({spec.start_xy_local[0]:.4f}, {spec.start_xy_local[1]:.4f}) yaw={spec.start_yaw:.4f} rad  -->
  <!--  placement center: ({placement[0]:.4f}, {placement[1]:.4f})   -->
  <!-- ═══════════════════════════════════════════════════════ -->
"""


def _maze_block(spec: MazeSpec, placement: tuple, wall_prefix: str,
                floor_name: str) -> str:
    """Comment banner + walls + floor for one maze placed at ``placement``.

    Walls are recomputed from the LOCAL walls at this placement
    (``spec.place_at(placement).walls_world``) so the combined world is
    correct for arbitrary grid slots.
    """
    placed = spec.place_at(placement)
    parts = [_maze_comment(placed, placement)]
    for i, w in enumerate(placed.walls_world):
        parts.append(wall_model_xml(f"{wall_prefix}_{i:03d}", w))
    bw = placed.wall_bbox[2] - placed.wall_bbox[0]
    bh = placed.wall_bbox[3] - placed.wall_bbox[1]
    parts.append(floor_model_xml(floor_name, placement[0], placement[1],
                                 bw + FLOOR_MARGIN, bh + FLOOR_MARGIN))
    return "\n".join(parts)


def build_standalone_text(spec: MazeSpec, world_name: str) -> str:
    """Full standalone world SDF for one maze at MAZE_CENTER_XY."""
    body = _maze_block(spec, tuple(spec.placement_offset), "wall", "floor")
    return (world_header(world_name, ground_size=50.0)
            + "\n" + body + "</world>\n</sdf>\n")


def build_combined_text(specs: dict, placements: dict) -> str:
    """Full combined world SDF: all mazes on the multi-placement grid."""
    parts = [world_header(C.MAZE_MULTI_WORLD_NAME, ground_size=80.0), "\n"]
    for name in sorted(specs):
        parts.append(_maze_block(specs[name], placements[name],
                                 f"{name}_wall", f"{name}_floor"))
        parts.append("\n")
    parts.append("</world>\n</sdf>\n")
    return "".join(parts)


# ══════════════════════════════════════════════════════════════════════════
# Seal verification (family-agnostic, 1 cm proof)
# ══════════════════════════════════════════════════════════════════════════

def _dilate4(mask: np.ndarray, iters: int) -> np.ndarray:
    """Binary dilation by ``iters`` cells, 4-neighborhood (np.roll-free,
    slice-shift loop; no scipy)."""
    out = mask.copy()
    for _ in range(iters):
        g = out.copy()
        g[1:, :] |= out[:-1, :]
        g[:-1, :] |= out[1:, :]
        g[:, 1:] |= out[:, :-1]
        g[:, :-1] |= out[:, 1:]
        out = g
    return out


def _flood4(passable: np.ndarray, seed) -> np.ndarray:
    """4-connected BFS flood fill from a (ix, iy) seed cell."""
    ny, nx = passable.shape
    seen = np.zeros_like(passable)
    if not passable[seed[1], seed[0]]:
        return seen
    seen[seed[1], seed[0]] = True
    q = deque([seed[1] * nx + seed[0]])
    while q:
        f = q.popleft()
        iy, ix = divmod(f, nx)
        for jx, jy in ((ix + 1, iy), (ix - 1, iy),
                       (ix, iy + 1), (ix, iy - 1)):
            if (0 <= jx < nx and 0 <= jy < ny
                    and passable[jy, jx] and not seen[jy, jx]):
                seen[jy, jx] = True
                q.append(jy * nx + jx)
    return seen


def _nudge_to_free(free: np.ndarray, ix: int, iy: int):
    """Spiral search for the nearest free cell around (ix, iy)."""
    if free[iy, ix]:
        return ix, iy
    ny, nx = free.shape
    for r in range(1, max(nx, ny)):
        for dy in range(-r, r + 1):
            for dx in range(-r, r + 1):
                jx, jy = ix + dx, iy + dy
                if 0 <= jx < nx and 0 <= jy < ny and free[jy, jx]:
                    return jx, jy
    raise RuntimeError("no free cell anywhere in the raster")


def _gate_mask(spec: MazeSpec, gx: np.ndarray, gy: np.ndarray,
               res: float) -> np.ndarray:
    """Gate boxes: within half_width + 2*res of the gap center along the
    border AND within 0.10 m of the border line along its normal, UNION a
    recess box per opening: the same along-border span, but along the
    normal from the border line through the recorded mouth line extended
    0.10 m into the maze interior. Openings recessed into a zigzag
    outline (sigma's north V mouth sits ~0.33 m below the bbox line)
    would otherwise leak past a gate pinned to the bbox line. For flush
    openings (xy_normal == border line) the recess box is contained in
    the first box, so nothing changes."""
    bx0, by0, bx1, by1 = spec.wall_bbox
    gate = np.zeros(gx.shape, dtype=bool)
    for op in spec.openings:
        ox, oy = op.xy_local
        span = op.half_width + 2 * res
        if op.border == "south":
            along = np.abs(gx - ox) <= span
            gate |= along & (np.abs(gy - by0) <= 0.10)
            gate |= along & (gy >= min(by0, oy - 0.10)) & (gy <= max(by0, oy))
        elif op.border == "north":
            along = np.abs(gx - ox) <= span
            gate |= along & (np.abs(gy - by1) <= 0.10)
            gate |= along & (gy >= min(by1, oy - 0.10)) & (gy <= max(by1, oy))
        elif op.border == "west":
            along = np.abs(gy - oy) <= span
            gate |= along & (np.abs(gx - bx0) <= 0.10)
            gate |= along & (gx >= min(bx0, ox - 0.10)) & (gx <= max(bx0, ox))
        elif op.border == "east":
            along = np.abs(gy - oy) <= span
            gate |= along & (np.abs(gx - bx1) <= 0.10)
            gate |= along & (gx >= min(bx1, ox - 0.10)) & (gx <= max(bx1, ox))
    return gate


def _ortho_perimeter_violations(spec: MazeSpec) -> list:
    """Walk the bbox rectangle perimeter at exactly 1 cm steps; every point
    must lie inside some wall (inflate 2 mm) except inside the openings'
    gaps. Walks are parameterized so the ENDPOINT lands exactly on the
    rectangle corner (no half-step overshoot)."""
    bx0, by0, bx1, by1 = spec.wall_bbox
    step = 0.01
    violations = []
    for border, a0, a1, fixed in (("south", bx0, bx1, by0),
                                  ("east", by0, by1, bx1),
                                  ("north", bx1, bx0, by1),
                                  ("west", by1, by0, bx0)):
        ops = [op for op in spec.openings if op.border == border]
        n = int(round(abs(a1 - a0) / step))
        sgn = 1.0 if a1 >= a0 else -1.0
        along_axis = 0 if border in ("south", "north") else 1
        for i in range(n + 1):
            a = a0 + sgn * i * step
            if i == n:
                a = a1                       # land exactly on the corner
            px, py = (a, fixed) if along_axis == 0 else (fixed, a)
            along = a
            if any(abs(along - op.xy_local[along_axis])
                   <= op.half_width + 0.005 for op in ops):
                continue
            if not any(w.contains(px, py, 0.002) for w in spec.walls_local):
                violations.append((border, round(px, 4), round(py, 4)))
    return violations


def verify_maze_seal(spec: MazeSpec, walls=None, res: float = 0.01,
                     robot_r: float = C.ROBOT_RADIUS) -> dict:
    """Family-agnostic 1 cm seal proof in the maze-LOCAL frame.

    a. Rasterize walls (cell center blocked iff inside any wall OBB).
    b. blocked = wall raster dilated by round(robot_r / res) cells
       (4-neighborhood) — the robot-radius inflation.
    c. free = ~blocked; flood_maze = 4-connected BFS from the start cell
       (nudged to a free cell) over free cells with all gate boxes CLOSED.
    d. flood_moat = BFS from a free perimeter-ring cell over the same
       passable raster.
    e. leaks = flood_maze ∩ flood_moat — cells reachable from BOTH the
       maze interior and the moat despite the closed gates, i.e. a border
       crossing that bypasses the registered openings. Must be empty.
       (Formulation note: with gates OPEN both floods trivially coincide
       with the single free component of every maze — the informative test
       is start↔moat reachability with the gates closed.)
    f. ortho extra: exact 1 cm perimeter walk of the bbox rectangle (2 mm
       inflate), skipping the two opening gaps.
    """
    if walls is None:
        walls = spec.walls_local
    bx0, by0, bx1, by1 = spec.bounds
    nx = int(round((bx1 - bx0) / res))
    ny = int(round((by1 - by0) / res))
    xs = bx0 + (np.arange(nx) + 0.5) * res
    ys = by0 + (np.arange(ny) + 0.5) * res
    gx, gy = np.meshgrid(xs, ys)

    wall_raster = np.zeros((ny, nx), dtype=bool)
    for w in walls:
        wall_raster |= w.contains_array(gx, gy, 0.0)
    blocked = _dilate4(wall_raster, int(round(robot_r / res)))
    free = ~blocked

    gate = _gate_mask(spec, gx, gy, res)
    passable = free & ~gate

    six = int(np.clip(int((spec.start_xy_local[0] - bx0) / res), 0, nx - 1))
    siy = int(np.clip(int((spec.start_xy_local[1] - by0) / res), 0, ny - 1))
    six, siy = _nudge_to_free(free, six, siy)
    flood_maze = _flood4(passable, (six, siy))

    ring = ([(ix, 0) for ix in range(nx)]
            + [(ix, ny - 1) for ix in range(nx)]
            + [(0, iy) for iy in range(1, ny - 1)]
            + [(nx - 1, iy) for iy in range(1, ny - 1)])
    mseed = next(((ix, iy) for ix, iy in ring if free[iy, ix]), None)
    if mseed is None:
        leaks = flood_maze.copy()          # moat fully walled: total leak
        leak_samples = []
    else:
        flood_moat = _flood4(passable, mseed)
        inter = flood_maze & flood_moat
        leaks = inter
        idx = np.argwhere(inter)
        leak_samples = [(round(bx0 + (jx + 0.5) * res, 3),
                         round(by0 + (iy + 0.5) * res, 3))
                        for iy, jx in idx[:10]]

    ortho_violations = _ortho_perimeter_violations(spec) \
        if spec.family == "ortho" else []

    return {
        "ok": int(leaks.sum()) == 0 and not ortho_violations,
        "leak_count": int(leaks.sum()),
        "leak_samples": leak_samples,
        "ortho_violations": len(ortho_violations),
        "ortho_violation_points": ortho_violations[:10],
        "free_cells": int(free.sum()),
        "maze_flood_cells": int(flood_maze.sum()),
        "moat_seed_free": mseed is not None,
    }


def verify_sdf_file(sdf_path: Path, spec: MazeSpec,
                    placement_offset) -> dict:
    """Parse the emitted SDF back into walls and close the loop
    SDF → geometry → seal (walls translated back to maze-local).

    In the combined world the file contains every maze's walls; the ones
    belonging to ``spec`` are selected by their placement: walls whose
    center lies within the maze's wall bbox (+ small margin) around
    ``placement_offset``.
    """
    parsed = parse_walls(sdf_path)
    ox, oy = placement_offset
    bx0, by0, bx1, by1 = spec.wall_bbox
    margin = 0.01
    own = [w for w in parsed
           if (bx0 + ox - margin) <= w.cx <= (bx1 + ox + margin)
           and (by0 + oy - margin) <= w.cy <= (by1 + oy + margin)]
    assert len(own) == len(spec.walls_local), (
        f"{sdf_path.name}: {len(own)} walls near placement "
        f"({ox:.3f}, {oy:.3f}) != {len(spec.walls_local)} registry walls")
    tol = 2e-3
    for i, (pw, sw) in enumerate(zip(own, spec.walls_local)):
        assert abs(pw.cx - (sw.cx + ox)) <= tol, \
            f"{sdf_path.name} wall {i}: Δcx={abs(pw.cx - (sw.cx + ox)):.5f}"
        assert abs(pw.cy - (sw.cy + oy)) <= tol, \
            f"{sdf_path.name} wall {i}: Δcy={abs(pw.cy - (sw.cy + oy)):.5f}"
        assert abs(pw.half_length - sw.half_length) <= tol, \
            f"{sdf_path.name} wall {i}: Δhalf_length"
        assert abs(pw.half_thickness - sw.half_thickness) <= tol, \
            f"{sdf_path.name} wall {i}: Δhalf_thickness"
        dyaw = (pw.yaw - sw.yaw + math.pi / 2.0) % math.pi - math.pi / 2.0
        assert abs(dyaw) <= tol, \
            f"{sdf_path.name} wall {i}: Δyaw={math.degrees(dyaw):.4f}°"
    walls_local = tuple(w.translated(-ox, -oy) for w in own)
    return verify_maze_seal(spec, walls=walls_local)


# ══════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════

def _safe_write(path: Path, text: str) -> None:
    """Write only if the target is NOT one of the three protected SDFs."""
    resolved = path.resolve()
    assert resolved not in PROTECTED_SDFS, (
        f"refusing to overwrite protected world: {resolved}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


def _print_row(name: str, spec: MazeSpec, seal: dict, path: Path) -> None:
    bw = spec.wall_bbox[2] - spec.wall_bbox[0]
    bh = spec.wall_bbox[3] - spec.wall_bbox[1]
    ops = " ".join(f"{op.role[:3]}:{op.border[:2]}={op.half_width:.2f}"
                   for op in spec.openings)
    status = "OK " if seal["ok"] else "FAIL"
    extra = ""
    if seal["leak_count"]:
        extra += f" LEAKS={seal['leak_count']} {seal['leak_samples']}"
    if seal["ortho_violations"]:
        extra += (f" ORTHO_VIO={seal['ortho_violations']} "
                  f"{seal['ortho_violation_points']}")
    print(f"  {status} {name:<9} walls={len(spec.walls_local):>3} "
          f"bbox={bw:.2f}x{bh:.2f} [{ops}] free={seal['free_cells']}"
          f"  -> {path.name}{extra}")


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Generate + verify standalone/combined maze world SDFs")
    ap.add_argument("--defs-dir", type=Path, default=None,
                    help="maze_defs directory (default config.MAZE_DEFS_DIR)")
    ap.add_argument("--out-dir", type=Path,
                    default=WS_ROOT / "worlds",
                    help="output directory (default <ws>/worlds)")
    ap.add_argument("--only", metavar="NAME", default=None,
                    help="generate only the standalone world of NAME")
    ap.add_argument("--skip-combined", action="store_true",
                    help="skip the combined multi-maze world")
    ap.add_argument("--combined-only", action="store_true",
                    help="generate only the combined world")
    args = ap.parse_args()

    registry = MR.load_registry(args.defs_dir)
    names = sorted(registry)
    if args.only is not None:
        if args.only not in registry:
            raise SystemExit(f"unknown maze {args.only!r} — known: "
                             f"{', '.join(names)}")
        names = [args.only]

    out_dir = args.out_dir
    failures = []

    if not args.combined_only:
        print(f"── standalone worlds ({len(names)}) ──")
        for name in names:
            spec = registry[name]
            sdf_path = out_dir / f"{MR.standalone_world_name(name)}.sdf"
            _safe_write(sdf_path, build_standalone_text(
                spec, MR.standalone_world_name(name)))
            seal = verify_sdf_file(sdf_path, spec, spec.placement_offset)
            _print_row(name, spec, seal, sdf_path)
            if not seal["ok"]:
                failures.append(name)

    if not args.skip_combined and args.only is None:
        print("── combined world ──")
        specs = {n: registry[n] for n in names}
        placements = MR.multi_placement(sorted(specs))
        sdf_path = Path(C.MAZE_MULTI_SDF)
        _safe_write(sdf_path, build_combined_text(specs, placements))
        for name in sorted(specs):
            spec = specs[name]
            center = placements[name]
            seal = verify_sdf_file(sdf_path, spec, center)
            _print_row(name, spec, seal, sdf_path)
            if not seal["ok"]:
                failures.append(f"{name}(combined)")

        x0 = min(placements[n][0] + registry[n].wall_bbox[0] for n in specs)
        x1 = max(placements[n][0] + registry[n].wall_bbox[2] for n in specs)
        y0 = min(placements[n][1] + registry[n].wall_bbox[1] for n in specs)
        y1 = max(placements[n][1] + registry[n].wall_bbox[3] for n in specs)
        print(f"  combined extent: x [{x0:.3f}, {x1:.3f}]  "
              f"y [{y0:.3f}, {y1:.3f}]  (ground 80 m)")

    print(f"── summary: {len(names)} mazes, "
          f"{'FAILED: ' + ', '.join(failures) if failures else 'all seal checks OK'}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
