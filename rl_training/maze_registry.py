#!/usr/bin/env python3
"""
maze_registry.py — Registry of the 13 decoded mazes (v5).

Loads the maze definitions decoded from ``mazes_png/*.png`` (worlds/
maze_defs/*.json, one JSON per maze) into immutable :class:`MazeSpec` objects
and provides the geometry/learning artifacts built from them: geodesic
distance fields (:meth:`build_distance_field`), occupancy-grid mappers
(:meth:`build_mapper`), zone-coverage trackers (:meth:`build_zone_coverage`)
and spawn helpers for the standalone and combined (multi-maze) worlds.

Conventions
───────────
* Maze-local frame: origin = wall-bbox center, +y up. Wall segments are
  stored as straight [x0, y0, x1, y1] lines, 0.03 m thick
  (``config.EXPL_WALL_T``), pre-extended by half a thickness on each end so
  junctions seal. ``bounds`` is the wall bbox grown by a 0.30 m moat.
* Openings: gap centers on the border lines (entrance = north / ymax, exit =
  south / ymin for all 13 mazes); the free-run half width along the border is
  measured from the wall geometry (see ``_opening_half_width``).
* Placement: each maze is placed by putting its wall-bbox CENTER at a world
  anchor point. Standalone worlds anchor the center at ``config.MAZE_CENTER_XY``
  (the old 0.75 m maze footprint center, decision 2026-08-30); the combined
  world places maze i on a CENTERED grid at
  ((col - (MAZE_MULTI_COLS-1)/2) * MAZE_MULTI_SPACING,
   (row - (MAZE_MULTI_COLS-1)/2) * MAZE_MULTI_SPACING) so the maze block is
  symmetric about the world origin and every maze footprint stays on the
  ground plane. World coordinates = maze-local + ``placement_offset``.

Used by the v5 explore-then-exit env and spawn scripts; importable without
ROS/Gazebo (pure numpy + the rl_training geometry helpers).
"""

from __future__ import annotations

import argparse
import base64
import json
import math
import zlib
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np

from rl_training import config as C
from rl_training.maze_field import MazeDistanceField, Wall, wall_from_segment
from rl_training.occ_map import OccupancyGridMapper
from rl_training.reward_shaping import ZoneCoverage

N_ZONES = 25
WALL_T = C.EXPL_WALL_T          # 0.03
ROBOT_R = C.ROBOT_RADIUS        # 0.10
PLAN_INFLATE = ROBOT_R + 0.06   # 0.16 — decode-time planning inflation


@dataclass(frozen=True)
class Opening:
    role: str                        # "entrance" | "exit"
    xy_local: Tuple[float, float]    # gap center on the border line (maze-local)
    border: str                      # "south" | "north" | "west" | "east"
    half_width: float                # free-run half width along the border (m)


@dataclass(frozen=True)
class MazeSpec:
    """Immutable description of one maze + its world placement.

    A frozen dataclass holding the decoded numpy label raster is fine: the
    array is treated as read-only and shared between artifacts.
    """
    name: str
    family: str                                  # "ortho" | "sigma" | "delta"
    walls_local: Tuple[Wall, ...]                # maze-local OBB walls
    wall_bbox: Tuple[float, float, float, float]  # xmin, ymin, xmax, ymax (local)
    bounds: Tuple[float, float, float, float]    # JSON bounds (wall bbox + moat)
    openings: Tuple[Opening, ...]                # entrance + exit
    start_xy_local: Tuple[float, float]
    start_yaw: float                             # radians
    zone_res: float
    zone_labels: np.ndarray                      # int16 [ny, nx]; -1 = wall/outside
    zone_centroids_local: Tuple[Tuple[float, float], ...]  # 25 zone centroids
    zone_start: int                              # zone id containing the start
    placement_offset: Tuple[float, float] = (0.0, 0.0)  # world = local + offset

    # ── Frame conversion ─────────────────────────────────────────────

    def to_world(self, xy_local: Tuple[float, float]) -> Tuple[float, float]:
        ox, oy = self.placement_offset
        return (xy_local[0] + ox, xy_local[1] + oy)

    def to_local(self, xy_world: Tuple[float, float]) -> Tuple[float, float]:
        ox, oy = self.placement_offset
        return (xy_world[0] - ox, xy_world[1] - oy)

    def place_at(self, center_xy: Tuple[float, float]) -> "MazeSpec":
        """Copy of this spec with its wall-bbox center anchored at center_xy."""
        return replace(self, placement_offset=(float(center_xy[0]),
                                               float(center_xy[1])))

    # ── World-frame geometry ─────────────────────────────────────────

    @property
    def walls_world(self) -> Tuple[Wall, ...]:
        return tuple(w.translated(*self.placement_offset)
                     for w in self.walls_local)

    @property
    def wall_bbox_world(self) -> Tuple[float, float, float, float]:
        ox, oy = self.placement_offset
        return (self.wall_bbox[0] + ox, self.wall_bbox[1] + oy,
                self.wall_bbox[2] + ox, self.wall_bbox[3] + oy)

    @property
    def start_xy_world(self) -> Tuple[float, float]:
        return self.to_world(self.start_xy_local)

    @property
    def norm_bounds_world(self) -> Tuple[float, float, float, float]:
        """(xmin, ymin, width, height) of the maze in world frame — for
        odometry normalization."""
        x0, y0, x1, y1 = self.wall_bbox_world
        return (x0, y0, x1 - x0, y1 - y0)

    @property
    def exit_opening(self) -> Opening:
        exits = [op for op in self.openings if op.role == "exit"]
        assert exits, f"{self.name}: no exit opening"
        return exits[0]

    # ── Zone raster lookup (world frame) ─────────────────────────────

    def zone_label_at_world(self, x: float, y: float) -> int:
        """Zone label at a world point; -1 outside the raster / on walls."""
        lx, ly = self.to_local((x, y))
        ix = int(round((lx - self.bounds[0]) / self.zone_res))
        iy = int(round((ly - self.bounds[1]) / self.zone_res))
        ny, nx = self.zone_labels.shape
        if not (0 <= ix < nx and 0 <= iy < ny):
            return -1
        return int(self.zone_labels[iy, ix])


# ══════════════════════════════════════════════════════════════════════════
# Registry loading
# ══════════════════════════════════════════════════════════════════════════

def load_registry(defs_dir=None,
                  center_xy=None) -> Dict[str, MazeSpec]:
    """Load all maze definitions (sorted *.json) as {name: MazeSpec}.

    Each maze is placed with its wall-bbox center at ``center_xy``
    (default ``config.MAZE_CENTER_XY`` — the standalone-world anchor).
    """
    defs_dir = Path(defs_dir) if defs_dir is not None else Path(C.MAZE_DEFS_DIR)
    if center_xy is None:
        center_xy = tuple(C.MAZE_CENTER_XY)
    registry: Dict[str, MazeSpec] = {}
    for path in sorted(defs_dir.glob("*.json")):
        spec = _maze_from_json(path, center_xy)
        registry[spec.name] = spec
    return registry


def _maze_from_json(path, center_xy) -> MazeSpec:
    data = json.loads(Path(path).read_text())

    walls_local = tuple(
        wall_from_segment(w[0], w[1], w[2], w[3], WALL_T)
        for w in data["walls"])
    xs = [w[0] for w in data["walls"]] + [w[2] for w in data["walls"]]
    ys = [w[1] for w in data["walls"]] + [w[3] for w in data["walls"]]
    wall_bbox = (min(xs), min(ys), max(xs), max(ys))

    openings = []
    for op in data["openings"]:
        gx, gy = float(op["xy"][0]), float(op["xy"][1])
        border = _opening_border(wall_bbox, gx, gy)
        if "half_width" in op:
            # decoder-measured mouth width (outline-depth runs for sigma);
            # same clamp as the measured path below
            half_width = float(min(max(float(op["half_width"]), 0.15), 0.5))
        else:
            half_width = _opening_half_width(walls_local, wall_bbox, (gx, gy),
                                             border)
        openings.append(Opening(role=op["role"], xy_local=(gx, gy),
                                border=border, half_width=half_width))

    shape = data["zone_labels_shape"]              # [ny, nx]
    labels = np.frombuffer(
        zlib.decompress(base64.b64decode(data["zone_labels_b64"])),
        dtype=np.int16).reshape(shape[0], shape[1])

    return MazeSpec(
        name=data["name"],
        family=data["family"],
        walls_local=walls_local,
        wall_bbox=wall_bbox,
        bounds=tuple(float(v) for v in data["bounds"]),
        openings=tuple(openings),
        start_xy_local=(float(data["start_xy"][0]), float(data["start_xy"][1])),
        start_yaw=math.radians(float(data["start_yaw_deg"])),
        zone_res=float(data["zone_res"]),
        zone_labels=labels,
        zone_centroids_local=tuple((float(c[0]), float(c[1]))
                                   for c in data["zone_centroids"]),
        zone_start=int(data["zone_start"]),
        placement_offset=(0.0, 0.0),
    ).place_at(center_xy)


def _opening_border(wall_bbox, gx: float, gy: float) -> str:
    """Nearest wall-bbox side to the gap center (openings lie ON a border)."""
    dist = {
        "south": abs(gy - wall_bbox[1]),
        "north": abs(gy - wall_bbox[3]),
        "west": abs(gx - wall_bbox[0]),
        "east": abs(gx - wall_bbox[2]),
    }
    for border, d in dist.items():
        if d < 0.02:
            return border
    return min(dist, key=dist.get)


def _opening_half_width(walls_local: Tuple[Wall, ...], wall_bbox,
                        gap_xy: Tuple[float, float], border: str) -> float:
    """Free-run half width along the border containing the gap.

    The border is scanned on a 3 cm-wide band centered on the border line:
    a point along the border is "sealed" when EITHER of the two lines offset
    ±0.018 m from it is covered by some wall (``contains_array`` with a
    0.001 inflate). The band — not the exact border line — mirrors how the
    decoder measured gaps on the image outline band: sigma/delta outlines
    are zigzags, so scanning the exact border LINE would wrongly merge gaps
    with outline dips, while the band catches diagonal outline walls
    crossing it. The offset is 0.018 m (not 0.012 m) because the decoded
    outline walls sit 13–14 mm below the nominal border line — a ±0.012 m
    scan band missed them and over-measured the sigma bottom gaps (raw
    0.965 m clamped to 0.5 m).
    """
    step = 0.005                                   # scan step (m)
    x0, y0, x1, y1 = wall_bbox
    horizontal = border in ("south", "north")      # scan along x
    line = y1 if border == "north" else y0 if border == "south" \
        else x1 if border == "east" else x0
    lo, hi = (x0, x1) if horizontal else (y0, y1)
    gap_c = gap_xy[0] if horizontal else gap_xy[1]

    n_samples = int(round((hi - lo) / step)) + 1
    coords = lo + np.arange(n_samples) * step

    def sealed_line(offset: float) -> np.ndarray:
        if horizontal:
            xs, ys = coords, np.full(n_samples, line + offset)
        else:
            xs, ys = np.full(n_samples, line + offset), coords
        mask = np.zeros(n_samples, dtype=bool)
        for w in walls_local:
            mask |= w.contains_array(xs, ys, inflate=0.001)
        return mask

    sealed = sealed_line(-0.018) | sealed_line(+0.018)

    gap_idx = min(max(int(round((gap_c - lo) / step)), 0), n_samples - 1)
    if sealed[gap_idx]:
        unsealed = np.nonzero(~sealed)[0]
        if unsealed.size == 0:                     # pathological: fully sealed
            return 0.15
        gap_idx = int(unsealed[np.argmin(np.abs(unsealed - gap_idx))])

    i0 = i1 = gap_idx                              # maximal run around the gap
    while i0 > 0 and not sealed[i0 - 1]:
        i0 -= 1
    while i1 < n_samples - 1 and not sealed[i1 + 1]:
        i1 += 1
    run_length = (i1 - i0 + 1) * step
    half_width = run_length / 2.0
    return float(min(max(half_width, 0.15), 0.5))


# ══════════════════════════════════════════════════════════════════════════
# Artifact builders
# ══════════════════════════════════════════════════════════════════════════

def build_distance_field(spec: MazeSpec, resolution: float = 0.025,
                         inflate: float = ROBOT_R) -> MazeDistanceField:
    """Geodesic distance field over the maze's free space, goal = exit.

    The goal is the exit gap center nudged INWARD (toward the bbox center)
    so it sits inside the maze; narrow gaps can leave a nudged goal inside
    an inflated wall, so inward nudges [0.18, 0.30, 0.45] are tried in turn.

    The raster covers ``spec.bounds`` — the wall bbox grown by the 0.30 m
    moat — NOT just the wall bbox. The sigma/delta decoded outlines are open
    zigzags (teeth end at the bbox edge), so the true free-space connectivity
    passes through the moat outside the wall bbox; clipping the raster to the
    bbox cut those paths and fragmented the field (start and exit in separate
    components, ``distance()`` falling back to garbage).

    ``require_connected=False``: sigma/delta free space may legitimately
    contain pockets sealed at the robot-radius inflation (e.g. slivers in
    front of narrow gaps) — ``distance()`` falls back to the nearest finite
    cell for those.

    A wrong-goal guard verifies the start cell is actually reachable and far
    from the goal: if ``d(start)`` is not finite, or is < 1.0 m while the
    euclidean start→goal distance exceeds 1.5 m, the field was built against
    the wrong goal (or a broken wall set) and a ValueError is raised.
    """
    walls_world = spec.walls_world
    ox, oy = spec.placement_offset
    bx0, by0, bx1, by1 = spec.bounds
    origin = (bx0 + ox, by0 + oy)
    extent_xy = (bx1 - bx0, by1 - by0)

    gx, gy = spec.to_world(spec.exit_opening.xy_local)
    wx0, wy0, wx1, wy1 = spec.wall_bbox_world
    cx, cy = (wx0 + wx1) / 2.0, (wy0 + wy1) / 2.0
    dx, dy = cx - gx, cy - gy
    norm = math.hypot(dx, dy)
    ux, uy = (dx / norm, dy / norm) if norm > 1e-9 else (0.0, 1.0)

    last_err: Optional[ValueError] = None
    for nudge in (0.18, 0.30, 0.45):
        goal = (gx + ux * nudge, gy + uy * nudge)
        try:
            field = MazeDistanceField.from_walls(
                walls_world, goal_xy=goal, origin_xy=origin,
                extent_xy=extent_xy, resolution=resolution,
                inflate=inflate, require_connected=False)
        except ValueError as err:
            if "goal" not in str(err):
                raise
            last_err = err
            continue
        _assert_goal_reachable(spec, field, goal)
        return field
    raise last_err


def _assert_goal_reachable(spec: MazeSpec,
                           field: MazeDistanceField,
                           goal: Tuple[float, float]) -> None:
    """Wrong-goal guard: d(start) must be finite and, when the start is far
    from the goal in euclidean terms, not suspiciously small (a sign the
    field was built against the wrong opening or a broken wall set)."""
    d_start = field.distance(*spec.start_xy_world)
    sx, sy = spec.start_xy_world
    euclid = math.hypot(goal[0] - sx, goal[1] - sy)
    if not math.isfinite(d_start) or (d_start < 1.0 and euclid > 1.5):
        raise ValueError(
            f"{spec.name}: distance-field goal sanity check failed — "
            f"d(start)={d_start!r} while euclid(start, goal)="
            f"{euclid:.3f} m (goal={goal!r}); the field is likely built "
            f"against the wrong opening or the wall set is broken")


def build_mapper(spec: MazeSpec, res: float = C.EXPL_MAP_RES,
                 margin: float = C.EXPL_MAP_MARGIN,
                 max_range: float = C.LIDAR_MAX) -> OccupancyGridMapper:
    """Square-grid occupancy mapper covering the (rectangular) maze.

    ``extent`` = the longer bbox side so the square grid always covers the
    maze; ``inner_rect`` confines coverage counting to the actual
    rectangular maze bbox.
    """
    bx0, by0, bx1, by1 = spec.wall_bbox_world
    return OccupancyGridMapper(
        origin_xy=(bx0, by0), extent=max(bx1 - bx0, by1 - by0),
        walls=spec.walls_world, res=res, margin=margin, max_range=max_range,
        inner_rect=(bx0, by0, bx1, by1))


def build_zone_coverage(spec: MazeSpec, bonus: float = 1.0) -> ZoneCoverage:
    """Zone-coverage tracker over the decoded 25-zone label raster."""
    return ZoneCoverage(spec.zone_labels,
                        bounds_min_xy=(spec.bounds[0], spec.bounds[1]),
                        res=spec.zone_res,
                        offset_xy=spec.placement_offset,
                        bonus=bonus, n_zones=N_ZONES)


# ══════════════════════════════════════════════════════════════════════════
# Geometric predicates
# ══════════════════════════════════════════════════════════════════════════

def exit_crossed(spec: MazeSpec, opening: Opening, xy_world,
                 eps: float = 0.05) -> bool:
    """True when the robot's center passed ≥ eps beyond the opening's border
    line, within the gap span."""
    bx0, by0, bx1, by1 = spec.wall_bbox_world
    ox, oy = spec.placement_offset
    x, y = xy_world
    if opening.border == "south":
        return y <= by0 - eps and abs(x - (opening.xy_local[0] + ox)) <= opening.half_width
    if opening.border == "north":
        return y >= by1 + eps and abs(x - (opening.xy_local[0] + ox)) <= opening.half_width
    if opening.border == "west":
        return x <= bx0 - eps and abs(y - (opening.xy_local[1] + oy)) <= opening.half_width
    if opening.border == "east":
        return x >= bx1 + eps and abs(y - (opening.xy_local[1] + oy)) <= opening.half_width
    raise ValueError(f"unknown border: {opening.border!r}")


def out_of_bounds(spec: MazeSpec, xy_world, eps: float = 0.02) -> bool:
    """True when the point is outside the wall bbox grown by eps."""
    bx0, by0, bx1, by1 = spec.wall_bbox_world
    x, y = xy_world
    return (x < bx0 - eps or x > bx1 + eps
            or y < by0 - eps or y > by1 + eps)


# ══════════════════════════════════════════════════════════════════════════
# World placement helpers
# ══════════════════════════════════════════════════════════════════════════

def sorted_registry_names(registry: Optional[Dict[str, MazeSpec]] = None
                          ) -> list:
    if registry is None:
        registry = load_registry()
    return sorted(registry.keys())


def multi_placement(names=None) -> Dict[str, Tuple[float, float]]:
    """Grid centers for the combined world: maze i at a CENTERED grid slot
    ((col - (MAZE_MULTI_COLS - 1) / 2) * MAZE_MULTI_SPACING,
     (row - (MAZE_MULTI_COLS - 1) / 2) * MAZE_MULTI_SPACING), so the maze
    block is symmetric about the world origin and every footprint stays
    inside the 80 m combined ground plane."""
    if names is None:
        names = sorted_registry_names()
    half = (C.MAZE_MULTI_COLS - 1) / 2.0
    return {
        name: ((i % C.MAZE_MULTI_COLS - half) * C.MAZE_MULTI_SPACING,
               (i // C.MAZE_MULTI_COLS - half) * C.MAZE_MULTI_SPACING)
        for i, name in enumerate(names)
    }


def standalone_world_name(name: str) -> str:
    """World name of the standalone per-maze world (e.g. nhom8_maze_sigma_1)."""
    return f"nhom8_maze_{name}"


# ══════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════

def _print_list(registry: Dict[str, MazeSpec]) -> None:
    print(f"{'name':<10} {'family':<7} {'walls':>5} "
          f"{'bbox_w':>7} {'bbox_h':>7}  openings                                   "
          f"   start world pose (x, y, yaw°)")
    for name in sorted(registry):
        s = registry[name]
        bw = s.wall_bbox[2] - s.wall_bbox[0]
        bh = s.wall_bbox[3] - s.wall_bbox[1]
        ops = "  ".join(f"{op.role}:{op.border} hw={op.half_width:.3f}"
                        for op in s.openings)
        sx, sy = s.start_xy_world
        print(f"{s.name:<10} {s.family:<7} {len(s.walls_local):>5} "
              f"{bw:>7.3f} {bh:>7.3f}  {ops:<44} "
              f"({sx:6.3f}, {sy:6.3f}, {math.degrees(s.start_yaw):7.2f})")


def _print_spawn(registry: Dict[str, MazeSpec], name: str,
                 multi: bool) -> None:
    if name not in registry:
        raise SystemExit(f"unknown maze {name!r} — "
                         f"known: {', '.join(sorted_registry_names(registry))}")
    if multi:
        world = C.MAZE_MULTI_WORLD_NAME
        placement = multi_placement()[name]   # full-registry grid position
    else:
        world = standalone_world_name(name)
        placement = tuple(C.MAZE_CENTER_XY)
    spec = registry[name].place_at(placement)
    sx, sy = spec.start_xy_world
    print(f"{world} {sx:.4f} {sy:.4f} {math.degrees(spec.start_yaw):.2f}")


def _main() -> int:
    ap = argparse.ArgumentParser(description="maze registry utilities")
    ap.add_argument("--defs-dir", type=Path, default=None,
                    help="maze_defs directory (default config.MAZE_DEFS_DIR)")
    ap.add_argument("--list", action="store_true",
                    help="print a summary table of all mazes")
    ap.add_argument("--spawn", metavar="NAME", default=None,
                    help="print 'WORLD X Y YAW_DEG' for the spawn script")
    ap.add_argument("--multi", action="store_true",
                    help="with --spawn: place in the combined multi-maze world")
    args = ap.parse_args()

    registry = load_registry(args.defs_dir)
    if args.spawn is not None:
        _print_spawn(registry, args.spawn, args.multi)
    else:
        _print_list(registry)          # default action (also --list)
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
