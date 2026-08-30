#!/usr/bin/env python3
"""
decode_maze_pngs.py — Decode mazegenerator.net maze PNGs into wall-segment
maze definitions for the Gazebo explore-then-exit task.

For every source PNG (``mazes_png/*.png``) the pipeline is:

 1. isolate the maze ink (drop title / copyright text by component shape),
 2. extract straight wall strokes per direction (0°, 60°, 90°, 120°) with
    rho-binning + run extraction + fill-ratio validation (phantom lines that
    merely chain junction pixels are rejected),
 3. snap stroke endpoints to shared vertices so Gazebo wall boxes meet
    exactly (no seams — same invariant as generate_maze_sdf.py),
 4. map into a maze-local world frame: same footprint as nhom8_maze75
    (longest side = 3.93 m), walls 0.03 m thick / 0.40 m tall,
    image "up" (top of the drawing) = world +y,
 5. find the two border openings (entrance + exit): flood fill with a
    sealed ring on the bbox edge; the interior component touches the ring
    ONLY through real openings,
 6. build 25 coverage zones: ortho mazes use their exact 5×5 cell grid,
    sigma/delta use a multi-source BFS watershed over the inflated free
    space (farthest-point seeds) — pure world construction, no planner in
    the learning loop,
 7. verify: both gaps connect to the interior, every zone is reachable
    from the start cell, corridor clearance fits the robot,
 8. emit ``worlds/maze_defs/<name>.json`` + a red-on-gray overlay render
    (``rl_training/reports/maze_overlays/<name>.png``) for visual diff.

Run:  ./rl_venv/bin/python scripts/decode_maze_pngs.py
"""

from __future__ import annotations

import json
import math
import sys
from collections import deque
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

WS_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = WS_ROOT / "mazes_png"
OUT_DIR = WS_ROOT / "worlds" / "maze_defs"
OVERLAY_DIR = WS_ROOT / "rl_training" / "reports" / "maze_overlays"

# World-frame maze constants (must match rl_training.config EXPL_*)
WALL_T = 0.03          # wall thickness (m)
WALL_H = 0.40          # wall height (m)
FOOTPRINT = 3.93       # longest maze side (m), like the original 5x5 maze
ROBOT_R = 0.10         # robot collision radius (m)
CLEARANCE = 0.06       # planning margin inside corridors (m)
N_ZONES = 25           # coverage cells per maze
RASTER_RES = 0.025     # world raster resolution (m)

DIRECTIONS = {  # theta (deg) -> unit vector (image coords, y down)
    0: (1.0, 0.0),
    60: (0.5, math.sqrt(3) / 2),
    90: (0.0, 1.0),
    120: (-0.5, math.sqrt(3) / 2),
}


# ────────────────────────────────────────────────────────────────────────────
# 1. ink isolation
# ────────────────────────────────────────────────────────────────────────────

def load_ink(png: Path) -> np.ndarray:
    """Binary maze-ink mask: black strokes, no title/copyright text."""
    mask = np.asarray(Image.open(png).convert("L")) < 128
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    out = np.zeros_like(mask, dtype=bool)
    for sy in range(h):
        for sx in np.nonzero(mask[sy] & ~seen[sy])[0]:
            if seen[sy, sx]:
                continue
            q = deque([(sy, sx)])
            seen[sy, sx] = True
            comp = [(sy, sx)]
            minx = maxx = sx
            miny = maxy = sy
            while q:
                y, x = q.popleft()
                minx, maxx = min(minx, x), max(maxx, x)
                miny, maxy = min(miny, y), max(maxy, y)
                for dy in (-1, 0, 1):
                    for dx in (-1, 0, 1):
                        ny, nx = y + dy, x + dx
                        if 0 <= ny < h and 0 <= nx < w and mask[ny, nx] \
                                and not seen[ny, nx]:
                            seen[ny, nx] = True
                            q.append((ny, nx))
                            comp.append((ny, nx))
            extent = max(maxx - minx, maxy - miny)
            # walls: elongated strokes or big blobs; text: small compact glyphs
            if extent >= 30 or len(comp) >= 1000:
                for y, x in comp:
                    out[y, x] = True
    if out.sum() < 1000:
        raise ValueError(f"{png.name}: maze ink not found")
    return out


# ────────────────────────────────────────────────────────────────────────────
# 2.-3. stroke extraction + vertex snapping
# ────────────────────────────────────────────────────────────────────────────

@dataclass
class Seg:
    x0: float
    y0: float
    x1: float
    y1: float
    theta: int

    @property
    def length(self) -> float:
        return math.hypot(self.x1 - self.x0, self.y1 - self.y0)


def extract_direction(mask: np.ndarray, theta: int, t_px: float,
                      min_len: float) -> list[Seg]:
    """Strokes parallel to ``theta`` via support-mask + rho grouping.

    A pixel "supports" direction theta when the ink run THROUGH the pixel
    along theta exceeds 2·t_px — moving along the wall direction stays
    inside the stroke, so straight walls light up and corners/junction
    blobs of other walls do not (they are only ~t_px long in any given
    direction). Supported pixels are grouped by perpendicular offset
    (rho) into lines, split into runs along the direction, and validated
    by a 1-D fill ratio.
    """
    ux, uy = DIRECTIONS[theta]
    nx, ny = -uy, ux
    h, w = mask.shape
    k_half = max(2, int(round(1.5 * t_px)))          # support half-length (px)
    # support[p] = ink at p and at every shift of −k..k along (ux, uy)
    support = mask.copy()
    for k in range(1, k_half + 1):
        dx, dy = int(round(k * ux)), int(round(k * uy))
        shifted = np.zeros_like(mask)
        ys0, ys1 = max(0, dy), h + min(0, dy)
        xs0, xs1 = max(0, dx), w + min(0, dx)
        shifted[ys0:ys1, xs0:xs1] = mask[ys0 - dy:ys1 - dy, xs0 - dx:xs1 - dx]
        support &= shifted
        shifted = np.zeros_like(mask)
        ys0, ys1 = max(0, -dy), h + min(0, -dy)
        xs0, xs1 = max(0, -dx), w + min(0, -dx)
        shifted[ys0:ys1, xs0:xs1] = mask[ys0 + dy:ys1 + dy, xs0 + dx:xs1 + dx]
        support &= shifted
    sup = mask & support
    ys, xs = np.nonzero(sup)
    if len(xs) == 0:
        return []
    rho = xs * nx + ys * ny
    r0 = float(rho.min())
    # cluster contiguous rho bins into lines (wall thickness in rho ≈ t_px)
    order = np.argsort(rho)
    lines: list[list[int]] = []
    cur = [int(order[0])]
    for idx in order[1:]:
        if rho[idx] - rho[cur[-1]] <= 1.5:
            cur.append(int(idx))
        else:
            lines.append(cur)
            cur = [int(idx)]
    lines.append(cur)
    segs: list[Seg] = []
    for line in lines:
        if len(line) < 0.5 * min_len:
            continue
        center = float(np.mean(rho[line]))
        half = max(t_px / 2.0, 2.0)
        tvals = xs[line] * ux + ys[line] * uy
        tv = np.sort(tvals)
        starts, ends = [tv[0]], []
        for a, b in zip(tv[:-1], tv[1:]):
            if b - a > t_px:          # junction blobs are ≈ t_px deep
                ends.append(a)
                starts.append(b)
        ends.append(tv[-1])
        for a, b in zip(starts, ends):
            L = b - a
            if L < min_len:
                continue
            fill = np.count_nonzero((tvals >= a) & (tvals <= b)) / L
            if fill < 0.55:
                continue
            segs.append(Seg(a * ux + center * nx, a * uy + center * ny,
                            b * ux + center * nx, b * uy + center * ny,
                            theta))
    return segs


def estimate_thickness(mask: np.ndarray) -> float:
    """Wall thickness (px) from the densest horizontal rho-line.

    All three maze families draw horizontal walls, and a straight border
    stroke gives a clean rho cluster whose width IS the stroke thickness
    (diagonal walls smear across rho bins and are ignored).
    """
    ux, uy = DIRECTIONS[0]
    nx, ny = -uy, ux
    ys, xs = np.nonzero(mask)
    rho = xs * nx + ys * ny
    r0 = float(rho.min())
    bins = np.zeros(int(math.ceil(rho.max() - r0)) + 3, dtype=np.int32)
    np.add.at(bins, np.round(rho - r0).astype(int), 1)
    peak = int(bins.argmax())
    width = 1
    lo = peak
    while lo - 1 >= 0 and bins[lo - 1] >= 0.60 * bins[peak]:
        lo -= 1
        width += 1
    hi = peak
    while hi + 1 < len(bins) and bins[hi + 1] >= 0.60 * bins[peak]:
        hi += 1
        width += 1
    return float(min(30, max(5, width)))

def snap_vertices(segs: list[Seg], tol: float) -> list[Seg]:
    """Snap stroke endpoints within ``tol`` to shared cluster centroids."""
    pts: list[tuple[float, float]] = []
    for s in segs:
        pts.append((s.x0, s.y0))
        pts.append((s.x1, s.y1))
    parent = list(range(len(pts)))

    def find(a: int) -> int:
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    # union-find over endpoints inside the tolerance (grid-hash for speed)
    cell = {}
    for i, (x, y) in enumerate(pts):
        key = (int(x // tol), int(y // tol))
        for dk in ((0, 0), (-1, 0), (1, 0), (0, -1), (0, 1),
                   (-1, -1), (1, 1), (-1, 1), (1, -1)):
            k2 = (key[0] + dk[0], key[1] + dk[1])
            if k2 in cell:
                for j in cell[k2]:
                    if math.hypot(pts[i][0] - pts[j][0],
                                  pts[i][1] - pts[j][1]) <= tol:
                        parent[find(i)] = find(j)
            cell.setdefault(k2, []).append(i)
    groups: dict[int, list[int]] = {}
    for i in range(len(pts)):
        groups.setdefault(find(i), []).append(i)
    means = {r: (float(np.mean([pts[i][0] for i in g])),
                 float(np.mean([pts[i][1] for i in g]))) for r, g in groups.items()}
    out = []
    for k, s in enumerate(segs):
        m0 = means[find(2 * k)]
        m1 = means[find(2 * k + 1)]
        if math.hypot(m0[0] - m1[0], m0[1] - m1[1]) < 0.02:
            continue  # degenerate (zero-length after snap)
        out.append(Seg(m0[0], m0[1], m1[0], m1[1], s.theta))
    return out


def _point_seg_dist(px: float, py: float, s: Seg) -> float:
    vx, vy = s.x1 - s.x0, s.y1 - s.y0
    L2 = vx * vx + vy * vy
    if L2 < 1e-12:
        return math.hypot(px - s.x0, py - s.y0)
    t = ((px - s.x0) * vx + (py - s.y0) * vy) / L2
    t = max(0.0, min(1.0, t))
    return math.hypot(px - (s.x0 + t * vx), py - (s.y0 + t * vy))


def weld_ends(segs: list[Seg], t_px: float, k_ext: float,
              min_len: float) -> list[Seg]:
    """Re-grow support-eroded wall ends and weld T/corner junctions.

    The support mask erodes every stroke by ``k_ext`` px at each end. An
    eroded end is extended back along its direction ONLY while the new tip
    approaches another wall (junction closure); ends facing a real passage
    stay put so openings are not sealed. Endpoints that already touch
    another wall are snapped onto it (shared junction coordinate -> sealed
    Gazebo joints).
    """
    segs = [Seg(s.x0, s.y0, s.x1, s.y1, s.theta) for s in segs]
    tol_snap = 0.9 * t_px

    def others(i: int) -> list[Seg]:
        return [s for j, s in enumerate(segs) if j != i]

    for i, s in enumerate(segs):
        L = s.length
        if L < 1e-6:
            continue
        ux, uy = (s.x1 - s.x0) / L, (s.y1 - s.y0) / L
        for sign, (ex, ey) in ((1.0, (s.x1, s.y1)), (-1.0, (s.x0, s.y0))):
            oth = [o for j, o in enumerate(segs) if j != i]
            if not oth:
                continue
            d_now = min(_point_seg_dist(ex, ey, o) for o in oth)
            if d_now <= tol_snap:
                # snap onto the nearest wall point
                best = min(oth, key=lambda o: _point_seg_dist(ex, ey, o))
                vx, vy = best.x1 - best.x0, best.y1 - best.y0
                L2 = vx * vx + vy * vy
                t = ((ex - best.x0) * vx + (ey - best.y0) * vy) / L2
                t = max(0.0, min(1.0, t))
                qx, qy = best.x0 + t * vx, best.y0 + t * vy
                if sign > 0:
                    segs[i].x1, segs[i].y1 = qx, qy
                else:
                    segs[i].x0, segs[i].y0 = qx, qy
                continue
            # conditional extension: grow in steps while approaching a wall
            step = k_ext / 4.0
            cx, cy = ex, ey
            best_d, best_pt = d_now, (ex, ey)
            for _ in range(4):
                nx_, ny_ = cx + sign * step * ux, cy + sign * step * uy
                dd = min(_point_seg_dist(nx_, ny_, o) for o in oth)
                if dd < best_d:
                    best_d, best_pt = dd, (nx_, ny_)
                else:
                    break
                cx, cy = nx_, ny_
            if sign > 0:
                segs[i].x1, segs[i].y1 = best_pt
            else:
                segs[i].x0, segs[i].y0 = best_pt
    # final snap pass (shared junctions)
    out = snap_vertices(segs, tol=tol_snap)
    return [s for s in out if s.length >= 0.5 * min_len]


# ────────────────────────────────────────────────────────────────────────────
# 4. world mapping
# ────────────────────────────────────────────────────────────────────────────

def to_world(segs: list[Seg], bbox: tuple[int, int, int, int]
             ) -> tuple[list[Seg], float]:
    """Map image segments to maze-local world frame (bbox center → origin,
    y up). Returns (segments, scale)."""
    x0, y0, x1, y1 = bbox
    w, h = x1 - x0, y1 - y0
    scale = FOOTPRINT / max(w, h)
    cx, cy = (x0 + x1) / 2.0, (y0 + y1) / 2.0
    out = []
    for s in segs:
        wx0 = (s.x0 - cx) * scale
        wy0 = (cy - s.y0) * scale
        wx1 = (s.x1 - cx) * scale
        wy1 = (cy - s.y1) * scale
        out.append(Seg(wx0, wy0, wx1, wy1, s.theta))
    return out, scale


# ────────────────────────────────────────────────────────────────────────────
# 5.-6. raster helpers: gaps, zones, start
# ────────────────────────────────────────────────────────────────────────────

class Raster:
    """World raster of one maze: walls, inflation, bounds."""

    def __init__(self, segs: list[Seg], bounds: tuple[float, float, float, float],
                 res: float = RASTER_RES):
        self.res = res
        self.bounds = bounds  # xmin, ymin, xmax, ymax (maze-local)
        xmin, ymin, xmax, ymax = bounds
        self.nx = int(round((xmax - xmin) / res)) + 1
        self.ny = int(round((ymax - ymin) / res)) + 1
        self.wall = np.zeros((self.ny, self.nx), dtype=bool)
        half = WALL_T / 2.0
        for s in segs:
            L = s.length
            n = max(2, int(L / (res / 2)))
            ux, uy = (s.x1 - s.x0) / L, (s.y1 - s.y0) / L
            px, py = -uy, ux
            for t in np.linspace(0, L, n):
                bx = s.x0 + t * ux
                by = s.y0 + t * uy
                for u in np.linspace(-half, half, 3):
                    ix = int(round((bx + u * px - xmin) / res))
                    iy = int(round((by + u * py - ymin) / res))
                    if 0 <= ix < self.nx and 0 <= iy < self.ny:
                        self.wall[iy, ix] = True

    def ix(self, x: float) -> int:
        return int(min(max(round((x - self.bounds[0]) / self.res), 0), self.nx - 1))

    def iy(self, y: float) -> int:
        return int(min(max(round((y - self.bounds[1]) / self.res), 0), self.ny - 1))

    def xy(self, ix: int, iy: int) -> tuple[float, float]:
        return (self.bounds[0] + ix * self.res,
                self.bounds[1] + iy * self.res)

    def dilate(self, mask: np.ndarray, r_cells: int) -> np.ndarray:
        out = mask.copy()
        for _ in range(r_cells):
            m = out
            out = m.copy()
            out[1:, :] |= m[:-1, :]
            out[:-1, :] |= m[1:, :]
            out[:, 1:] |= m[:, :-1]
            out[:, :-1] |= m[:, 1:]
        return out

    def flood(self, seeds: list[tuple[int, int]], blocked: np.ndarray
              ) -> np.ndarray:
        """4-connected flood on ~blocked. Returns visited mask."""
        seen = np.zeros_like(blocked, dtype=bool)
        q = deque()
        for ix, iy in seeds:
            if 0 <= ix < self.nx and 0 <= iy < self.ny \
                    and not blocked[iy, ix] and not seen[iy, ix]:
                seen[iy, ix] = True
                q.append((ix, iy))
        while q:
            x, y = q.popleft()
            for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                nx_, ny_ = x + dx, y + dy
                if 0 <= nx_ < self.nx and 0 <= ny_ < self.ny \
                        and not blocked[ny_, nx_] and not seen[ny_, nx_]:
                    seen[ny_, nx_] = True
                    q.append((nx_, ny_))
        return seen

def find_openings_from_ink(mask: np.ndarray, bbox: tuple[int, int, int, int],
                           t_px: float, scale: float,
                           wall_bbox: tuple[float, float, float, float]
                           ) -> list[dict]:
    """Entrance/exit gap centers from the top/bottom outline bands.

    mazegenerator draws the entrance on the top outline and the exit on the
    bottom outline for every maze family. The outermost 2·t_px image rows
    of the drawing are covered with ink everywhere except at the gap, so a
    free run in the top band IS the entrance (bottom band → exit).
    Returns gap centers in maze-local world coordinates.
    """
    x0, y0, x1, y1 = bbox
    depth = max(3, int(round(2.0 * t_px)))
    cx_img = (x0 + x1) / 2.0
    cy_img = (y0 + y1) / 2.0

    def band_gap(band_top: int, band_bot: int) -> tuple[float, float]:
        """Widest ink-free column run inside the band → gap center (img)."""
        band = mask[band_top:band_bot, x0:x1 + 1]
        col_ink = band.any(axis=0)
        runs = []
        start = None
        for i, v in enumerate(col_ink):
            if not v and start is None:
                start = i
            elif v and start is not None:
                runs.append((start, i))
                start = None
        if start is not None:
            runs.append((start, len(col_ink)))
        # ignore tiny anti-alias slivers
        runs = [r for r in runs if r[1] - r[0] >= 3.0 * t_px]
        if not runs:
            raise ValueError("no border gap found in outline band")
        a, b = max(runs, key=lambda r: r[1] - r[0])
        return (x0 + (a + b) / 2.0, band_top)

    gx_top, _ = band_gap(y0, y0 + depth)
    gx_bot, _ = band_gap(y1 - depth + 1, y1 + 1)
    # image → maze-local world (bbox center = origin, y up)
    def to_world(ix_img: float, iy_img: float) -> tuple[float, float]:
        return ((ix_img - cx_img) * scale, (cy_img - iy_img) * scale)

    entrance = to_world(gx_top, y0)
    exit_ = to_world(gx_bot, y1)
    # project onto the wall bbox border line
    entrance = (entrance[0], wall_bbox[3])
    exit_ = (exit_[0], wall_bbox[1])
    return [
        {"role": "entrance", "xy": [round(entrance[0], 4), round(entrance[1], 4)]},
        {"role": "exit", "xy": [round(exit_[0], 4), round(exit_[1], 4)]},
    ]


def _outline_depth_side(mask: np.ndarray, bbox: tuple[int, int, int, int],
                        t_px: float, scale: float, maxdepth: int,
                        edge_y: int, inward: float) -> dict | None:
    """One side's mouth from the outline depth profile, or None.

    Per column the first-ink depth from the bbox edge is profiled up to
    ``maxdepth``; absent ink → sentinel ``maxdepth`` (an outline that
    merely slides deeper than the window — hexagon-corner recessions —
    then never registers as a jump). A column is a mouth candidate when
    its depth exceeds 3·t_px. Candidate runs are split at abrupt jumps
    (> 3·t_px between adjacent columns) and only sub-runs bounded by
    jumps on BOTH ends qualify: the sigma V/funnel mouth's flanking
    strokes end abruptly, corner recessions ramp away gradually. The
    widest qualifying run wins; its width (raw, unclamped) is the
    half_width·2 and the mouth line sits at the edge minus the tip depth
    (min first-ink depth of the columns adjacent to the run) — NOT the
    bbox border line, onto which recessed mouths must not be projected.
    """
    x0, y0, x1, y1 = bbox
    cx_img = (x0 + x1) / 2.0
    cy_img = (y0 + y1) / 2.0
    K = 3.0 * t_px
    if inward > 0:
        band = mask[edge_y:edge_y + maxdepth + 1, x0:x1 + 1]
    else:
        band = mask[max(0, edge_y - maxdepth):edge_y + 1, x0:x1 + 1][::-1]
    has = band.any(axis=0)
    prof = np.argmax(band, axis=0).astype(np.float64)
    prof[~has] = float(maxdepth)               # absent → sentinel
    n = len(prof)
    jumps = np.zeros(n, dtype=bool)            # jumps[i]: Δ between i-1, i
    jumps[1:] = np.abs(np.diff(prof)) > K
    cand = prof > K
    runs, start = [], None
    for i in range(n):
        if cand[i]:
            if start is None:
                start = i
        elif start is not None:
            runs.append((start, i))
            start = None
    if start is not None:
        runs.append((start, n))
    best = None
    for a, b in runs:
        cuts = [a] + [i for i in range(a + 1, b) if jumps[i]] + [b]
        for ca, cb in zip(cuts[:-1], cuts[1:]):
            if (ca > 0 and jumps[ca] and cb < n and jumps[cb]
                    and (best is None or cb - ca > best[1] - best[0])):
                best = (ca, cb)
    if best is None:
        return None
    a, b = best
    if not np.isfinite(prof[a - 1]) or not np.isfinite(prof[b]):
        return None                            # no finite tip → no mouth line
    tip = min(prof[a - 1], prof[b])
    wx = (x0 + (a + b) / 2.0 - cx_img) * scale
    wy = (cy_img - (edge_y + inward * tip)) * scale
    return {"xy": [round(wx, 4), round(wy, 4)],
            "half_width": round((b - a) * scale / 2.0, 4)}


def find_openings_outline_depth(mask: np.ndarray,
                                bbox: tuple[int, int, int, int],
                                t_px: float, scale: float,
                                wall_bbox: tuple[float, float, float, float]
                                ) -> list[dict | None]:
    """Sigma openings from the outline depth profile (hexagonal zigzag).

    ``find_openings_from_ink`` scans a flat band along each bbox edge; on
    sigma's zigzag outline the widest flat-band background runs are the
    bbox-corner EXTERIOR recessions, not the mouths. This detector
    profiles the first-ink depth per column instead (see
    :func:`_outline_depth_side`) with
    ``maxdepth = 35%`` of the bbox long side, and picks the top side's
    run as the entrance and the bottom side's as the exit — sigma mouths
    are always N/S. Each entry is the usual opening dict plus a raw
    (unclamped) ``half_width``; a side with no qualifying run yields
    None so the caller can fall back to ``find_openings_from_ink``.
    """
    x0, y0, x1, y1 = bbox
    maxdepth = int(0.35 * max(x1 - x0, y1 - y0))
    out = []
    for role, edge_y, inward in (("entrance", y0, 1.0), ("exit", y1, -1.0)):
        op = _outline_depth_side(mask, bbox, t_px, scale, maxdepth,
                                 edge_y, inward)
        if op is not None:
            op = {"role": role, "xy": op["xy"],
                  "half_width": op["half_width"]}
        out.append(op)
    return out


def _point_in_outline(mask: np.ndarray, bbox: tuple[int, int, int, int],
                      ix_img: float, iy_img: float) -> bool:
    """8-ray parity test against the source ink: majority of rays crossing
    an odd number of background→ink run groups ⇒ inside the outline.
    Border mouths let at most two rays escape un-crossed, so the majority
    still classifies interior vs exterior (notch/moat) cells correctly."""
    x0, y0, x1, y1 = bbox
    h, w = mask.shape
    diag = 0.5 * math.sqrt(2.0)
    odd = 0
    for dx, dy in ((1.0, 0.0), (-1.0, 0.0), (0.0, 1.0), (0.0, -1.0),
                   (diag, diag), (diag, -diag), (-diag, diag), (-diag, -diag)):
        runs = 0
        in_ink = False
        k = 1
        while True:
            x = int(round(ix_img + k * dx))
            y = int(round(iy_img + k * dy))
            if not (0 <= x < w and 0 <= y < h) \
                    or not (x0 - 2 <= x <= x1 + 2 and y0 - 2 <= y <= y1 + 2):
                break
            ink = bool(mask[y, x])
            if ink and not in_ink:
                runs += 1
            in_ink = ink
            k += 1
        if runs % 2 == 1:
            odd += 1
    return odd >= 5


def build_zones(ras: Raster, blocked: np.ndarray, n_zones: int,
                inside: tuple[float, float, float, float] = None
                ) -> tuple[np.ndarray, list[tuple[float, float]]]:
    """Multi-source BFS watershed over free space from farthest-point seeds.

    ``inside`` confines zones to the wall bounding box (the moat outside is
    a decoding artifact and must not become a zone).
    """
    free_ix, free_iy = np.nonzero(~blocked)
    pts = list(zip(free_ix.tolist(), free_iy.tolist()))
    if inside is not None:
        ixmin, iymin = ras.ix(inside[0]), ras.iy(inside[1])
        ixmax, iymax = ras.ix(inside[2]), ras.iy(inside[3])
        pts = [(x, y) for x, y in pts
               if ixmin <= x <= ixmax and iymin <= y <= iymax]
    if len(pts) < n_zones * 4:
        raise ValueError("free space too small for zoning")

    def propagate(seed: tuple[int, int], dist: np.ndarray) -> None:
        dist[seed[1], seed[0]] = 0.0
        q = deque([seed])
        while q:
            x, y = q.popleft()
            for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                nx_, ny_ = x + dx, y + dy
                if 0 <= nx_ < ras.nx and 0 <= ny_ < ras.ny \
                        and not blocked[ny_, nx_] \
                        and dist[ny_, nx_] > dist[y, x] + 1:
                    dist[ny_, nx_] = dist[y, x] + 1
                    q.append((nx_, ny_))

    dist = np.full((ras.ny, ras.nx), np.inf)
    center = min(pts, key=lambda p: (p[0] - ras.nx / 2) ** 2
                 + (p[1] - ras.ny / 2) ** 2)
    propagate(center, dist)
    # farthest-point sampling over REACHABLE cells only (sealed pockets
    # behind the clearance inflation must not stall the loop)
    reachable = [(x, y) for x, y in pts if np.isfinite(dist[y, x])]
    seeds = [center]
    for _ in range(n_zones - 1):
        far = max(reachable, key=lambda p: dist[p[1], p[0]])
        if dist[far[1], far[0]] < 2:
            break
        seeds.append(far)
        propagate(far, dist)
    if len(seeds) < n_zones:
        raise ValueError(f"only {len(seeds)} zones fit (need {n_zones})")
    # multi-source BFS labeling
    labels = np.full((ras.ny, ras.nx), -1, dtype=np.int16)
    q = deque()
    for k, (sx, sy) in enumerate(seeds):
        labels[sy, sx] = k
        q.append((sx, sy, k))
    while q:
        x, y, k = q.popleft()
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nx_, ny_ = x + dx, y + dy
            if 0 <= nx_ < ras.nx and 0 <= ny_ < ras.ny \
                    and not blocked[ny_, nx_] and labels[ny_, nx_] == -1:
                labels[ny_, nx_] = k
                q.append((nx_, ny_, k))
    centroids = []
    for k in range(len(seeds)):
        m = labels == k
        if not m.any():
            raise ValueError(f"zone {k} empty")
        iy, ix = np.nonzero(m)
        centroids.append(ras.xy(float(ix.mean()), float(iy.mean())))
    return labels, centroids


def find_start(ras: Raster, blocked: np.ndarray, gap_xy: tuple[float, float],
               inside_fn=None) -> tuple[tuple[float, float], float]:
    """Start pose: inflated-free cell ~0.35 m inward from the entrance.

    With ``inside_fn`` (sigma: point-in-outline parity test on the source
    ink) a candidate cell is only accepted when it lies INSIDE the maze
    outline — exterior cells (V-mouth notch, corner recessions, moat) are
    free in the raster and would otherwise be selectable near the mouth.
    Without it the plain nearest-to-target candidate is kept.
    """
    cx = sum(ras.bounds[::2]) / 2.0
    cy = sum(ras.bounds[1::2]) / 2.0
    gx, gy = gap_xy
    inward = np.array([cx - gx, cy - gy])
    inward /= max(np.linalg.norm(inward), 1e-9)
    target = np.array([gx, gy]) + inward * 0.35
    # BFS geodesic distances from the gap cell over inflated-free space
    free = ~blocked
    dist = np.full((ras.ny, ras.nx), np.inf)
    si, sj = ras.iy(gy), ras.ix(gx)
    # nudge seed to a free cell
    seed = None
    for r in range(0, 8):
        for dy in range(-r, r + 1):
            for dx in range(-r, r + 1):
                iy, ix = si + dy, sj + dx
                if 0 <= iy < ras.ny and 0 <= ix < ras.nx and free[iy, ix]:
                    seed = (ix, iy)
                    break
            if seed:
                break
        if seed:
            break
    if seed is None:
        raise ValueError("entrance gap has no free cell nearby")
    dist[seed[1], seed[0]] = 0.0
    q = deque([seed])
    while q:
        x, y = q.popleft()
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nx_, ny_ = x + dx, y + dy
            if 0 <= nx_ < ras.nx and 0 <= ny_ < ras.ny and free[ny_, nx_] \
                    and dist[ny_, nx_] > dist[y, x] + ras.res:
                dist[ny_, nx_] = dist[y, x] + ras.res
                q.append((nx_, ny_))
    iy, ix = np.nonzero(free & np.isfinite(dist))
    if inside_fn is None:
        best, best_d = None, None
        for k in range(len(ix)):
            px, py = ras.xy(ix[k], iy[k])
            if not (0.25 <= dist[iy[k], ix[k]] <= 0.7):
                continue
            d = math.hypot(px - target[0], py - target[1])
            if best_d is None or d < best_d:
                best, best_d = (px, py), d
    else:
        cands = []
        for k in range(len(ix)):
            px, py = ras.xy(ix[k], iy[k])
            if not (0.25 <= dist[iy[k], ix[k]] <= 0.7):
                continue
            cands.append((math.hypot(px - target[0], py - target[1]),
                          px, py))
        best = None
        for d, px, py in sorted(cands, key=lambda c: c[0]):
            if inside_fn(px, py):
                best = (px, py)
                break
    if best is None:
        raise ValueError("no start cell 0.25–0.7 m from entrance")
    yaw = math.atan2(inward[1], inward[0])
    return best, yaw


# ────────────────────────────────────────────────────────────────────────────
# per-maze driver
# ────────────────────────────────────────────────────────────────────────────

def decode(png: Path, family: str) -> dict:
    mask = load_ink(png)
    ys, xs = np.nonzero(mask)
    bbox = (int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max()))
    t_px = estimate_thickness(mask)
    min_len = max(28.0, 2.2 * t_px)

    if family == "ortho":
        segs_img = decode_ortho(mask, bbox, t_px)
    else:
        segs_img = []
        dirs = (0, 60, 90, 120) if family == "sigma" else (0, 60, 120)
        for theta in dirs:
            segs_img += extract_direction(mask, theta, t_px, min_len)
        segs_img = weld_ends(segs_img, t_px, k_ext=1.6 * t_px,
                             min_len=min_len)

    segs_w, scale = to_world(segs_img, bbox)
    # world segment end extension seals junctions (same trick as the
    # pitch-span walls of generate_maze_sdf.py)
    ext = []
    for s in segs_w:
        L = s.length
        if L < 1e-6:
            continue
        ux, uy = (s.x1 - s.x0) / L, (s.y1 - s.y0) / L
        ext.append(Seg(s.x0 - ux * WALL_T / 2, s.y0 - uy * WALL_T / 2,
                       s.x1 + ux * WALL_T / 2, s.y1 + uy * WALL_T / 2,
                       s.theta))
    xs_w = [c for s in ext for c in (s.x0, s.x1)]
    ys_w = [c for s in ext for c in (s.y0, s.y1)]
    wall_bbox = (min(xs_w), min(ys_w), max(xs_w), max(ys_w))
    margin = 0.30  # moat + ring for opening detection
    bounds = tuple(v + sgn * margin for v, sgn in
                   zip(wall_bbox, (-1, -1, 1, 1)))
    ras = Raster(ext, bounds)
    blocked = ras.dilate(ras.wall, int(round((ROBOT_R + CLEARANCE) / ras.res)))
    if family == "sigma":
        ol = find_openings_outline_depth(mask, bbox, t_px, scale, wall_bbox)
        if ol is None or ol[0] is None or ol[1] is None:
            fb = find_openings_from_ink(mask, bbox, t_px, scale, wall_bbox)
            openings = [
                ol[0] if ol is not None and ol[0] is not None else fb[0],
                ol[1] if ol is not None and ol[1] is not None else fb[1],
            ]
        else:
            openings = ol
    else:
        openings = find_openings_from_ink(mask, bbox, t_px, scale, wall_bbox)
    labels, centroids = build_zones(ras, blocked, N_ZONES, inside=wall_bbox)
    inside_fn = None
    if family == "sigma":
        cx_img = (bbox[0] + bbox[2]) / 2.0
        cy_img = (bbox[1] + bbox[3]) / 2.0

        def inside_fn(wx: float, wy: float) -> bool:
            return _point_in_outline(mask, bbox,
                                     cx_img + wx / scale, cy_img - wy / scale)

    start_xy, start_yaw = find_start(ras, blocked, openings[0]["xy"],
                                     inside_fn=inside_fn)

    # verify: every zone reachable from start (same blocked graph)
    zone_of_start = labels[ras.iy(start_xy[1]), ras.ix(start_xy[0])]
    zone_cells = {int(k): int((labels == k).sum()) for k in range(N_ZONES)}
    sizes = [zone_cells[k] for k in range(N_ZONES)]
    if min(sizes) < 3:
        raise ValueError(f"tiny zone: {min(sizes)} cells")

    # corridor clearance at zone centroids
    free = ~blocked
    clearance = _wall_distance(ras, ras.wall)
    cent_clear = [clearance[ras.iy(c[1]), ras.ix(c[0])] for c in centroids]

    return {
        "name": png.stem.replace("-1", ""),
        "family": family,
        "scale": scale,
        "img_bbox": [int(v) for v in bbox],
        "thickness_px": t_px,
        "bounds": [float(b) for b in bounds],
        "walls": [[round(s.x0, 4), round(s.y0, 4), round(s.x1, 4), round(s.y1, 4)]
                  for s in ext],
        "openings": openings,
        "start_xy": [round(start_xy[0], 4), round(start_xy[1], 4)],
        "start_yaw_deg": round(math.degrees(start_yaw), 2),
        "zone_res": ras.res,
        "zone_labels_shape": [int(labels.shape[0]), int(labels.shape[1])],
        "zone_labels_b64": _encode_labels(labels),
        "zone_centroids": [[round(c[0], 4), round(c[1], 4)] for c in centroids],
        "zone_start": int(zone_of_start),
        "centroid_clearance_min": round(float(min(cent_clear)), 3),
    }


def _wall_distance(ras: Raster, wall: np.ndarray) -> np.ndarray:
    """BFS distance (m) from walls over free cells (approx clearance)."""
    d = np.full((ras.ny, ras.nx), np.inf)
    q = deque()
    iy, ix = np.nonzero(wall)
    for y, x in zip(iy.tolist(), ix.tolist()):
        d[y, x] = 0.0
        q.append((x, y))
    while q:
        x, y = q.popleft()
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nx_, ny_ = x + dx, y + dy
            if 0 <= nx_ < ras.nx and 0 <= ny_ < ras.ny \
                    and d[ny_, nx_] > d[y, x] + ras.res:
                d[ny_, nx_] = d[y, x] + ras.res
                q.append((nx_, ny_))
    return d


def _encode_labels(labels: np.ndarray) -> str:
    import base64, zlib
    flat = labels.astype(np.int16).flatten().tobytes()
    return base64.b64encode(zlib.compress(flat, 6)).decode("ascii")


def decode_ortho(mask: np.ndarray, bbox: tuple[int, int, int, int],
                 t_px: float) -> list[Seg]:
    """Exact 5×5 grid segments in image coords from the ink mask."""
    x0, y0, x1, y1 = bbox
    w, h = x1 - x0, y1 - y0
    pitch = (w - t_px) / 5.0
    pitch_y = (h - t_px) / 5.0
    if abs(pitch - pitch_y) > 3.0:
        raise ValueError(f"ortho maze not square: {pitch} vs {pitch_y}")
    gx = [x0 + t_px / 2 + i * pitch for i in range(6)]
    gy = [y0 + t_px / 2 + j * pitch for j in range(6)]
    segs: list[Seg] = []
    frac_line = _frac_along
    for i in range(6):
        for j in range(5):
            # vertical wall on x=gx[i] between gy[j], gy[j+1]
            f = frac_line(mask, gx[i], gy[j], gx[i], gy[j + 1], t_px / 2)
            if f >= 0.5:
                segs.append(Seg(gx[i], gy[j], gx[i], gy[j + 1], 90))
    for j in range(6):
        for i in range(5):
            f = frac_line(mask, gx[i], gy[j], gx[i + 1], gy[j], t_px / 2)
            if f >= 0.5:
                segs.append(Seg(gx[i], gy[j], gx[i + 1], gy[j], 0))
    # border sanity: only 2 border cells open
    open_north = [i for i in range(5) if not _has(mask, gx[i], gy[0], gx[i + 1], gy[0], t_px / 2)]
    open_south = [i for i in range(5) if not _has(mask, gx[i], gy[5], gx[i + 1], gy[5], t_px / 2)]
    open_west = [j for j in range(5) if not _has(mask, gx[0], gy[j], gx[0], gy[j + 1], t_px / 2)]
    open_east = [j for j in range(5) if not _has(mask, gx[5], gy[j], gx[5], gy[j + 1], t_px / 2)]
    n_open = len(open_north) + len(open_south) + len(open_west) + len(open_east)
    if n_open != 2:
        raise ValueError(f"ortho border openings = {n_open} (N{open_north} "
                         f"S{open_south} W{open_west} E{open_east}), expected 2")
    return segs


def _frac_along(mask: np.ndarray, x0: float, y0: float,
                x1: float, y1: float, half_t: float) -> float:
    n = max(2, int(math.hypot(x1 - x0, y1 - y0)))
    ux, uy = (x1 - x0) / n, (y1 - y0) / n
    px, py = -uy, ux
    hit = 0
    for k in range(1, n - 1):
        bx = x0 + k * ux
        by = y0 + k * uy
        ok = False
        for u in (-half_t, 0.0, half_t):
            xi = int(round(bx + u * px))
            yi = int(round(by + u * py))
            if mask[yi, xi]:
                ok = True
                break
        hit += ok
    return hit / max(1, n - 2)


def _has(mask: np.ndarray, x0: float, y0: float, x1: float, y1: float,
         half_t: float) -> bool:
    return _frac_along(mask, x0, y0, x1, y1, half_t) >= 0.5


# ────────────────────────────────────────────────────────────────────────────
# overlay render (visual diff) + CLI
# ────────────────────────────────────────────────────────────────────────────

def render_overlay(png: Path, segs_world: list[Seg], scale: float,
                   bbox: tuple[int, int, int, int], out: Path,
                   openings: list[dict] | None = None) -> None:
    """Red overlay of decoded walls on the source image (visual diff).

    ``openings`` (sigma only) additionally marks each detected mouth with
    an orange disc.
    """
    mask = np.asarray(Image.open(png).convert("L"))
    img = np.full(mask.shape + (3,), 255, dtype=np.uint8)
    img[mask < 200] = (210, 210, 210)
    cx = (bbox[0] + bbox[2]) / 2.0
    cy = (bbox[1] + bbox[3]) / 2.0
    for s in segs_world:
        # world → image (inverse of to_world)
        ax = cx + s.x0 / scale
        ay = cy - s.y0 / scale
        bx = cx + s.x1 / scale
        by = cy - s.y1 / scale
        n = int(max(abs(bx - ax), abs(by - ay))) + 1
        for t in np.linspace(0, 1, max(2, n)):
            xi = int(round(ax + t * (bx - ax)))
            yi = int(round(ay + t * (by - ay)))
            if 0 <= yi < img.shape[0] and 0 <= xi < img.shape[1]:
                img[yi, xi] = (255, 0, 0)
    for op in openings or []:
        mxi = int(round(cx + op["xy"][0] / scale))
        myi = int(round(cy - op["xy"][1] / scale))
        for dy in range(-5, 6):
            for dx in range(-5, 6):
                if dx * dx + dy * dy <= 25 \
                        and 0 <= myi + dy < img.shape[0] \
                        and 0 <= mxi + dx < img.shape[1]:
                    img[myi + dy, mxi + dx] = (255, 140, 0)
    Image.fromarray(img).save(out)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    OVERLAY_DIR.mkdir(parents=True, exist_ok=True)
    results = []
    for png in sorted(SRC_DIR.glob("*.png")):
        family = png.stem.split("_")[0]
        try:
            d = decode(png, family)
        except Exception as exc:  # keep going, report at the end
            print(f"[FAIL] {png.name}: {exc}")
            results.append((png.name, False, str(exc)))
            continue
        out = OUT_DIR / f"{d['name']}.json"
        out.write_text(json.dumps(d, indent=1))
        segs = [Seg(*w, 0) for w in d["walls"]]
        render_overlay(png, segs, d["scale"], tuple(d["img_bbox"]),
                       OVERLAY_DIR / f"{d['name']}.png",
                       openings=d["openings"] if d["family"] == "sigma"
                       else None)
        print(f"[OK] {d['name']:>8s} walls={len(d['walls']):3d} "
              f"start=({d['start_xy'][0]:+.2f},{d['start_xy'][1]:+.2f}) "
              f"yaw={d['start_yaw_deg']:+6.1f}° "
              f"clear={d['centroid_clearance_min']:.2f}m")
        results.append((d["name"], True, ""))
    failed = [r for r in results if not r[1]]
    print(f"\n{len(results) - len(failed)}/{len(results)} mazes decoded"
          + (f" — FAILURES: {[f[0] for f in failed]}" if failed else ""))
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
