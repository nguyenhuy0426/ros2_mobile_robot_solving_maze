#!/usr/bin/env python3
"""
pdf_maze_extract.py — Extract exact maze wall geometry from mazegenerator.net PDFs.

The maze PDFs draw every wall as a vector line segment (`x1 y1 m x2 y2 l`)
inside a Flate-compressed content stream — exact coordinates, no image
guesswork. This script decompresses the FIRST page's stream (page 2 is the
solution overlay and must be ignored), classifies the maze type, and emits
one JSON maze definition per PDF into ``maze_defs/``:

  orthogonal — axis-aligned segments snapped to the 6×6 gridlines:
      {vwalls: {gridline: [rows]}, hwalls: {gridline: [cols]},
       gaps: [{side, index}]}
  angled (sigma/delta) — raw wall segments in maze units (scaled so the
      bounding box spans 5 "cell" units):
      {segments: [[x1, y1, x2, y2], ...], gaps: [...]}

Coordinates: x right, y DOWN is avoided — PDF y is up, so PDF bottom maps to
maze row 0 (south) directly. Border gaps (entrance/exit) are detected as
missing spans on the bounding-box border lines.

Usage:  ./rl_venv/bin/python scripts/pdf_maze_extract.py
"""

from __future__ import annotations

import json
import re
import zlib
from pathlib import Path

WS = Path(__file__).resolve().parents[1]
MAZE_DIR = WS / "maze"
OUT_DIR = WS / "maze_defs"

EPS = 1e-6


# ── PDF plumbing ─────────────────────────────────────────────────────────

def _page1_stream(pdf_path: Path) -> bytes:
    """Decompress the FIRST page's content stream (not the solution page)."""
    raw = pdf_path.read_bytes()
    streams = [m.group(1) for m in re.finditer(rb"stream\r?\n(.*?)endstream", raw, re.S)]
    for data in streams:
        try:
            text = zlib.decompress(data)
        except zlib.error:
            continue
        if b" l\n" not in text and b" l\r" not in text and b" l " not in text:
            continue  # not a path-drawing stream
        # The solution page draws the same maze PLUS a route overlay; the
        # maze page comes first in mazegenerator PDFs. Sanity guard: the
        # first path stream is the maze page.
        return text
    raise ValueError(f"no path-drawing content stream in {pdf_path.name}")


def extract_segments(pdf_path: Path) -> list[tuple[float, float, float, float]]:
    """All line segments (x1, y1, x2, y2) of the maze page, in PDF points."""
    text = _page1_stream(pdf_path).decode("latin-1")
    # Stop at the first text block (BT) — walls come before any labels.
    text = text.split("BT")[0]
    segs = []
    for m in re.finditer(
            r"([\d.]+)\s+([\d.]+)\s+m\s*\n?\s*([\d.]+)\s+([\d.]+)\s+l", text):
        x1, y1, x2, y2 = (float(v) for v in m.groups())
        segs.append((x1, y1, x2, y2))
    if not segs:
        raise ValueError(f"no wall segments found in {pdf_path.name}")
    return segs


# ── Orthogonal decoding ──────────────────────────────────────────────────

def decode_orthogonal(segs):
    """Snap axis-aligned segments to the 6×6 gridline lattice.

    Returns (vwalls, hwalls, gaps, n) where vwalls[x_line] = sorted rows,
    hwalls[y_line] = sorted cols, gaps = border openings as dicts, and n = 5.
    """
    xs = sorted({round(s[0], 4) for s in segs}
                | {round(s[2], 4) for s in segs})
    ys = sorted({round(s[1], 4) for s in segs}
                | {round(s[3], 4) for s in segs})
    # Gridlines = distinct coordinates that repeat across many segments.
    def gridlines(vals):
        lines = []
        for v in sorted(set(vals)):
            if not lines or v - lines[-1] > 5.0:
                lines.append(v)
        return lines
    gx = gridlines(xs)
    gy = gridlines(ys)
    if len(gx) != 6 or len(gy) != 6:
        raise ValueError(f"expected 6×6 gridlines, got {len(gx)}×{len(gy)}")
    cell_x = (gx[-1] - gx[0]) / 5.0
    cell_y = (gy[-1] - gy[0]) / 5.0

    vwalls: dict[int, list[int]] = {i: [] for i in range(6)}
    hwalls: dict[int, list[int]] = {j: [] for j in range(6)}

    for x1, y1, x2, y2 in segs:
        if abs(y1 - y2) < EPS:  # horizontal wall on y-gridline gy[j]
            j = min(range(6), key=lambda k: abs(y1 - gy[k]))
            if abs(y1 - gy[j]) > 2.0:
                continue  # stray line (should not happen in these PDFs)
            a, b = sorted((x1, x2))
            cols = [i for i in range(5)
                    if gx[i] + 2.0 <= a and b <= gx[i + 1] - 2.0
                    or abs(a - gx[i]) < 2.0 and abs(b - gx[i + 1]) < 2.0]
            # robust: segment must span exactly one cell
            for i in range(5):
                if abs(a - gx[i]) < 2.0 and abs(b - gx[i + 1]) < 2.0:
                    if i not in hwalls[j]:
                        hwalls[j].append(i)
        elif abs(x1 - x2) < EPS:  # vertical wall on x-gridline gx[i]
            i = min(range(6), key=lambda k: abs(x1 - gx[k]))
            if abs(x1 - gx[i]) > 2.0:
                continue
            a, b = sorted((y1, y2))
            for j in range(5):
                if abs(a - gy[j]) < 2.0 and abs(b - gy[j + 1]) < 2.0:
                    if j not in vwalls[i]:
                        vwalls[i].append(j)
        else:
            raise ValueError("non-axis-aligned segment in orthogonal maze")

    # Border gaps: a border line index with fewer than 5 spans has gaps.
    def missing_border_spans(side):
        if side in ("north", "south"):
            j = 5 if side == "north" else 0
            present = set(hwalls[j])
        else:
            i = 5 if side == "east" else 0
            present = set(vwalls[i])
        return [c for c in range(5) if c not in present]

    gaps = []
    for side in ("north", "south", "west", "east"):
        for idx in missing_border_spans(side):
            gaps.append({"side": side, "index": idx})
    if len(gaps) != 2:
        raise ValueError(f"expected exactly 2 border gaps, found {gaps}")

    # Entrance: prefer north, then west; exit = the other gap.
    pri = {"north": 0, "west": 1, "south": 2, "east": 3}
    gaps.sort(key=lambda g: pri[g["side"]])
    return ({"type": "orthogonal", "n": 5,
             "vwalls": {str(k): sorted(v) for k, v in vwalls.items()},
             "hwalls": {str(k): sorted(v) for k, v in hwalls.items()},
             "entrance": gaps[0], "exit": gaps[1]}, cell_x)


def orthogonal_cell_graph(defn):
    """BFS connectivity over cells; verifies the maze matches the PDF walls."""
    vwalls = {int(k): set(v) for k, v in defn["vwalls"].items()}
    hwalls = {int(k): set(v) for k, v in defn["hwalls"].items()}

    def open_neighbors(c, r):
        out = []
        if c not in vwalls.get(c, {0, 1, 2, 3, 4}):
            pass
        # west edge open iff gridline c has no wall at row r
        if r not in vwalls.get(c, set()):
            out.append((c - 1, r))
        if r not in vwalls.get(c + 1, set()):
            out.append((c + 1, r))
        if c not in hwalls.get(r, set()):
            out.append((c, r - 1))
        if c not in hwalls.get(r + 1, set()):
            out.append((c, r + 1))
        return [(a, b) for a, b in out if 0 <= a < 5 and 0 <= b < 5]

    start = _entrance_cell(defn)
    seen = {start}
    stack = [start]
    while stack:
        c, r = stack.pop()
        for nb in open_neighbors(c, r):
            if nb not in seen:
                seen.add(nb)
                stack.append(nb)
    return seen


def _entrance_cell(defn):
    """Cell adjacent to the entrance gap (inside the maze)."""
    g = defn["entrance"]
    if g["side"] == "north":
        return (g["index"], 4)
    if g["side"] == "south":
        return (g["index"], 0)
    if g["side"] == "west":
        return (0, g["index"])
    return (4, g["index"])


def _exit_cell(defn):
    g = defn["exit"]
    if g["side"] == "north":
        return (g["index"], 4)
    if g["side"] == "south":
        return (g["index"], 0)
    if g["side"] == "west":
        return (0, g["index"])
    return (4, g["index"])


# ── Sigma / delta (angled) decoding ──────────────────────────────────────

def decode_angled(segs, name):
    """Angled mazes: keep raw segments, normalised to a 0..5 box.

    Coordinates are normalised so the maze bounding box maps to [0, 5]²
    ("cell units"). PDF y-up is kept (row 0 at the bottom).
    """
    xs = [s[0] for s in segs] + [s[2] for s in segs]
    ys = [s[1] for s in segs] + [s[3] for s in segs]
    x0, x1 = min(xs), max(xs)
    y0, y1 = min(ys), max(ys)
    scale = 5.0 / max(x1 - x0, y1 - y0)
    norm = [((a - x0) * scale, (b - y0) * scale,
             (c - x0) * scale, (d - y0) * scale)
            for a, b, c, d in segs]
    # Border gaps: endpoints lying ON the bounding-box border lines whose
    # neighbours along the border are far away (> 0.8 cell) — i.e. openings.
    gaps = _detect_angled_gaps(norm)
    return {"type": name.split()[1], "n": 5, "segments":
            [[round(v, 4) for v in s] for s in norm], "gaps": gaps}


def _detect_angled_gaps(segs):
    """Find openings on each bounding-box side of an angled maze.

    A side's border is traced by the wall endpoints that lie on it; a gap is
    a span between consecutive border endpoints (or box corner) wider than
    0.6 cell that is NOT covered by any wall segment running along that side.
    """
    e = 0.05  # "on the border" tolerance in cell units
    out = []

    def covered_along(vals, lo, hi, axis):
        """Any segment collinear with the border covering [lo, hi] range?"""
        for x1, y1, x2, y2 in segs:
            if axis == "x":  # horizontal border (y fixed)
                if abs(y1 - vals) < e and abs(y2 - vals) < e:
                    a, b = sorted((x1, x2))
                    if a <= lo + e and b >= hi - e:
                        return True
            else:            # vertical border (x fixed)
                if abs(x1 - vals) < e and abs(x2 - vals) < e:
                    a, b = sorted((y1, y2))
                    if a <= lo + e and b >= hi - e:
                        return True
        return False

    xs = [s[0] for s in segs] + [s[2] for s in segs]
    ys = [s[1] for s in segs] + [s[3] for s in segs]
    x0, x1, y0, y1 = min(xs), max(xs), min(ys), max(ys)

    for side, (axis, val, lo_hi) in {
        "north": ("x", y1, (x0, x1)),
        "south": ("x", y0, (x0, x1)),
        "west": ("y", x0, (y0, y1)),
        "east": ("y", x1, (y0, y1)),
    }.items():
        lo, hi = lo_hi
        # endpoints on this border
        pts = sorted({p for s in segs
                      for p in ((s[0], s[1]), (s[2], s[3]))
                      if abs((p[1] if axis == "x" else p[0]) - val) < e
                      for p in [p]})
        coords = sorted((p[0] if axis == "x" else p[1]) for p in pts)
        # candidate gap spans between consecutive border endpoints
        cand = [lo] + coords + [hi]
        for a, b in zip(cand, cand[1:]):
            if b - a > 0.6 and not covered_along(val, a, b, axis):
                mid = (a + b) / 2.0
                out.append({"side": side, "index": round(mid - lo, 3)})
    return out


# ── Main ─────────────────────────────────────────────────────────────────

def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    summary = []
    for pdf in sorted(MAZE_DIR.glob("*.pdf")):
        segs = extract_segments(pdf)
        name = pdf.stem  # e.g. "5 by 5 orthogonal maze-3"
        try:
            defn, cell = decode_orthogonal(segs)
            defn["cell_pt"] = round(cell, 3)
            reached = orthogonal_cell_graph(defn)
            all_cells = {(c, r) for c in range(5) for r in range(5)}
            if reached != all_cells:
                raise ValueError(
                    f"unreachable cells from entrance: {sorted(all_cells - reached)}")
            out = OUT_DIR / f"{name}.json"
            out.write_text(json.dumps(defn, indent=1))
            summary.append((name, defn["type"],
                            f"in={defn['entrance']['side']}:{defn['entrance']['index']}",
                            f"out={defn['exit']['side']}:{defn['exit']['index']}"))
        except ValueError as exc:
            if "non-axis-aligned" in str(exc):
                defn = decode_angled(segs, name)
                out = OUT_DIR / f"{name}.json"
                out.write_text(json.dumps(defn, indent=1))
                summary.append((name, defn["type"],
                                f"gaps={[ (g['side'], g['index']) for g in defn['gaps'] ]}",
                                f"segs={len(defn['segments'])}"))
            else:
                raise
        print(f"{name}: {summary[-1][1:]}")
    print(f"\nwrote {len(summary)} maze definitions to {OUT_DIR}")


if __name__ == "__main__":
    main()
