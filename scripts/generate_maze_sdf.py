#!/usr/bin/env python3
"""
generate_maze_sdf.py — Generate the 5×5 maze world with 0.75 m cells.

Produces ``worlds/nhom8_maze75.sdf`` for the explore-then-exit task:
  * Same interior wall topology as the original 0.5 m maze
    (worlds/nhom8_maze.sdf), scaled to 0.75 m cells.
  * Border 100% sealed EXCEPT one exit gap: the east border at row 2
    (aligned with the START row) — this is the "way out of the maze" the
    robot must find after finishing its 2D map.
  * START cell (col 0, row 2); the robot spawns there every episode.

The module also exposes the maze geometry so tests can verify connectivity
and the env can align its coverage grid / exit detection with the world.

Run:  ./rl_venv/bin/python scripts/generate_maze_sdf.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

# ══════════════════════════════════════════════════════════════════════════
# Maze geometry (single source of truth for the 0.75 m maze)
# ══════════════════════════════════════════════════════════════════════════
ORIGIN_XY = (2.0, -4.5)     # SW corner of the maze outer rectangle
CELL = 0.75                 # cell size (m) — expanded from 0.50
WALL_T = 0.03               # wall thickness (m)
WALL_H = 0.40               # wall height (m)
PITCH = CELL + WALL_T       # 0.78 m between adjacent gridlines
GRID = 5                    # 5×5 cells
OUTER = GRID * PITCH + WALL_T   # 3.93 m outer size

# Exit gap: east border (x-gridline GRID), row EXIT_ROW is OPEN.
EXIT_ROW = 2

# Interior wall topology, decoded from worlds/nhom8_maze.sdf (0.5 m maze):
#   VWALLS: world-vertical walls → x-gridline i (0..GRID) → open rows j (0..GRID-1)
#   HWALLS: world-horizontal walls → y-gridline j (0..GRID) → open cols i (0..GRID-1)
VWALLS: dict[int, list[int]] = {
    0: [0, 1, 2, 3, 4],     # west border
    1: [1],
    2: [0, 2],
    3: [1, 2, 3],
    4: [2],
    5: [0, 1, 3, 4],        # east border — row EXIT_ROW left open (the exit)
}
HWALLS: dict[int, list[int]] = {
    0: [0, 1, 2, 3, 4],     # south border
    1: [3],
    2: [0, 2, 4],
    3: [1],
    4: [1, 2, 3, 4],
    5: [0, 1, 2, 3, 4],     # north border
}

START_CELL = (0, 2)         # (col, row) — same row as the exit gap


def gridline(axis: str, idx: int) -> float:
    """World coordinate of gridline ``idx`` along axis 'x' or 'y'."""
    base = ORIGIN_XY[0] if axis == "x" else ORIGIN_XY[1]
    return base + idx * PITCH


def cell_center(col: int, row: int) -> tuple[float, float]:
    """World center of maze cell (col, row)."""
    x = gridline("x", col) + WALL_T + CELL / 2.0
    y = gridline("y", row) + WALL_T + CELL / 2.0
    return x, y


def gap_y_range() -> tuple[float, float]:
    """World y-range of the exit gap on the east border.

    Walls span their FULL gridline pitch, so the gap is exactly one pitch
    wide: [gy[row], gy[row] + pitch].
    """
    y0 = gridline("y", EXIT_ROW)
    return y0, y0 + PITCH


# ══════════════════════════════════════════════════════════════════════════
# Topology verification (imported by the unit tests)
# ══════════════════════════════════════════════════════════════════════════

def has_vwall(i: int, j: int) -> bool:
    """World-vertical wall on x-gridline i spanning row j?"""
    return j in VWALLS.get(i, [])


def has_hwall(j: int, i: int) -> bool:
    """World-horizontal wall on y-gridline j spanning col i?"""
    return i in HWALLS.get(j, [])


def open_neighbours(col: int, row: int) -> list[tuple[int, int]]:
    """Maze cells reachable in one step from (col, row)."""
    out = []
    if not has_vwall(col, row):                 # west edge of the cell
        out.append((col - 1, row))
    if not has_vwall(col + 1, row):             # east edge of the cell
        out.append((col + 1, row))
    if not has_hwall(row, col):                 # south edge of the cell
        out.append((col, row - 1))
    if not has_hwall(row + 1, col):             # north edge of the cell
        out.append((col, row + 1))
    return [(c, r) for c, r in out if 0 <= c < GRID and 0 <= r < GRID]


def reachable_cells(start: tuple[int, int] = START_CELL) -> set[tuple[int, int]]:
    """BFS over the cell graph (used to prove every cell is visitable)."""
    seen = {start}
    frontier = [start]
    while frontier:
        c, r = frontier.pop()
        for nb in open_neighbours(c, r):
            if nb not in seen:
                seen.add(nb)
                frontier.append(nb)
    return seen


def verify_topology() -> None:
    """Assert the maze is fully connected and the exit is where we expect."""
    all_cells = {(c, r) for c in range(GRID) for r in range(GRID)}
    reached = reachable_cells()
    if reached != all_cells:
        missing = sorted(all_cells - reached)
        raise ValueError(f"maze not fully connected; unreachable cells: {missing}")
    if has_vwall(GRID, EXIT_ROW):
        raise ValueError(f"exit gap missing on east border at row {EXIT_ROW}")
    # Border sealed everywhere else
    for r in range(GRID):
        if r != EXIT_ROW and not has_vwall(GRID, r):
            raise ValueError(f"unexpected east-border hole at row {r}")
        if not has_vwall(0, r):
            raise ValueError(f"west border open at row {r}")
    for c in range(GRID):
        if not has_hwall(0, c) or not has_hwall(GRID, c):
            raise ValueError(f"south/north border open at col {c}")
    # Collinear segments on one gridline must be CONTIGUOUS: each segment
    # spans its full pitch [g(i), g(i)+pitch], so neighbours share endpoints.
    # (A previous version offset segments by the wall thickness and left
    # 3 cm seams at every junction — visibly open walls in Gazebo.)
    for i, rows in VWALLS.items():
        for j in rows:
            start = gridline("y", j)
            end = start + PITCH
            if j + 1 in rows and abs(end - gridline("y", j + 1)) > 1e-9:
                raise ValueError(f"wx_{i}_{j} not contiguous with wx_{i}_{j+1}")
    for j, cols in HWALLS.items():
        for i in cols:
            start = gridline("x", i)
            end = start + PITCH
            if i + 1 in cols and abs(end - gridline("x", i + 1)) > 1e-9:
                raise ValueError(f"wy_{j}_{i} not contiguous with wy_{j}_{i+1}")


# ══════════════════════════════════════════════════════════════════════════
# SDF emission
# ══════════════════════════════════════════════════════════════════════════

def _wall_model(name: str, cx: float, cy: float, sx: float, sy: float) -> str:
    return f"""    <model name="{name}">
      <static>true</static>
      <pose>{cx:.4f} {cy:.4f} {WALL_H / 2:.4f} 0 0 0</pose>
      <link name="link">
        <collision name="col">
          <geometry><box><size>{sx:.4f} {sy:.4f} {WALL_H:.4f}</size></box></geometry>
        </collision>
        <visual name="vis">
          <geometry><box><size>{sx:.4f} {sy:.4f} {WALL_H:.4f}</size></box></geometry>
          <material><ambient>0.15 0.45 0.85 1</ambient><diffuse>0.15 0.45 0.85 1</diffuse></material>
        </visual>
      </link>
    </model>
"""


def build_sdf_text() -> str:
    """Return the full world SDF text (walls axis-aligned, yaw 0).

    Every wall spans its FULL gridline pitch (cell + thickness), so collinear
    neighbours share endpoints and junctions are physically sealed — no seams.
    """
    parts: list[str] = []
    for i, rows in sorted(VWALLS.items()):
        for j in sorted(rows):
            parts.append(_wall_model(
                f"wx_{i}_{j}", gridline("x", i), gridline("y", j) + PITCH / 2,
                WALL_T, PITCH))
    for j, cols in sorted(HWALLS.items()):
        for i in sorted(cols):
            parts.append(_wall_model(
                f"wy_{j}_{i}", gridline("x", i) + PITCH / 2, gridline("y", j),
                PITCH, WALL_T))
    walls = "\n".join(parts)

    sx, sy = cell_center(*START_CELL)
    gy0, _gy1 = gap_y_range()
    return f"""<?xml version="1.0" ?>
<sdf version="1.10">
<world name="nhom8_maze75">
  <physics name="8ms" type="ignored">
    <max_step_size>0.008</max_step_size>
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
        <geometry><plane><normal>0 0 1</normal><size>50 50</size></plane></geometry>
        <surface><friction><ode><mu>40</mu></ode></friction></surface>
      </collision>
      <visual name="visual">
        <geometry><plane><normal>0 0 1</normal><size>50 50</size></plane></geometry>
        <material>
          <ambient>0.8 0.8 0.8 1</ambient><diffuse>0.8 0.8 0.8 1</diffuse>
          <specular>0.8 0.8 0.8 1</specular>
        </material>
      </visual>
    </link>
  </model>

  <!-- ═══════════════════════════════════════════════════════ -->
  <!--  MÊ CUNG 5×5  ô 0.75m×0.75m (mở rộng từ 0.50m)          -->
  <!--  Biên kín trừ CỬA RA: biên Đông, hàng {EXIT_ROW}        -->
  <!--  Start (col {START_CELL[0]}, row {START_CELL[1]}): world ({sx:.3f}, {sy:.3f})          -->
  <!--  Exit: x = {gridline('x', GRID):.2f}, y ∈ [{gy0:.3f}, {gy0 + CELL:.3f}]    -->
  <!--  wx_i_j: tường dọc (world-Y) trên lưới x = i            -->
  <!--  wy_j_i: tường ngang (world-X) trên lưới y = j          -->
  <!-- ═══════════════════════════════════════════════════════ -->

{walls}
    <model name="floor">
      <static>true</static>
      <pose>{ORIGIN_XY[0] + OUTER / 2:.4f} {ORIGIN_XY[1] + OUTER / 2:.4f} -0.02 0 0 0</pose>
      <link name="link">
        <collision name="col">
          <geometry><box><size>{OUTER:.2f} {OUTER:.2f} 0.03</size></box></geometry>
        </collision>
        <visual name="vis">
          <geometry><box><size>{OUTER:.2f} {OUTER:.2f} 0.03</size></box></geometry>
          <material><ambient>0.87 0.87 0.82 1</ambient><diffuse>0.87 0.87 0.82 1</diffuse></material>
        </visual>
      </link>
    </model>
</world>
</sdf>
"""


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate the 0.75 m maze world")
    ap.add_argument("--out", type=Path,
                    default=Path(__file__).resolve().parents[1] / "worlds" / "nhom8_maze75.sdf")
    args = ap.parse_args()

    verify_topology()
    n_walls = (sum(len(v) for v in VWALLS.values())
               + sum(len(v) for v in HWALLS.values()))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(build_sdf_text())

    sx, sy = cell_center(*START_CELL)
    print(f"wrote {args.out}")
    print(f"  cells     : {GRID}×{GRID} × {CELL} m (outer {OUTER:.2f} m)")
    print(f"  walls     : {n_walls} (border sealed except east row {EXIT_ROW} = EXIT)")
    print(f"  start     : cell {START_CELL} → ({sx:.3f}, {sy:.3f})")
    print(f"  exit gap  : x = {gridline('x', GRID):.2f}, y ∈ {gap_y_range()}")


if __name__ == "__main__":
    main()
