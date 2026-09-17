"""Bounded SE(2) geometry audit for narrow mazes; NOT neural/Gazebo evidence.

Searches unicycle motion primitives with the rectangular robot footprint and
10 mm margin. All intermediate poses are checked against exact wall boxes.
A failed bounded search does not prove a maze physically unsolvable.
"""
import argparse
import heapq
import itertools
import json
import math
from pathlib import Path
import time

import numpy as np

from rl_training import config as C
from rl_training.eval_hybrid import maze_setup
from rl_training.hybrid_navigation import integrate, wrap
from rl_training.maze_registry import load_registry


def safe_pose(geom, pose, margin=0.01):
    dx, dy = (geom.center - pose[:2]).T
    c, s = np.abs(np.cos(geom.angle - pose[2])), np.abs(np.sin(geom.angle - pose[2]))
    cr, sr = math.cos(pose[2]), math.sin(pose[2])
    rx, ry = C.CHASSIS_L / 2 + margin, C.WHEEL_SEP / 2 + 0.01 + margin
    wx, wy = geom.half.T
    return not np.any((np.abs(cr * dx + sr * dy) <= rx + wx * c + wy * s)
                      & (np.abs(-sr * dx + cr * dy) <= ry + wx * s + wy * c)
                      & (np.abs(geom.c * dx + geom.s * dy) <= wx + rx * c + ry * s)
                      & (np.abs(-geom.s * dx + geom.c * dy) <= wy + rx * s + ry * c))


def distance_field(planner, goal):
    # Lower-bound heuristic: orientation ignored, only robot half-width used.
    allowed = planner.clearance > C.WHEEL_SEP / 2 + 0.01
    distance = np.full(allowed.shape, np.inf)
    gx, gy = planner.cell(goal)
    distance[gy, gx] = 0
    queue = [(0., gx, gy)]
    while queue:
        d, x, y = heapq.heappop(queue)
        if d > distance[y, x]:
            continue
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (1, -1), (-1, 1), (-1, -1)):
            nx, ny = x + dx, y + dy
            if not (0 <= ny < allowed.shape[0] and 0 <= nx < allowed.shape[1]) or not allowed[ny, nx]:
                continue
            nd = d + math.hypot(dx, dy) * planner.res
            if nd < distance[ny, nx]:
                distance[ny, nx] = nd
                heapq.heappush(queue, (nd, nx, ny))
    return distance


def search(spec, seconds=60., max_nodes=50000, start_pose=None, margin=.01):
    planner, geometry, goal = maze_setup(spec)
    field = distance_field(planner, goal)
    start = np.array([*spec.start_xy_local, spec.start_yaw] if start_pose is None else start_pose)

    def key(pose):
        return (*np.floor((pose[:2] - planner.origin) / 0.04).astype(int),
                int(round(float(wrap(pose[2])) / (math.pi / 8))) % 16)

    def heuristic(pose):
        x, y = planner.cell(pose[:2])
        if not (0 <= y < field.shape[0] and 0 <= x < field.shape[1]):
            return math.inf
        return float(field[y, x])

    primitives = [(v, w, 0.75) for v in (0.12, -0.12) for w in (0., math.pi / 6, -math.pi / 6)]
    primitives += [(0., math.pi / 8, 1.), (0., -math.pi / 8, 1.)]
    serial = itertools.count()
    # Immutable tree nodes: pose, parent index, primitive (v,w,dt).
    nodes = [(start, None, None)]
    best = {key(start): 0.}
    queue = [(heuristic(start), next(serial), 0., 0)]
    began, expanded = time.monotonic(), 0
    final = None
    while queue and expanded < max_nodes and time.monotonic() - began < seconds:
        _, _, cost, index = heapq.heappop(queue)
        pose = nodes[index][0]
        if cost > best.get(key(pose), math.inf) + 1e-9:
            continue
        expanded += 1
        if math.dist(pose[:2], goal) < 0.06:
            final = index
            break
        for v, w, dt in primitives:
            candidate = integrate(pose, v, w, dt)
            h = heuristic(candidate)
            if not math.isfinite(h):
                continue
            nk = key(candidate)
            nc = cost + (abs(v) * dt * (1.4 if v < 0 else 1.) + abs(w) * dt * .09)
            if nc >= best.get(nk, math.inf):
                continue
            # Enlarge each sampled rectangle by the maximum displacement
            # of any footprint point to the nearest sample in time.
            radius = math.hypot(C.CHASSIS_L / 2 + margin, C.WHEEL_SEP / 2 + .01 + margin)
            interpolation_pad = (abs(v) + radius * abs(w)) * dt / 20
            if not all(safe_pose(geometry, integrate(pose, v, w, t), margin + interpolation_pad)
                       for t in np.linspace(0., dt, 11)):
                continue
            best[nk] = nc
            nodes.append((candidate, index, (v, w, dt)))
            heapq.heappush(queue, (nc + 1.5 * h, next(serial), nc, len(nodes) - 1))
    path, commands = [], []
    while final is not None:
        pose, parent, primitive = nodes[final]
        path.append(pose.tolist())
        if primitive:
            commands.append(primitive)
        final = parent
    return {"maze": spec.name, "path_found": bool(path), "expanded": expanded,
            "seconds": time.monotonic() - began, "margin_m": margin,
            "path": path[::-1], "commands": commands[::-1],
            "scope": "known-map SE2 geometry only; not a learned policy or Gazebo solve"}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seconds", type=float, default=60.)
    p.add_argument("--mazes", nargs="*", default=[f"sigma_{i}" for i in range(1, 6)])
    p.add_argument("--output", default="hybrid_runs/sigma_se2_audit")
    args = p.parse_args()
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    registry = load_registry()
    for name in args.mazes:
        result = search(registry[name], args.seconds)
        (out / f"{name}.json").write_text(json.dumps(result, indent=2) + "\n")
        print({k: v for k, v in result.items() if k not in ("path", "commands")}, flush=True)


if __name__ == "__main__":
    main()
