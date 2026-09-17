"""Known-map, exact-pose 2D diagnostic on the 13 repository mazes.

This is NOT SLAM, Gazebo, or hardware evidence. Neural weights train only on
synthetic local observations. Maze walls are used by the planner and simulator,
never passed to the neural network. Collision checking uses continuous small
substeps and an independent enclosing rectangle, not the planner's raster.
"""
import argparse
from dataclasses import asdict
import json
import math
from pathlib import Path

import numpy as np
import torch

from rl_training import config as C
from rl_training.hybrid_navigation import H, GridPlanner, LocalPolicy, filter_command, integrate, observation
from rl_training.maze_registry import load_registry


class Geometry:
    def __init__(self, walls):
        self.center = np.array([[w.cx, w.cy] for w in walls])
        self.half = np.array([[w.half_length, w.half_thickness] for w in walls])
        self.angle = np.array([w.yaw for w in walls])
        self.c, self.s = np.cos(self.angle), np.sin(self.angle)

    def scan(self, pose):
        angles = pose[2] + np.arange(C.N_RAYS) * 2 * np.pi / C.N_RAYS
        sensor = pose[:2] + C.LIDAR_X_OFF * np.array([math.cos(pose[2]), math.sin(pose[2])])
        delta = sensor - self.center
        origins = np.column_stack((self.c * delta[:, 0] + self.s * delta[:, 1],
                                   -self.s * delta[:, 0] + self.c * delta[:, 1]))
        directions = np.stack((np.cos(angles[:, None] - self.angle),
                               np.sin(angles[:, None] - self.angle)), axis=-1)
        parallel = np.abs(directions) < 1e-10
        safe_d = np.where(parallel, 1., directions)
        t1 = (-self.half - origins) / safe_d
        t2 = (self.half - origins) / safe_d
        low, high = np.minimum(t1, t2), np.maximum(t1, t2)
        outside = np.abs(origins) > self.half
        low = np.where(parallel, np.where(outside, np.inf, -np.inf), low)
        high = np.where(parallel, np.where(outside, -np.inf, np.inf), high)
        enter, leave = low.max(axis=-1), high.min(axis=-1)
        hit = np.where(leave >= np.maximum(enter, 0), np.maximum(enter, 0), np.inf)
        return np.clip(hit.min(axis=1), C.LIDAR_MIN, C.LIDAR_MAX).astype(np.float32)

    def collides(self, pose):
        # Separating-axis test: robot rectangle includes the wheel extents.
        dx, dy = (self.center - pose[:2]).T
        cr, sr = math.cos(pose[2]), math.sin(pose[2])
        c, s = np.abs(np.cos(self.angle - pose[2])), np.abs(np.sin(self.angle - pose[2]))
        rx, ry = C.CHASSIS_L / 2, C.WHEEL_SEP / 2 + 0.01
        wx, wy = self.half.T
        overlap = ((np.abs(cr * dx + sr * dy) <= rx + wx * c + wy * s)
                   & (np.abs(-sr * dx + cr * dy) <= ry + wx * s + wy * c)
                   & (np.abs(self.c * dx + self.s * dy) <= wx + rx * c + ry * s)
                   & (np.abs(-self.s * dx + self.c * dy) <= wy + rx * s + ry * c))
        return bool(overlap.any())


def maze_setup(spec):
    # A supplied, known map is an intentional upper-bound diagnostic.
    origin = np.asarray(spec.bounds[:2]) - 0.4
    size = np.asarray(spec.bounds[2:]) - origin + 0.4
    res = 0.025
    nx, ny = np.ceil(size / res).astype(int)
    xs, ys = np.meshgrid(origin[0] + (np.arange(nx) + 0.5) * res,
                         origin[1] + (np.arange(ny) + 0.5) * res)
    occupied = np.zeros((ny, nx), dtype=bool)
    for wall in spec.walls_local:
        occupied |= wall.contains_array(xs, ys)
    planner = GridPlanner(occupied.astype(np.int8) * 100, res, origin)
    goal = np.array(spec.exit_opening.xy_local)
    border = spec.exit_opening.border
    goal += {"south": [0, -0.25], "north": [0, 0.25], "west": [-0.25, 0], "east": [0.25, 0]}[border]
    return planner, Geometry(spec.walls_local), goal


def run_episode(spec, model, trial, max_steps):
    planner, geom, goal = maze_setup(spec)
    rng = np.random.default_rng(10000 + trial)
    pose = np.array([*spec.start_xy_local, spec.start_yaw + rng.uniform(-0.15, 0.15)])
    path = planner.plan(pose[:2], goal)
    record = {"maze": spec.name, "trial": trial, "success": False,
              "collision": geom.collides(pose), "steps": 0, "overrides": 0,
              "path_length": 0., "reason": "no_safe_path" if not path else "timeout"}
    trajectory = [pose.tolist()]
    if not path or record["collision"]:
        return record, trajectory
    for step in range(max_steps):
        scan = geom.scan(pose)
        subgoal = planner.subgoal(pose[:2], path)
        v, w = model.command(observation(scan, pose, subgoal))
        v, w, override = filter_command(planner, pose, scan, v, w)
        record["overrides"] += int(override)
        prev = pose.copy()
        for _ in range(8):  # swept collision test every 25 ms at dt=0.2
            pose = integrate(pose, v, w, 0.025)
            if geom.collides(pose):
                record["collision"], record["reason"] = True, "collision"
                break
        record["steps"] = step + 1
        record["path_length"] += math.dist(prev[:2], pose[:2])
        trajectory.append(pose.tolist())
        if record["collision"]:
            break
        if math.dist(pose[:2], goal) < H.goal_tolerance:
            record["success"], record["reason"] = True, "goal"
            break
        if step > 150 and math.dist(trajectory[-1][:2], trajectory[-150][:2]) < 0.01:
            record["reason"] = "stalled"
            break
    return record, trajectory


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default="hybrid_runs/release_latest/local_policy.pt")
    p.add_argument("--output", default="hybrid_runs/evaluations/simulated.json")
    p.add_argument("--trials", type=int, default=3)
    p.add_argument("--max-steps", type=int, default=2500)
    p.add_argument("--mazes", nargs="*")
    args = p.parse_args()
    torch.set_num_threads(1)
    model = LocalPolicy()
    model.load_state_dict(torch.load(args.model, map_location="cpu", weights_only=True))
    model.eval()
    records, trajectories = [], {}
    for name, spec in sorted(load_registry().items()):
        if args.mazes and name not in args.mazes:
            continue
        for trial in range(args.trials):
            result, trajectory = run_episode(spec, model, trial, args.max_steps)
            records.append(result)
            trajectories[f"{name}_{trial}"] = trajectory
            print(json.dumps(result), flush=True)
    payload = {"scope": "known-map exact-pose 2D diagnostic; NOT Gazebo/SLAM/hardware",
               "config": asdict(H), "episodes": len(records),
               "successes": sum(r["success"] for r in records),
               "collisions": sum(r["collision"] for r in records), "results": records}
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2) + "\n")
    out.with_suffix(".trajectories.json").write_text(json.dumps(trajectories) + "\n")
    print(f"TOTAL {payload['successes']}/{len(records)} successes, {payload['collisions']} collisions")


if __name__ == "__main__":
    main()
