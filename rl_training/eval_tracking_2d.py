"""Independent dense-collision diagnostic for the neural SE(2) tracker."""
import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch

from rl_training.eval_hybrid import maze_setup
from rl_training.hybrid_navigation import integrate
from rl_training.maze_registry import load_registry
from rl_training.oriented_navigation import OrientedRoute, TrackingPolicy, tracking_observation, filter_rectangular


def episode(spec, model, seed):
    _, geometry, goal = maze_setup(spec)
    pose = np.array([*spec.start_xy_local, spec.start_yaw + np.random.default_rng(seed).uniform(-.5, .5)])
    row = {"maze": spec.name, "seed": seed, "success": False, "collision": False,
           "steps": 0, "overrides": 0, "reason": "timeout"}
    trace = [pose.tolist()]
    try:
        route = OrientedRoute(spec, pose)
    except RuntimeError as exc:
        row["reason"] = str(exc)
        return row, trace
    stopped = 0
    for step in range(1500):
        scan = geometry.scan(pose)
        reference, command = route.reference(pose)
        v, w = model.command(tracking_observation(scan, pose, reference, command))
        v, w, override = filter_rectangular(geometry, pose, scan, v, w)
        row["overrides"] += int(override)
        for _ in range(10):
            pose = integrate(pose, v, w, .02)
            if geometry.collides(pose):
                row.update(collision=True, reason="collision")
                break
        row["steps"] = step + 1
        trace.append(pose.tolist())
        if row["collision"]:
            break
        if math.dist(pose[:2], goal) < .07:
            row.update(success=True, reason="goal")
            break
        stopped = stopped + 1 if v == 0 and w == 0 else 0
        if stopped > 25:
            row["reason"] = "safety_stop"
            break
    return row, trace


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default="hybrid_runs/release_latest/tracking_policy.pt")
    p.add_argument("--output", default="hybrid_runs/evaluations/tracking_2d.json")
    p.add_argument("--trials", type=int, default=3)
    args = p.parse_args()
    torch.set_num_threads(1)
    model = TrackingPolicy()
    model.load_state_dict(torch.load(args.model, weights_only=True, map_location="cpu"))
    rows, traces = [], {}
    for name, spec in load_registry().items():
        if not name.startswith("sigma"):
            continue
        for trial in range(args.trials):
            row, trace = episode(spec, model, 9100 + trial)
            rows.append(row)
            traces[f"{name}_{trial}"] = trace
            print(row, flush=True)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({"scope": "2D exact-pose diagnostic, not Gazebo or SLAM",
                                 "results": rows}, indent=2))
    output.with_suffix(".trajectories.json").write_text(json.dumps(traces))


if __name__ == "__main__":
    main()
