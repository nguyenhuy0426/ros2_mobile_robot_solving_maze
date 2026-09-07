#!/usr/bin/env python3
"""Live "which robot is in which maze" view for the multi-robot explore stack.

WHY THIS EXISTS. The combined world holds 13 mazes but the watchdog spawns
MAZE_FLEET robots (default 4), so 9 slots are empty on screen at all times.
The 4 that exist still train on all 13 mazes -- they TELEPORT into the next
maze of their own group at every reset -- so the robot/maze pairing changes
every episode and is not visible from the GUI. The trainer prints
"[robot 3 | ortho_5]" only when an episode ENDS, which is exactly too late
to look up at the screen and find the robot.

This reads each robot's live pose off ROS and resolves it against the maze
footprints, so the pairing can be read at any moment, mid-episode.

Usage:
    rl_venv/bin/python -m rl_training.fleet_status            # one snapshot
    rl_venv/bin/python -m rl_training.fleet_status --watch 2  # refresh every 2s
"""
from __future__ import annotations

import argparse
import time
from typing import Dict, Optional, Tuple

import rclpy
from geometry_msgs.msg import Pose
from rclpy.node import Node

import rl_training.maze_registry as MR

# A pose a few cm outside a wall is an out_of_bounds death in progress --
# the single most informative moment to be able to name the maze -- so the
# lookup is inflated rather than going blank right when it matters.
LOCATE_MARGIN = 0.15


class FleetWatcher(Node):
    """Subscribes to every robot pose the world is currently publishing."""

    def __init__(self, robot_ids) -> None:
        super().__init__("fleet_status")
        self._pose: Dict[int, Optional[Tuple[float, float]]] = {
            i: None for i in robot_ids}
        qos = rclpy.qos.QoSProfile(
            reliability=rclpy.qos.ReliabilityPolicy.BEST_EFFORT,
            history=rclpy.qos.HistoryPolicy.KEEP_LAST, depth=1)
        for i in robot_ids:
            self.create_subscription(
                Pose, f"/model/robot_{i}/pose",
                lambda msg, k=i: self._pose.__setitem__(
                    k, (msg.position.x, msg.position.y)),
                qos)

    def poses(self) -> Dict[int, Optional[Tuple[float, float]]]:
        return dict(self._pose)


def discover_robot_ids(node: Node) -> list:
    """Robot numbers actually present, read from the live pose topics.

    Discovering rather than assuming 1..N: the fleet size is set by
    MAZE_FLEET in the watchdog, and a robot whose spawn failed leaves a gap
    that is itself worth seeing.
    """
    ids = []
    for name, _types in node.get_topic_names_and_types():
        parts = name.split("/")
        if len(parts) == 4 and parts[1] == "model" and parts[3] == "pose":
            model = parts[2]
            if model.startswith("robot_") and model[6:].isdigit():
                ids.append(int(model[6:]))
    return sorted(set(ids))


def render(poses, mazes) -> str:
    """One table: occupied slots first, then the mazes standing empty."""
    located: Dict[str, list] = {}
    lines = ["robot | maze      | pose"]
    lines.append("------+-----------+---------------------")
    for rid in sorted(poses):
        xy = poses[rid]
        if xy is None:
            lines.append(f"{rid:5d} | {'(no pose)':9s} | -")
            continue
        maze = MR.maze_at_world(xy, margin=LOCATE_MARGIN)
        located.setdefault(maze or "(outside)", []).append(rid)
        lines.append(f"{rid:5d} | {maze or '(outside)':9s} | "
                     f"x={xy[0]:7.2f} y={xy[1]:7.2f}")
    empty = [m for m in mazes if m not in located]
    lines.append("")
    lines.append(f"occupied {len(mazes) - len(empty)}/{len(mazes)} mazes; "
                 f"empty: {', '.join(empty) if empty else 'none'}")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--watch", type=float, default=0.0,
                    help="refresh every N seconds (0 = single snapshot)")
    ap.add_argument("--settle", type=float, default=2.0,
                    help="seconds to collect poses before the first print")
    args = ap.parse_args()

    mazes = MR.sorted_registry_names()
    rclpy.init()
    probe = Node("fleet_status_probe")
    # Discovery is not instantaneous; spin briefly so the topic graph fills.
    deadline = time.monotonic() + args.settle
    ids: list = []
    while time.monotonic() < deadline and not ids:
        rclpy.spin_once(probe, timeout_sec=0.2)
        ids = discover_robot_ids(probe)
    probe.destroy_node()

    if not ids:
        print("no /model/robot_*/pose topics — is the simulator running?")
        rclpy.shutdown()
        return

    watcher = FleetWatcher(ids)
    try:
        while True:
            deadline = time.monotonic() + args.settle
            while time.monotonic() < deadline:
                rclpy.spin_once(watcher, timeout_sec=0.1)
            print(render(watcher.poses(), mazes), flush=True)
            if args.watch <= 0:
                break
            print("", flush=True)
            time.sleep(args.watch)
    except KeyboardInterrupt:
        pass
    finally:
        watcher.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
