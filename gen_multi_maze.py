#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
gen_multi_maze.py  —  Tạo world SDF với N mê cung song song
═══════════════════════════════════════════════════════════

Đọc file maze gốc (nhom8_maze.sdf hoặc nhom8_mecanum.sdf),
sao chép N lần theo hướng X với khoảng cách DX_STEP,
tạo file output chứa N mê cung giống hệt nhau.

Tại sao DX_STEP = 15m?
  • Lidar max range = 12m
  • Mê cung rộng 2.5m (x∈[2.0, 4.5])
  • Robot gần goal nhất: x=4.25m → reach = 4.25+12 = 16.25m
  • Maze kế tiếp bắt đầu tại x = 2.0 + 15 = 17.0m  > 16.25m ✓
  → Không có lidar cross-contamination giữa các maze

Cách dùng:
  python3 gen_multi_maze.py nhom8_maze.sdf --n 5 --out nhom8_multi5.sdf
  python3 gen_multi_maze.py nhom8_maze.sdf --n 4 --out nhom8_multi4.sdf

Sau đó:
  gz sim nhom8_multi5.sdf
  ./spawn_robot.sh 5   # spawn robot_1..robot_5
  python3 ga_nn_controller.py --pop 20 --gen 500 --workers 5
"""

import argparse
import re
import sys
import xml.etree.ElementTree as ET


DX_STEP = 15.0   # m between maze origins — safe for 12m lidar
# Base maze geometry (from nhom8_mecanum.sdf inspection):
#   walls:  x ∈ [2.0, 4.5],  y ∈ [-4.5, -2.0]
#   START:  (2.25, -3.25)
#   GOAL:   (4.25, -3.25)


def shift_pose_x(pose_str: str, dx: float) -> str:
    """
    SDF pose = "x y z roll pitch yaw"
    Add dx to the x component.
    """
    parts = pose_str.strip().split()
    if len(parts) < 1:
        return pose_str
    parts[0] = f"{float(parts[0]) + dx:.6f}"
    return " ".join(parts)


def duplicate_maze(sdf_text: str, n_copies: int) -> str:
    """
    Parse SDF, clone all maze models (everything except ground_plane/sun/lights)
    N times with increasing X offset.

    Returns modified SDF string.
    """
    root = ET.fromstring(sdf_text)
    world = root.find("world")
    if world is None:
        # SDF may not have <world> wrapper — wrap it
        world = root

    # Collect all models that are maze geometry (walls, floor, markers)
    # Exclude: ground_plane, sun (light), and any model already robot_*
    SKIP = {"ground_plane"}
    maze_models = []
    for model in world.findall("model"):
        name = model.get("name", "")
        if name not in SKIP and not name.startswith("robot_"):
            maze_models.append(model)

    if not maze_models:
        print("[WARN] No maze models found. Check SDF model names.")
        sys.exit(1)

    print(f"  Found {len(maze_models)} maze model(s) to duplicate.")

    # Copy N-1 additional mazes (maze 0 = original, already in file)
    for copy_idx in range(1, n_copies):
        dx = DX_STEP * copy_idx
        for orig_model in maze_models:
            # Deep-copy via XML text round-trip
            orig_str = ET.tostring(orig_model, encoding="unicode")
            new_model = ET.fromstring(orig_str)

            # Rename model: hw_0_0 → hw_0_0_m1 (maze index suffix)
            orig_name = new_model.get("name", "model")
            new_model.set("name", f"{orig_name}_m{copy_idx}")

            # Shift all <pose> elements within the model
            # Top-level model pose
            top_pose = new_model.find("pose")
            if top_pose is not None and top_pose.text:
                top_pose.text = shift_pose_x(top_pose.text, dx)
            else:
                # No explicit pose → add one at dx offset
                new_pose = ET.SubElement(new_model, "pose")
                new_pose.text = f"{dx:.6f} 0 0 0 0 0"

            # Also shift any link-level poses (some SDF files use this)
            for link in new_model.findall(".//link"):
                link_pose = link.find("pose")
                if link_pose is not None and link_pose.text:
                    # Link poses are relative to model — don't shift these
                    pass

            world.append(new_model)

    # Add visual START/GOAL markers for each maze copy (optional, helps debugging)
    for copy_idx in range(n_copies):
        dx = DX_STEP * copy_idx
        sx, sy = 2.25 + dx, -3.25
        gx, gy = 4.25 + dx, -3.25

        for tag, x, y, r, g, b in [
            (f"start_marker_m{copy_idx}", sx, sy, 0.0, 1.0, 0.0),
            (f"goal_marker_m{copy_idx}",  gx, gy, 1.0, 0.5, 0.0),
        ]:
            marker = ET.fromstring(f"""
            <model name="{tag}">
              <static>true</static>
              <pose>{x} {y} 0.01 0 0 0</pose>
              <link name="link">
                <visual name="vis">
                  <geometry><cylinder><radius>0.15</radius><length>0.02</length></cylinder></geometry>
                  <material>
                    <ambient>{r} {g} {b} 0.8</ambient>
                    <diffuse>{r} {g} {b} 0.8</diffuse>
                  </material>
                </visual>
              </link>
            </model>""")
            world.append(marker)

    # Serialize back to string
    ET.indent(root, space="  ")
    result = ET.tostring(root, encoding="unicode", xml_declaration=False)
    return '<?xml version="1.0" ?>\n' + result


def print_summary(n: int):
    """Print maze coordinates summary for ga_nn_controller.py config."""
    print(f"\n  Copy the following MAZE_CONFIGS into ga_nn_controller.py Cfg:")
    print(f"  MAZE_CONFIGS = [")
    for i in range(n):
        dx = DX_STEP * i
        sx = 2.25 + dx; gx = 4.25 + dx
        xmin = 1.85 + dx; xmax = 4.65 + dx
        print(f"    # Maze {i}: robot_{i+1}")
        print(f"    ({sx:.2f}, -3.25, {gx:.2f}, -3.25, {xmin:.2f}, {xmax:.2f}, -4.65, -1.85),")
    print(f"  ]")
    print(f"\n  Spawn robots:")
    print(f"    ./spawn_robot.sh {n}   # spawns robot_1..robot_{n}")
    print(f"\n  Train:")
    print(f"    python3 ga_nn_controller.py --pop 20 --gen 500 --workers {n}")


def main():
    ap = argparse.ArgumentParser(
        description="Generate N-maze SDF world for parallel GA training")
    ap.add_argument("maze_sdf",    help="Input maze SDF file (nhom8_maze.sdf)")
    ap.add_argument("--n",  type=int, default=5, help="Number of maze copies (default 5)")
    ap.add_argument("--out", type=str, default=None, help="Output SDF path")
    ap.add_argument("--dx", type=float, default=DX_STEP,
                    help=f"X spacing between mazes in meters (default {DX_STEP})")
    args = ap.parse_args()

    global DX_STEP
    DX_STEP = args.dx

    out = args.out or args.maze_sdf.replace(".sdf", f"_multi{args.n}.sdf")

    print(f"  Input:  {args.maze_sdf}")
    print(f"  Copies: {args.n}")
    print(f"  DX:     {DX_STEP}m  (lidar safe: {DX_STEP > 12:.0f})")
    print(f"  Output: {out}")

    with open(args.maze_sdf, "r") as f:
        sdf_text = f.read()

    result = duplicate_maze(sdf_text, args.n)

    with open(out, "w") as f:
        f.write(result)

    print(f"\n  ✓ Wrote {out}  ({len(result)//1024}KB)")
    print_summary(args.n)


if __name__ == "__main__":
    main()
