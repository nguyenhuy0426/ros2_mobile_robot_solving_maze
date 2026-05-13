#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
gen_multi_maze.py  —  Tạo world SDF với N mê cung xếp theo trục Y
══════════════════════════════════════════════════════════════════

Các mê cung xếp HÀNG NGANG (theo trục Y âm), cách nhau 1m.
X và GOAL không thay đổi — chỉ Y shift từng maze.

Tại sao xếp theo Y (không phải X)?
  → Compact hơn, dễ quan sát trong Gazebo
  → X không đổi → tất cả robot cùng hướng START_YAW=0 (→+x)

Tại sao gap chỉ 1m là đủ an toàn?
  Robot nằm BÊN TRONG maze. Tường ngoài (dày 0.03m) chặn lidar:
  ┌──────────────┐  ← tường ngoài maze 0 (y = -4.5)
  │    maze 0    │     robot bên trong, lidar bị tường chặn
  └──────────────┘  ← tường ngoài maze 0 (y = -2.0)
      1.0m gap
  ┌──────────────┐  ← tường ngoài maze 1 (y = -5.5 sau shift)
  │    maze 1    │
  └──────────────┘
  → Lidar không xuyên qua tường → không cross-contamination

Layout kết quả (N=5, DY=3.5m):
  Maze 0: y∈[-4.5,-2.0]   robot_1  START=(2.25,-3.25)
  Maze 1: y∈[-8.0,-5.5]   robot_2  START=(2.25,-6.75)
  Maze 2: y∈[-11.5,-9.0]  robot_3  START=(2.25,-10.25)
  Maze 3: y∈[-15.0,-12.5] robot_4  START=(2.25,-13.75)
  Maze 4: y∈[-18.5,-16.0] robot_5  START=(2.25,-17.25)

Cách dùng:
  python3 gen_multi_maze.py nhom8_maze.sdf --n 5
  python3 gen_multi_maze.py nhom8_maze.sdf --n 5 --gap 1.5
  python3 gen_multi_maze.py nhom8_maze.sdf --n 5 --out nhom8_multi5.sdf
"""

import argparse
import sys
import xml.etree.ElementTree as ET

# ── Maze geometry constants (từ nhom8_mecanum.sdf) ──────────────
MAZE_Y_SPAN = 2.5   # m: wall-to-wall height (y từ -4.5 đến -2.0)
GAP_DEFAULT = 1.0   # m: khoảng cách tường ngoài giữa 2 maze kề nhau
# DY_STEP = MAZE_Y_SPAN + GAP — tính khi biết GAP

# Tọa độ maze gốc
BASE_START_X = 2.25
BASE_GOAL_X  = 4.25
BASE_START_Y = -3.25   # = center of maze 0
BASE_Y_WALL_BOTTOM = -4.5   # tường dưới maze 0 (y âm nhất)
BASE_Y_WALL_TOP    = -2.0   # tường trên maze 0


def shift_pose_y(pose_str: str, dy: float) -> str:
    """
    SDF pose = "x y z roll pitch yaw"
    Cộng dy vào component Y (index 1).
    """
    parts = pose_str.strip().split()
    if len(parts) < 2:
        return pose_str
    parts[1] = f"{float(parts[1]) + dy:.6f}"
    return " ".join(parts)


def duplicate_maze_y(sdf_text: str, n_copies: int, gap: float) -> str:
    """
    Clone tất cả wall models N lần, mỗi lần shift Y thêm DY_STEP.
    maze 0 = original (không thay đổi).

    Args:
        sdf_text: nội dung SDF gốc
        n_copies: số maze (bao gồm maze 0)
        gap: khoảng cách tường ngoài giữa các maze (m)
    """
    dy_step = (MAZE_Y_SPAN + gap)   # âm vì y đi xuống

    root  = ET.fromstring(sdf_text)
    world = root.find("world") or root

    # Thu thập tất cả wall models (bỏ ground_plane, sun, robot_*)
    SKIP = {"ground_plane"}
    maze_models = [
        m for m in world.findall("model")
        if m.get("name","") not in SKIP
        and not m.get("name","").startswith("robot_")
    ]

    if not maze_models:
        print("[ERROR] Không tìm thấy wall models trong SDF.")
        sys.exit(1)
    print(f"  Tìm thấy {len(maze_models)} wall model(s) để nhân bản.")

    # Tạo N-1 bản copy (maze 0 đã có sẵn)
    for copy_i in range(1, n_copies):
        dy = dy_step * copy_i
        for orig in maze_models:
            xml_str  = ET.tostring(orig, encoding="unicode")
            new_m    = ET.fromstring(xml_str)

            # Đổi tên: hw_0_0 → hw_0_0_m1
            orig_name = new_m.get("name", "model")
            new_m.set("name", f"{orig_name}_m{copy_i}")

            # Shift pose của model (tuyệt đối trong world)
            top_pose = new_m.find("pose")
            if top_pose is not None and top_pose.text:
                top_pose.text = shift_pose_y(top_pose.text, dy)
            else:
                new_pose      = ET.SubElement(new_m, "pose")
                new_pose.text = f"0 {dy:.6f} 0 0 0 0"

            world.append(new_m)

    # # Thêm markers START (xanh lá) và GOAL (cam) cho mỗi maze
    # for i in range(n_copies):
    #     dy = dy_step * i
    #     sx, sy = BASE_START_X, BASE_START_Y + dy
    #     gx, gy = BASE_GOAL_X,  BASE_START_Y + dy

    #     for tag, x, y, r, g, b in [
    #         (f"start_m{i}", sx, sy, 0.0, 1.0, 0.0),
    #         (f"goal_m{i}",  gx, gy, 1.0, 0.5, 0.0),
    #     ]:
    #         world.append(ET.fromstring(
    #             f'<model name="{tag}">'
    #             f'<static>true</static>'
    #             f'<pose>{x} {y} 0.01 0 0 0</pose>'
    #             f'<link name="link">'
    #             f'<visual name="vis">'
    #             f'<geometry><cylinder><radius>0.12</radius>'
    #             f'<length>0.02</length></cylinder></geometry>'
    #             f'<material>'
    #             f'<ambient>{r} {g} {b} 0.9</ambient>'
    #             f'<diffuse>{r} {g} {b} 0.9</diffuse>'
    #             f'</material></visual></link></model>'
    #         ))

    ET.indent(root, space="  ")
    body = ET.tostring(root, encoding="unicode", xml_declaration=False)
    return '<?xml version="1.0" ?>\n' + body


def print_summary(n: int, gap: float):
    """In tọa độ tất cả mazes để copy vào code."""
    dy_step = -(MAZE_Y_SPAN + gap)
    print(f"\n  ╔══════════════════════════════════════════════════════╗")
    print(f"  ║  Layout: {n} mazes, DY={abs(dy_step):.1f}m/maze ({MAZE_Y_SPAN}m + {gap}m gap)    ║")
    print(f"  ╠══════════════════════════════════════════════════════╣")
    for i in range(n):
        dy   = dy_step * i
        sy   = BASE_START_Y + dy
        ymin = BASE_Y_WALL_BOTTOM + dy
        ymax = BASE_Y_WALL_TOP    + dy
        print(f"  ║  Maze {i} robot_{i+1}: y∈[{ymin:.2f},{ymax:.2f}]  "
              f"START=(2.25,{sy:.2f})  GOAL=(4.25,{sy:.2f})")
    print(f"  ╠══════════════════════════════════════════════════════╣")
    print(f"  ║  Spawn: ./spawn_robot.sh {n}")
    print(f"  ║  Train: python3 ga_nn_controller.py --workers {n}")
    print(f"  ╚══════════════════════════════════════════════════════╝")


def main():
    ap = argparse.ArgumentParser(
        description="Tạo SDF với N mê cung xếp theo trục Y")
    ap.add_argument("maze_sdf", help="File SDF mê cung gốc (nhom8_maze.sdf)")
    ap.add_argument("--n",   type=int,   default=5,
                    help="Số bản sao (default 5)")
    ap.add_argument("--gap", type=float, default=GAP_DEFAULT,
                    help=f"Khoảng cách giữa tường ngoài các maze (default {GAP_DEFAULT}m)")
    ap.add_argument("--out", type=str,   default=None,
                    help="File output SDF (default: <input>_multiN.sdf)")
    args = ap.parse_args()

    out = args.out or args.maze_sdf.replace(".sdf", f"_multi{args.n}.sdf")
    dy  = (MAZE_Y_SPAN + args.gap)

    print(f"\n  Input : {args.maze_sdf}")
    print(f"  Copies: {args.n}")
    print(f"  Gap   : {args.gap}m  →  DY = {abs(dy):.2f}m per maze")
    print(f"  Output: {out}")

    with open(args.maze_sdf) as f:
        sdf_text = f.read()

    result = duplicate_maze_y(sdf_text, args.n, args.gap)

    with open(out, "w") as f:
        f.write(result)

    sz = len(result) // 1024
    print(f"\n  ✓ Wrote {out}  ({sz}KB)")
    print_summary(args.n, args.gap)


if __name__ == "__main__":
    main()