#!/bin/bash
# ============================================================
#  run_solver.sh — Khởi chạy maze_solver_node cho robot N
# ============================================================
#
#  CÚ PHÁP:
#    ./run_solver.sh [robot_id]
#
#  Ví dụ:
#    ./run_solver.sh        → giải mê cung với robot_1
#    ./run_solver.sh 2      → giải mê cung với robot_2
#
#  Luồng sử dụng đầy đủ:
#    Terminal 1: gz sim nhom8_mecanum_maze.sdf
#    Terminal 2: ./spawn_robot.sh 1 2.5,-2.5
#    Terminal 3: ./run_solver.sh 1
# ============================================================

RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'
CYAN='\033[0;36m'; BOLD='\033[1m'; NC='\033[0m'

ROBOT_ID="${1:-1}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SOLVER="$SCRIPT_DIR/maze_solver_node.py"

if [ ! -f "$SOLVER" ]; then
  echo -e "${RED}[ERROR]${NC} Không tìm thấy: $SOLVER"
  exit 1
fi

echo -e ""
echo -e "${BOLD}╔══════════════════════════════════════════╗${NC}"
echo -e "${BOLD}║   🤖  Maze Solver — robot_${ROBOT_ID}             ║${NC}"
echo -e "${BOLD}╚══════════════════════════════════════════╝${NC}"
echo -e ""
echo -e "  ${CYAN}Scan topic${NC}  : /scan${ROBOT_ID}"
echo -e "  ${CYAN}Cmd topic${NC}   : /model/robot_${ROBOT_ID}/cmd_vel"
echo -e "  ${CYAN}Odom topic${NC}  : /model/robot_${ROBOT_ID}/odometry"
echo -e "  ${CYAN}Goal${NC}        : (6.5, -6.5)"
echo -e ""
echo -e "  Nhấn ${RED}Ctrl+C${NC} để dừng."
echo -e ""

python3 "$SOLVER" --ros-args -p robot_id:="${ROBOT_ID}"
