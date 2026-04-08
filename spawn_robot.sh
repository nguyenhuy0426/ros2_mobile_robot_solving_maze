#!/bin/bash
# spawn_robot.sh — Spawn N mecanum robots (robot_1 .. robot_N) vào world nhom8_mecanum
#                  + N ros_gz_bridge  +  N static_transform_publisher
#
# Cách dùng:
#   ./spawn_robot.sh <N>            # spawn robot_1 .. robot_N tại START mê cung
#   ./spawn_robot.sh <N> [x] [y] [z]  # override vị trí spawn (tất cả robot)
#
# Ví dụ:
#   ./spawn_robot.sh 1              # 1 robot
#   ./spawn_robot.sh 5              # 5 robots song song, không va chạm nhau
#   ./spawn_robot.sh 20             # 20 robots (GA population = 20)
#
# Ctrl+C → kill TẤT CẢ bridge + TF của N robot cùng lúc.
#
# ─────────────────────────────────────────────────────────────────────────────
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'
CYAN='\033[0;36m'; BOLD='\033[1m'; NC='\033[0m'

# ── Kiểm tra tham số ─────────────────────────────────────────────────────────
N=${1:?"$(echo -e "${RED}[ERROR]${NC} Thiếu tham số.\nCách dùng: ./spawn_robot.sh <N> [x] [y] [z]")"}
if ! [[ "$N" =~ ^[1-9][0-9]*$ ]]; then
    echo -e "${RED}[ERROR]${NC} N phải là số nguyên dương, nhận được: '$N'"
    exit 1
fi

# ── Tọa độ START mê cung nhom8_mecanum ───────────────────────────────────────
# Start cell (5,3) → world (2.25, -3.25, 0.024)
# Goal  cell (1,3) → world (4.25, -3.25)
DEFAULT_X=2.25
DEFAULT_Y=-3.25
DEFAULT_Z=0.024

X=${2:-$DEFAULT_X}
Y=${3:-$DEFAULT_Y}
Z=${4:-$DEFAULT_Z}

WORLD="nhom8_mecanum"

# ── Kiểm tra template SDF ─────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TEMPLATE="$SCRIPT_DIR/one_robot.sdf"

if [ ! -f "$TEMPLATE" ]; then
    echo -e "${RED}[ERROR]${NC} Không tìm thấy template SDF: $TEMPLATE"
    exit 1
fi

# ── Mảng lưu PID ─────────────────────────────────────────────────────────────
declare -a BRIDGE_PIDS=()
declare -a TF_PIDS=()

# ── Hàm cleanup khi Ctrl+C ────────────────────────────────────────────────────
cleanup() {
    echo -e "\n\n${YELLOW}[STOP]${NC} Đang dừng tất cả bridge và TF publishers..."
    local all_pids=("${BRIDGE_PIDS[@]}" "${TF_PIDS[@]}")
    if [ ${#all_pids[@]} -gt 0 ]; then
        kill "${all_pids[@]}" 2>/dev/null
        sleep 0.5
        # Force kill nếu vẫn còn
        for pid in "${all_pids[@]}"; do
            kill -9 "$pid" 2>/dev/null
        done
    fi
    echo -e "${GREEN}[STOP]${NC} Đã dừng ${N} robot(s). Bye!"
    exit 0
}
trap cleanup INT TERM

# ═════════════════════════════════════════════════════════════════════════════
# BƯỚC 1 — Spawn tất cả N robots song song
# ═════════════════════════════════════════════════════════════════════════════
echo ""
echo -e "  ${CYAN}${BOLD}╔══════════════════════════════════════════════════════╗${NC}"
echo -e "  ${CYAN}${BOLD}║  Spawning ${YELLOW}${N}${CYAN} robots vào world ${YELLOW}${WORLD}${CYAN}           ║${NC}"
echo -e "  ${CYAN}${BOLD}║  START: x=${X}  y=${Y}  z=${Z}              ║${NC}"
echo -e "  ${CYAN}${BOLD}║  GOAL : x=2.25  y=-3.25  (cell 1,3)          ║${NC}"
echo -e "  ${CYAN}${BOLD}╚══════════════════════════════════════════════════════╝${NC}"
echo ""
echo -e "${GREEN}[1/3]${NC} Tạo SDF và spawn ${YELLOW}${N}${NC} robot(s) song song..."

declare -a SPAWN_PIDS=()
for i in $(seq 1 "$N"); do
    TMP_SDF="/tmp/robot_${i}.sdf"
    sed "s/ROBOT_ID/${i}/g" "$TEMPLATE" > "$TMP_SDF"

    ros2 run ros_gz_sim create \
        -world  "$WORLD"       \
        -file   "$TMP_SDF"     \
        -name   "robot_${i}"   \
        -x "$X" -y "$Y" -z "$Z" \
        > "/tmp/spawn_log_${i}.txt" 2>&1 &
    SPAWN_PIDS+=($!)
    echo -e "  Spawning ${YELLOW}robot_${i}${NC} (PID ${SPAWN_PIDS[-1]})..."
done

# Chờ tất cả spawn hoàn tất
FAIL=0
for i in $(seq 1 "$N"); do
    wait "${SPAWN_PIDS[$((i-1))]}"
    EXIT_CODE=$?
    if [ $EXIT_CODE -ne 0 ]; then
        echo -e "  ${RED}[FAIL]${NC} robot_${i} spawn lỗi (exit=$EXIT_CODE)"
        cat "/tmp/spawn_log_${i}.txt"
        FAIL=1
    else
        echo -e "  ${GREEN}[OK]${NC}   robot_${i} spawned."
    fi
done

if [ $FAIL -ne 0 ]; then
    echo -e "\n${RED}[ERROR]${NC} Một số robot spawn thất bại! Kiểm tra:"
    echo -e "  • Gazebo đã chạy với world ${WORLD}?"
    echo -e "  • ros_gz_sim đã cài? (ros2 run ros_gz_sim create --help)"
    exit 1
fi

echo -e "\n  ${GREEN}✓ Tất cả ${N} robot(s) đã spawn xong.${NC}"
sleep 1  # Cho Gazebo kịp khởi tạo entity

# ═════════════════════════════════════════════════════════════════════════════
# BƯỚC 2 — Khởi động N ros_gz_bridge song song
# ═════════════════════════════════════════════════════════════════════════════
echo ""
echo -e "${GREEN}[2/3]${NC} Khởi động ${YELLOW}${N}${NC} ros_gz_bridge(s)..."

for i in $(seq 1 "$N"); do
    ROBOT="robot_${i}"
    SCAN_T="/scan${i}"
    CMDVEL_T="/model/${ROBOT}/cmd_vel"
    ODOM_T="/model/${ROBOT}/odometry"
    POSE_T="/model/${ROBOT}/pose"

    ros2 run ros_gz_bridge parameter_bridge \
        "${SCAN_T}@sensor_msgs/msg/LaserScan[gz.msgs.LaserScan"       \
        "${CMDVEL_T}@geometry_msgs/msg/Twist]gz.msgs.Twist"            \
        "${ODOM_T}@nav_msgs/msg/Odometry[gz.msgs.Odometry"             \
        "${POSE_T}@geometry_msgs/msg/Pose[gz.msgs.Pose"                \
        --ros-args                                                       \
        -p "qos_overrides.${SCAN_T}.publisher.reliability:=best_effort" \
        -p "qos_overrides.${ODOM_T}.publisher.reliability:=best_effort" \
        -p "qos_overrides.${POSE_T}.publisher.reliability:=best_effort" \
        > "/tmp/bridge_log_${i}.txt" 2>&1 &
    BRIDGE_PIDS+=($!)
    echo -e "  Bridge ${YELLOW}${ROBOT}${NC}: scan+cmd_vel+odom+pose  (PID ${BRIDGE_PIDS[-1]})"
done

echo -e "\n  ${GREEN}✓ ${N} bridge(s) đang chạy.${NC}"
sleep 0.5

# ═════════════════════════════════════════════════════════════════════════════
# BƯỚC 3 — Khởi động N×2 static_transform_publisher
# ═════════════════════════════════════════════════════════════════════════════
echo ""
echo -e "${GREEN}[3/3]${NC} Khởi động static TF publishers..."

for i in $(seq 1 "$N"); do
    ROBOT="robot_${i}"
    FRAME_WORLD="$WORLD"
    FRAME_CHASSIS="${ROBOT}/chassis"
    FRAME_LIDAR="${ROBOT}/chassis/gpu_lidar"

    # TF: world → chassis (vị trí spawn)
    ros2 run tf2_ros static_transform_publisher \
        "$X" "$Y" "$Z"  0 0 0 \
        "$FRAME_WORLD" "$FRAME_CHASSIS" \
        > /dev/null 2>&1 &
    TF_PIDS+=($!)

    # TF: chassis → gpu_lidar (offset từ SDF: 0.08 0 0.06)
    ros2 run tf2_ros static_transform_publisher \
        0.08 0 0.06  0 0 0 \
        "$FRAME_CHASSIS" "$FRAME_LIDAR" \
        > /dev/null 2>&1 &
    TF_PIDS+=($!)
done

echo -e "  ${GREEN}✓ $((N*2)) TF publisher(s) đang chạy.${NC}"

# ═════════════════════════════════════════════════════════════════════════════
# TÓM TẮT
# ═════════════════════════════════════════════════════════════════════════════
echo ""
echo -e "  ${GREEN}╔══════════════════════════════════════════════════════════════╗${NC}"
echo -e "  ${GREEN}║  ${BOLD}${N} robot(s) đã sẵn sàng${NC}${GREEN}                                  ║${NC}"
echo -e "  ${GREEN}╠══════════════════════════════════════════════════════════════╣${NC}"
echo -e "  ${GREEN}║${NC}  World      : ${WORLD}"
echo -e "  ${GREEN}║${NC}  START      : x=${X}  y=${Y}  z=${Z}"
echo -e "  ${GREEN}║${NC}  GOAL       : x=2.25  y=-3.25"
echo -e "  ${GREEN}║${NC}  ──────────────────────────────────────────────────────"
echo -e "  ${GREEN}║${NC}  Robots     : robot_1 .. robot_${N}"
echo -e "  ${GREEN}║${NC}  Scan topics: /scan1 .. /scan${N}  (36 tia, 10°/tia)"
echo -e "  ${GREEN}║${NC}  Cmd topics : /model/robot_1/cmd_vel .. /model/robot_${N}/cmd_vel"
echo -e "  ${GREEN}║${NC}  Pose topics: /model/robot_1/pose   .. /model/robot_${N}/pose"
echo -e "  ${GREEN}║${NC}  ──────────────────────────────────────────────────────"
echo -e "  ${GREEN}║${NC}  Bridge PIDs : ${BRIDGE_PIDS[*]}"
echo -e "  ${GREEN}║${NC}  TF PIDs     : ${TF_PIDS[*]}"
echo -e "  ${GREEN}╠══════════════════════════════════════════════════════════════╣${NC}"
echo -e "  ${GREEN}║${NC}  Test nhanh:"
echo -e "  ${GREEN}║${NC}    ros2 topic echo /scan1 --once"
echo -e "  ${GREEN}║${NC}    ros2 topic echo /model/robot_1/pose --once"
echo -e "  ${GREEN}║${NC}    ros2 topic pub /model/robot_1/cmd_vel \\"
echo -e "  ${GREEN}║${NC}      geometry_msgs/msg/Twist '{linear:{x:0.2}}'"
echo -e "  ${GREEN}╠══════════════════════════════════════════════════════════════╣${NC}"
echo -e "  ${GREEN}║${NC}  Chạy GA training:"
echo -e "  ${GREEN}║${NC}    python3 ga_nn_controller.py --robots ${N} --generations 100"
echo -e "  ${GREEN}╚══════════════════════════════════════════════════════════════╝${NC}"
echo ""
echo -e "  Nhấn ${RED}Ctrl+C${NC} để dừng TẤT CẢ ${N} bridge + TF publishers..."

# ── Kiểm tra bridge còn sống không (background health check) ─────────────────
(
    sleep 5
    DEAD=0
    for i in "${!BRIDGE_PIDS[@]}"; do
        if ! kill -0 "${BRIDGE_PIDS[$i]}" 2>/dev/null; then
            echo -e "\n  ${YELLOW}[WARN]${NC} Bridge robot_$((i+1)) (PID ${BRIDGE_PIDS[$i]}) đã chết!"
            echo -e "         Xem log: /tmp/bridge_log_$((i+1)).txt"
            DEAD=1
        fi
    done
    if [ $DEAD -eq 0 ]; then
        echo -e "  ${GREEN}[OK]${NC} Tất cả ${#BRIDGE_PIDS[@]} bridge(s) vẫn đang chạy sau 5s."
    fi
) &

# ── Giữ script sống; Ctrl+C sẽ gọi cleanup() ─────────────────────────────────
wait