#!/bin/bash
#  spawn_robot.sh  —  Spawn mecanum robot N vào world nhom8_mecanum
#                     + ros_gz_bridge  + static_transform_publisher
#  Cách dùng:
#    ./spawn_robot.sh <N>  [x]  [y]  [z]
#
#  Ví dụ:
#    ./spawn_robot.sh 1
#    ./spawn_robot.sh 2 5.0 0.0 0.5
#    ./spawn_robot.sh 3 10.0 2.0 0.5
# ---------- màu in terminal ----------
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; NC='\033[0m'

# ---------- tham số ----------
N=${1:?"Thiếu số robot. Dùng: ./spawn_robot.sh <N> [x] [y] [z]"}
X=${2:-0.0}
Y=${3:-0.0}
Z=${4:-0.5}
WORLD="nhom8_mecanum"

# ---------- tên model & topic ----------
ROBOT_NAME="robot_${N}"
SCAN_TOPIC="/scan${N}"
CMDVEL_TOPIC="/model/${ROBOT_NAME}/cmd_vel"
ODOM_TOPIC="/model/${ROBOT_NAME}/odometry"
FRAME_CHASSIS="${ROBOT_NAME}/chassis"
FRAME_LIDAR="${ROBOT_NAME}/chassis/gpu_lidar"

# ---------- đường dẫn ----------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TEMPLATE="$SCRIPT_DIR/one_robot.sdf"
TMP_SDF="/tmp/${ROBOT_NAME}.sdf"

if [ ! -f "$TEMPLATE" ]; then
  echo -e "${RED}[ERROR]${NC} Không tìm thấy template: $TEMPLATE"
  exit 1
fi

# BƯỚC 1 — Tạo SDF tạm: thay ROBOT_ID → N
sed "s/ROBOT_ID/${N}/g" "$TEMPLATE" > "$TMP_SDF"
echo -e "${GREEN}[1/3]${NC} Tạo SDF tạm: $TMP_SDF"
echo -e "      model_name = ${YELLOW}${ROBOT_NAME}${NC}  |  lidar_topic = ${YELLOW}${SCAN_TOPIC}${NC}"

# BƯỚC 2 — Spawn model vào Gazebo
echo -e "${GREEN}[2/3]${NC} Spawning ${ROBOT_NAME} vào world '${WORLD}' tại (${X}, ${Y}, ${Z})..."

ros2 run ros_gz_sim create \
  -world  "$WORLD"       \
  -file   "$TMP_SDF"     \
  -name   "$ROBOT_NAME"  \
  -x "$X" -y "$Y" -z "$Z"

if [ $? -ne 0 ]; then
  echo -e "${RED}[ERROR]${NC} Spawn thất bại! Kiểm tra Gazebo đã chạy chưa."
  exit 1
fi
echo -e "      ${GREEN}Spawn thành công.${NC}"

# BƯỚC 3 — ros_gz_bridge: bridge topics cho robot N
echo -e "${GREEN}[3/3]${NC} Khởi động ros_gz_bridge cho ${ROBOT_NAME}..."

ros2 run ros_gz_bridge parameter_bridge \
  "${SCAN_TOPIC}@sensor_msgs/msg/LaserScan[gz.msgs.LaserScan" \
  "${CMDVEL_TOPIC}@geometry_msgs/msg/Twist]gz.msgs.Twist" \
  "${ODOM_TOPIC}@nav_msgs/msg/Odometry[gz.msgs.Odometry" \
  --ros-args \
    -p "qos_overrides.${SCAN_TOPIC}.publisher.reliability:=best_effort" \
    -p "qos_overrides.${ODOM_TOPIC}.publisher.reliability:=best_effort" \
  &
BRIDGE_PID=$!
echo -e "      bridge PID = ${YELLOW}${BRIDGE_PID}${NC}"
echo -e "      ${SCAN_TOPIC}        LaserScan  (Gz -> ROS)"
echo -e "      ${CMDVEL_TOPIC}   Twist      (ROS -> Gz)"
echo -e "      ${ODOM_TOPIC}  Odometry   (Gz -> ROS)"

# BƯỚC 4 — Static TF publishers
# 4a. world frame (nhom8_mecanum) → chassis robot N
echo -e "${GREEN}[TF]${NC} ${WORLD} -> ${FRAME_CHASSIS}"
ros2 run tf2_ros static_transform_publisher \
  "$X" "$Y" "$Z"  0 0 0 \
  "$WORLD" "$FRAME_CHASSIS" &
TF_WORLD_PID=$!

# 4b. chassis → gpu_lidar
echo -e "${GREEN}[TF]${NC} ${FRAME_CHASSIS} -> ${FRAME_LIDAR}"
ros2 run tf2_ros static_transform_publisher \
  0 0 0.1  0 0 0 \
  "$FRAME_CHASSIS" "$FRAME_LIDAR" &
TF_LIDAR_PID=$!

# Tóm tắt
echo ""
echo -e "  ${GREEN}${ROBOT_NAME}${NC} da san sang"
echo -e "  World        : ${WORLD}"
echo -e "  Vi tri       : x=${X}  y=${Y}  z=${Z}"
echo -e "  Lidar topic  : ${YELLOW}${SCAN_TOPIC}${NC}"
echo -e "  Cmd_vel      : ${YELLOW}${CMDVEL_TOPIC}${NC}"
echo -e "  Odometry     : ${YELLOW}${ODOM_TOPIC}${NC}"
echo -e "  TF world->chassis  PID: ${TF_WORLD_PID}"
echo -e "  TF chassis->lidar  PID: ${TF_LIDAR_PID}"
echo -e "  Bridge             PID: ${BRIDGE_PID}"
echo -e "  RViz2 Fixed Frame: ${YELLOW}${WORLD}${NC}"
echo -e "  Lang nghe lidar  : ros2 topic echo ${SCAN_TOPIC}"
echo -e "  Dieu khien:"
echo -e "    ros2 topic pub ${CMDVEL_TOPIC} geometry_msgs/msg/Twist '{linear:{x:0.5}}'"
echo ""
echo -e "Nhan ${RED}Ctrl+C${NC} de dung bridge va TF cua ${ROBOT_NAME}..."

# Giữ script sống; Ctrl+C kill bridge + tf cùng lúc
trap "echo -e '\nDung ${ROBOT_NAME}...'; kill $BRIDGE_PID $TF_WORLD_PID $TF_LIDAR_PID 2>/dev/null; exit 0" INT TERM
wait