#!/bin/bash
# spawn_robot_maze.sh — Spawn ONE mecanum robot into a parameterized maze world.
#
# Derived from spawn_robot_explore.sh (same template SDF handling, same
# ros_gz_bridge setup, same trap/cleanup); only the world name and the spawn
# pose are parameterized, so the script works for the per-maze worlds and the
# combined maze world.
#
# Usage: ./spawn_robot_maze.sh WORLD_NAME X Y YAW_DEG [ROBOT_ID]   # default robot_id = 1
#   WORLD_NAME : gz world name (e.g. nhom8_maze_sigma_1 or nhom8_maze_multi)
#   X Y        : robot spawn position (m, world frame)
#   YAW_DEG    : robot spawn yaw in DEGREES (maze registry stores start_yaw_deg)
set -u

usage() {
    echo "Usage: $0 WORLD_NAME X Y YAW_DEG [ROBOT_ID]" >&2
    echo "  WORLD_NAME : gz world name (e.g. nhom8_maze_sigma_1 or nhom8_maze_multi)" >&2
    echo "  X Y        : robot spawn position (m, world frame)" >&2
    echo "  YAW_DEG    : robot spawn yaw in degrees (maze registry start_yaw_deg)" >&2
    echo "  ROBOT_ID   : optional, default 1, positive integer" >&2
}

if [ $# -lt 4 ] || [ $# -gt 5 ]; then
    echo "[ERROR] wrong number of arguments" >&2
    usage
    exit 1
fi

WORLD="$1"
X="$2"
Y="$3"
YAW_DEG="$4"
ID=${5:-1}

NUM_RE='^[+-]?([0-9]+([.][0-9]*)?|[.][0-9]+)$'
for v in "$X" "$Y" "$YAW_DEG"; do
    if ! [[ "$v" =~ $NUM_RE ]]; then
        echo "[ERROR] X, Y and YAW_DEG must be numeric, got: '$v'" >&2
        usage
        exit 1
    fi
done

if ! [[ "$ID" =~ ^[1-9][0-9]*$ ]]; then
    echo "[ERROR] robot_id must be a positive integer, got: '$ID'" >&2
    exit 1
fi

# YAW_DEG -> radians (python3 preferred, awk fallback)
if command -v python3 >/dev/null 2>&1; then
    YAW_RAD=$(python3 -c "import math,sys; print(math.radians(float(sys.argv[1])))" "$YAW_DEG")
else
    YAW_RAD=$(awk "BEGIN{print $YAW_DEG*3.14159265358979/180}")
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TEMPLATE="$SCRIPT_DIR/one_robot.sdf"
if [ ! -f "$TEMPLATE" ]; then
    echo "[ERROR] template not found: $TEMPLATE" >&2
    exit 1
fi

# Substitute robot id and remove the VelocityControl plugin block
TMP_SDF="/tmp/robot_maze_${ID}.sdf"
sed "s/ROBOT_ID/${ID}/g" "$TEMPLATE" \
    | sed '/gz-sim-velocity-control-system/,/<\/plugin>/d' > "$TMP_SDF"

if grep -q "velocity-control" "$TMP_SDF"; then
    echo "[ERROR] failed to strip VelocityControl plugin from SDF" >&2
    exit 1
fi

echo "[1/2] Spawning robot_${ID} at ($X, $Y) yaw=${YAW_DEG} deg into world '$WORLD'..."
if ! ros2 run ros_gz_sim create \
        -world "$WORLD" -file "$TMP_SDF" -name "robot_${ID}" \
        -x "$X" -y "$Y" -z 0 -Y "$YAW_RAD"; then
    echo "[ERROR] spawn failed. Is Gazebo running with world '$WORLD'?" >&2
    exit 1
fi
sleep 1

echo "[2/2] Starting ros_gz_bridge (scan, pose, 4 wheel commands)..."
BRIDGE_PID=""
cleanup() {
    echo ""
    echo "[STOP] Stopping bridge..."
    [ -n "$BRIDGE_PID" ] && kill "$BRIDGE_PID" 2>/dev/null
    exit 0
}
trap cleanup INT TERM

ros2 run ros_gz_bridge parameter_bridge \
    "/scan${ID}@sensor_msgs/msg/LaserScan[gz.msgs.LaserScan" \
    "/model/robot_${ID}/pose@geometry_msgs/msg/Pose[gz.msgs.Pose" \
    "/wheel_fl_${ID}@std_msgs/msg/Float64]gz.msgs.Double" \
    "/wheel_fr_${ID}@std_msgs/msg/Float64]gz.msgs.Double" \
    "/wheel_rl_${ID}@std_msgs/msg/Float64]gz.msgs.Double" \
    "/wheel_rr_${ID}@std_msgs/msg/Float64]gz.msgs.Double" \
    --ros-args \
    -p "qos_overrides./scan${ID}.publisher.reliability:=best_effort" \
    -p "qos_overrides./model/robot_${ID}/pose.publisher.reliability:=best_effort" &
BRIDGE_PID=$!
sleep 1

if ! kill -0 "$BRIDGE_PID" 2>/dev/null; then
    echo "[ERROR] ros_gz_bridge died on startup" >&2
    exit 1
fi

echo ""
echo "  robot_${ID} ready (maze mode, world '$WORLD')"
echo "  Spawn pose: ($X, $Y) yaw=${YAW_DEG} deg (${YAW_RAD} rad)"
echo "  Scan : /scan${ID}    Pose : /model/robot_${ID}/pose"
echo "  Map  : /map (nav_msgs/OccupancyGrid, latched — hiển thị trong RViz)"
echo ""
echo "  For the 13 registry mazes, the start pose comes from"
echo "  worlds/maze_defs/<name>.json (start_xy + start_yaw_deg) with the maze"
echo "  bbox center placed at (3.965, -2.535). For 'nhom8_maze_multi' the"
echo "  combined world layout is a 4x4 grid with 8 m spacing."
echo "  # Example: ./spawn_robot_maze.sh nhom8_maze_multi 1.60 -0.75 -56.6 1"
echo "  (illustrative only — real poses come from the registry CLI)"
echo ""
wait "$BRIDGE_PID"
