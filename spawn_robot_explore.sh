#!/bin/bash
# spawn_robot_explore.sh — Spawn ONE mecanum robot for the explore-then-exit
# task on the 0.75 m maze (worlds/nhom8_maze75.sdf, world 'nhom8_maze75').
#
# Same template/bridges as spawn_robot_wheel.sh; only the world name and the
# spawn cell differ (start cell (col 0, row 2) of the 0.75 m maze).
#
# Usage: ./spawn_robot_explore.sh [robot_id]     # default robot_id = 1
set -u

ID=${1:-1}
if ! [[ "$ID" =~ ^[1-9][0-9]*$ ]]; then
    echo "[ERROR] robot_id must be a positive integer, got: '$ID'" >&2
    exit 1
fi

WORLD="nhom8_maze75"
X=2.405; Y=-2.535; Z=0    # start cell (col 0, row 2) center

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TEMPLATE="$SCRIPT_DIR/one_robot.sdf"
if [ ! -f "$TEMPLATE" ]; then
    echo "[ERROR] template not found: $TEMPLATE" >&2
    exit 1
fi

# Substitute robot id and remove the VelocityControl plugin block
TMP_SDF="/tmp/robot_explore_${ID}.sdf"
sed "s/ROBOT_ID/${ID}/g" "$TEMPLATE" \
    | sed '/gz-sim-velocity-control-system/,/<\/plugin>/d' > "$TMP_SDF"

if grep -q "velocity-control" "$TMP_SDF"; then
    echo "[ERROR] failed to strip VelocityControl plugin from SDF" >&2
    exit 1
fi

echo "[1/2] Spawning robot_${ID} at ($X, $Y) into world '$WORLD'..."
if ! ros2 run ros_gz_sim create \
        -world "$WORLD" -file "$TMP_SDF" -name "robot_${ID}" \
        -x "$X" -y "$Y" -z "$Z"; then
    echo "[ERROR] spawn failed. Is Gazebo running with worlds/nhom8_maze75.sdf?" >&2
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
echo "  robot_${ID} ready (explore-then-exit mode, 0.75 m maze)"
echo "  Scan : /scan${ID}    Pose : /model/robot_${ID}/pose"
echo "  Map  : /map (nav_msgs/OccupancyGrid, latched — hiển thị trong RViz)"
echo ""
wait "$BRIDGE_PID"
