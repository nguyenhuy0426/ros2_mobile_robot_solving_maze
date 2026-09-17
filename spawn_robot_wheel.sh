#!/bin/bash
# spawn_robot_wheel.sh — Spawn ONE mecanum robot for direct wheel-velocity RL.
#
# Differences vs spawn_robot.sh:
#   * The VelocityControl (cmd_vel) plugin is STRIPPED from the SDF so it
#     cannot pin the model velocity and fight the 4 JointController plugins.
#   * Bridges the 4 wheel command topics (ROS Float64 -> gz.msgs.Double).
#
# Usage:
#   ./spawn_robot_wheel.sh [robot_id]     # default robot_id = 1
#
# Requires: Gazebo already running with worlds/nhom8_maze.sdf (unpaused),
#           ROS 2 Jazzy sourced.
set -u

ID=${1:-1}
if ! [[ "$ID" =~ ^[1-9][0-9]*$ ]]; then
    echo "[ERROR] robot_id must be a positive integer, got: '$ID'" >&2
    exit 1
fi

WORLD="nhom8_mecanum"
X=2.25; Y=-3.25; Z=0    # maze start cell (row 2, col 0)

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TEMPLATE="$SCRIPT_DIR/one_robot.sdf"
if [ ! -f "$TEMPLATE" ]; then
    echo "[ERROR] template not found: $TEMPLATE" >&2
    exit 1
fi

# Substitute robot id and remove the VelocityControl plugin block
TMP_SDF="/tmp/robot_wheel_${ID}.sdf"
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
    echo "[ERROR] spawn failed. Is Gazebo running with worlds/nhom8_maze.sdf?" >&2
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
echo "  robot_${ID} ready (wheel-control mode, no cmd_vel plugin)"
echo "  Scan : /scan${ID}"
echo "  Pose : /model/robot_${ID}/pose"
echo "  Wheels (std_msgs/Float64, rad/s):"
echo "    /wheel_fl_${ID}  /wheel_fr_${ID}  /wheel_rl_${ID}  /wheel_rr_${ID}"
echo ""
echo "  Quick test:"
echo "    ros2 topic pub -r 10 /wheel_fl_${ID} std_msgs/msg/Float64 '{data: 5.0}'"
echo ""
echo "  Ctrl+C stops the bridge."
wait "$BRIDGE_PID"
