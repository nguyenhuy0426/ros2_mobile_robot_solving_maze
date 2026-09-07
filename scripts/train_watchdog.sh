#!/usr/bin/env bash
# Watchdog for the multi-robot explore training stack.
#
# Every INTERVAL seconds it (1) appends a timestamped analyzer snapshot to the
# newest isolated run's logs/analysis.log and (2) checks the simulator half of the
# stack. The trainer itself is kept alive by its own supervisor loop; what
# that loop cannot repair is a dead Gazebo server, because every robot and
# every ros_gz bridge dies with it. So if the server is gone this script
# rebuilds server + GUI + all 13 robots, and the trainer's supervisor then
# reconnects on its next restart.
#
# Usage: ./scripts/train_watchdog.sh [INTERVAL_SEC]
set +u
cd /home/huynn/ros2_gazebo/ros2_ws
source /opt/ros/jazzy/setup.bash
source install/setup.bash

INTERVAL=${1:-600}
HUNG=0                  # consecutive silent-transport probes
WORLD=worlds/nhom8_maze_multi.sdf
RUN_ROOT=sac_explore_multi_runs
# Fleet size is a throughput/fidelity tradeoff, not "more is better". All 13
# gpu_lidar sensors render in one thread, so 13 robots hold the scan rate at
# 6.5 Hz against a configured 10 Hz -- and since the control loop waits on a
# fresh scan, that alone stretches the control period past EXPL_DT and lets
# each latched wheel command integrate for longer than one period. Measured
# end-to-end: N=4 gives 16 env-steps/s at 2.1 cm of travel per step, N=8 gives
# 9.8 at 4.9 cm (33 cm worst case, half a maze cell), N=13 does not run at all
# -- _wait_fresh_step trips its 5 s timeout within minutes. Override with
# MAZE_FLEET to re-measure.
ALL_MAZES=(delta_1 delta_2 ortho_1 ortho_2 ortho_3 ortho_4 ortho_5 ortho_6 \
       sigma_1 sigma_2 sigma_3 sigma_4 sigma_5)
# The fleet still trains on ALL 13 mazes: train_explore_multi deals the pool
# round-robin into MAZE_FLEET disjoint groups (_partition_mazes) and each
# robot cycles through its own group per episode. MAZES below is only where
# each robot is first SPAWNED, and mazes[k::n] starts with mazes[k], so the
# first MAZE_FLEET names are exactly the first maze of each group.
MAZES=("${ALL_MAZES[@]:0:${MAZE_FLEET:-4}}")

spawn_all() {
    local i=0
    for m in "${MAZES[@]}"; do
        i=$((i + 1))
        read -r W X Y YAW < <(rl_venv/bin/python -m rl_training.maze_registry \
                                --spawn "$m" --multi)
        setsid nohup ./spawn_robot_maze.sh "$W" "$X" "$Y" "$YAW" "$i" \
               > "/tmp/spawn_${i}_${m}.log" 2>&1 < /dev/null &
        sleep 3
    done
}

while true; do
    # Discover by the CSV, not by directory shape. A campaign root
    # (sac_explore_multi_runs/<campaign>/) is itself a directory one level
    # down and is newer than any attempt inside it, so a depth-limited find
    # returns the ROOT -- which has no logs/explore_multi.csv, so the
    # analyzer below fails silently and the snapshot is empty for the whole
    # campaign. Glob the CSV at any depth and take the newest.
    RUN=$(ls -1t "$RUN_ROOT"/*/logs/explore_multi.csv \
             "$RUN_ROOT"/*/*/logs/explore_multi.csv 2>/dev/null \
         | head -1)
    RUN="${RUN%/logs/explore_multi.csv}"
    OUT="${RUN:+$RUN/logs/analysis.log}"
    OUT="${OUT:-/tmp/maze_watchdog.log}"
    mkdir -p "$(dirname "$OUT")"
    {
        echo "================ $(date -Is) ================"
        # Real-time factor first: at RTF 0.11 an episode buys only ~5 m of
        # travel while visiting 25 zones needs ~19 m, so a silent RTF
        # collapse makes success physically impossible and looks exactly
        # like a policy that will not explore. Always read it before the
        # behavioural numbers below.
        echo "real_time_factor: $(timeout 8 gz topic -e -t /stats -n 1 \
              2>/dev/null | grep -m1 real_time_factor | awk '{print $2}')"
        if [[ -n "$RUN" ]]; then
            rl_venv/bin/python -m rl_training.analyze_explore_log \
                --csv "$RUN/logs/explore_multi.csv"
        else
            echo "[analyze] no isolated trainer run yet"
        fi
        echo
    } >> "$OUT" 2>&1

    # Liveness is NOT the same as responsiveness. A hung server keeps its
    # process (and so passes pgrep) while answering nothing on the transport:
    # set_pose then reports success without moving the robot, every reset
    # hands the trainer a stale pose, and each episode dies on step 1 with an
    # out-of-bounds -30 that goes straight into the replay buffer. That is
    # exactly how the v9 campaign collapsed -- 1059 poisoned episodes, mean
    # coverage 0.35 -> 0.00 -- with this watchdog reporting the stack healthy
    # the whole time. So probe the transport too, and require two consecutive
    # failures so a single timeout under load does not restart a good run.
    if pgrep -f "gz sim -r -s" > /dev/null; then
        if [[ -n "$(timeout 15 gz topic -l 2>/dev/null | head -1)" ]]; then
            HUNG=0
        else
            HUNG=$((HUNG + 1))
            echo "[watchdog] $(date -Is) gz transport silent ($HUNG/2)" >> "$OUT"
        fi
    else
        HUNG=2      # process gone: no point probing, rebuild now
    fi

    if [[ $HUNG -ge 2 ]]; then
        echo "[watchdog] $(date -Is) gz server DOWN or HUNG -- rebuilding" \
             >> "$OUT"
        # A hung server ignores SIGTERM, so the old pkill left it holding the
        # world name and the new one could never bind. Take it down hard.
        pkill -9 -f "gz sim -r -s"
        pkill -9 -f "gz sim -g"
        sleep 3
        HUNG=0
        pkill -f parameter_bridge
        pkill -f spawn_robot_maze
        setsid nohup gz sim -r -s --headless-rendering "$WORLD" \
               > /tmp/gz_multi.log 2>&1 < /dev/null &
        sleep 15
        setsid nohup gz sim -g > /tmp/gz_gui.log 2>&1 < /dev/null &
        sleep 5
        spawn_all
        echo "[watchdog] $(date -Is) stack rebuilt" >> "$OUT"
    fi

    sleep "$INTERVAL"
done
