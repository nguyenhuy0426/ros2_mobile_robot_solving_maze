"""Bounded, isolated Gazebo test of the neural hybrid with a KNOWN map.

Uses Gazebo pose for localization. Does not claim SLAM or real-robot success.
Starts its own Gazebo partition and ROS domain, cleans up only its processes.
Run after sourcing /opt/ros/jazzy/setup.bash. No long training is launched.
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import time
import uuid
import xml.etree.ElementTree as ET

import numpy as np
import torch

from rl_training import config as C
from rl_training.eval_hybrid import maze_setup
from rl_training.hybrid_navigation import H, LocalPolicy, filter_command, observation
from rl_training.maze_registry import load_registry
from rl_training.train_hybrid import teacher
from rl_training.holdout_mazes import generate_hex_holdout, generate_holdout
from rl_training.oriented_navigation import (OrientedRoute, filter_rectangular, TrackingPolicy,
                                              tracking_observation, tracking_teacher)

SCAN_MATCH_MAX_RESIDUAL = .08
SCAN_MATCH_XY_PRIOR = .10
SCAN_MATCH_YAW_PRIOR = .025


def worker_identity(domain):
    token = uuid.uuid4().hex
    return "hybrid_" + token, f"{domain}_{token[:12]}"


class CommandOdometry:
    """Unicycle dead reckoning from applied commands and simulation timestamps."""

    def __init__(self, pose, linear_scale=1., angular_scale=1.):
        self.pose = np.asarray(pose, dtype=float).copy()
        self.stamp = None
        self.v = 0.
        self.w = 0.
        self.linear_scale = float(linear_scale)
        self.angular_scale = float(angular_scale)

    def set_command(self, v, w):
        self.v, self.w = float(v), float(w)

    def update(self, stamp):
        stamp = float(stamp)
        if self.stamp is None:
            self.stamp = stamp
            return self.pose.copy()
        dt = stamp - self.stamp
        self.stamp = stamp
        if not 0 < dt <= 1.:
            return self.pose.copy()
        x, y, yaw = self.pose
        v = self.v * self.linear_scale
        w = self.w * self.angular_scale
        turn = w * dt
        if abs(w) < 1e-8:
            x += v * dt * math.cos(yaw)
            y += v * dt * math.sin(yaw)
        else:
            x += v / w * (math.sin(yaw + turn) - math.sin(yaw))
            y -= v / w * (math.cos(yaw + turn) - math.cos(yaw))
        self.pose[:] = x, y, math.atan2(math.sin(yaw + turn), math.cos(yaw + turn))
        return self.pose.copy()


def corrupt_lidar(scan, rng, noise_std=0., dropout=0.):
    """Apply deterministic sensor stress without changing the Gazebo truth audit."""
    values = np.asarray(scan, dtype=np.float32).copy()
    finite = np.isfinite(values)
    if noise_std:
        values[finite] += rng.normal(0., noise_std, int(finite.sum())).astype(np.float32)
    if dropout:
        values[rng.random(values.shape) < dropout] = C.LIDAR_MAX
    values[finite] = np.clip(values[finite], C.LIDAR_MIN, C.LIDAR_MAX)
    return values


def scan_match_pose(geometry, estimate, scan, valid_pose=None):
    """Two-stage local scan match with an odometry prior and rejection gate."""
    estimate = np.asarray(estimate, dtype=float)
    origin = estimate.copy()
    scan = np.asarray(scan, dtype=np.float32)
    valid = np.isfinite(scan) & (scan >= C.LIDAR_MIN) & (scan < C.LIDAR_MAX - .02)
    if valid.sum() < 20:
        return estimate.copy(), math.inf

    def search(center, xy_step, yaw_step):
        best_pose, best_cost, best_residual = center.copy(), math.inf, math.inf
        for dx in (-xy_step, 0., xy_step):
            for dy in (-xy_step, 0., xy_step):
                for dyaw in (-yaw_step, 0., yaw_step):
                    candidate = center + [dx, dy, dyaw]
                    candidate[2] = math.atan2(math.sin(candidate[2]), math.cos(candidate[2]))
                    if geometry.collides(candidate) or (valid_pose is not None and not valid_pose(candidate)):
                        continue
                    predicted = geometry.scan(candidate)
                    residual = np.minimum(np.abs(predicted[valid] - scan[valid]), .5)
                    measurement = float(np.mean(residual))
                    yaw_delta = abs(math.atan2(math.sin(candidate[2] - origin[2]),
                                               math.cos(candidate[2] - origin[2])))
                    cost = (measurement
                            + SCAN_MATCH_XY_PRIOR * float(np.linalg.norm(candidate[:2] - origin[:2]))
                            + SCAN_MATCH_YAW_PRIOR * yaw_delta)
                    if cost < best_cost:
                        best_pose, best_cost, best_residual = candidate, cost, measurement
        return best_pose, best_residual

    coarse, _ = search(estimate, .10, .15)
    corrected, residual = search(coarse, .035, .05)
    if residual > SCAN_MATCH_MAX_RESIDUAL:
        return estimate.copy(), residual
    return corrected, residual


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--maze", default="delta_1")
    p.add_argument("--holdout-seed", type=int,
                   help="generate a deterministic maze excluded from the training registry")
    p.add_argument("--hex-holdout-seed", type=int,
                   help="generate a deterministic hexagonal-cell maze")
    p.add_argument("--holdout-size", type=int, default=5)
    p.add_argument("--holdout-cell", type=float, default=.75)
    p.add_argument("--model", default="hybrid_runs/release_latest/local_policy.pt")
    p.add_argument("--tracking-model", default="hybrid_runs/release_latest/tracking_policy.pt")
    p.add_argument("--output", default="hybrid_runs/evaluations/gazebo_delta_1")
    p.add_argument("--seconds", type=float, default=180.)
    p.add_argument("--navigation", choices=["disk", "auto"], default="disk",
                   help="auto uses orientation-aware planning when disk planning fails")
    p.add_argument("--domain-id", type=int, default=87)
    p.add_argument("--seed", type=int, default=17)
    p.add_argument("--yaw-jitter", type=float, default=0.)
    p.add_argument("--lidar-noise-std", type=float, default=0.,
                   help="Gaussian range noise in metres applied to the controller scan")
    p.add_argument("--lidar-dropout", type=float, default=0.,
                   help="fraction of controller scan rays replaced by no-return")
    p.add_argument("--odom-linear-scale", type=float, default=1.,
                   help="systematic scale applied to linear command odometry")
    p.add_argument("--odom-angular-scale", type=float, default=1.,
                   help="systematic scale applied to angular command odometry")
    p.add_argument("--localization", choices=["truth", "command_odom", "scan_match"], default="truth",
                   help="controller pose source; non-truth modes never use live Gazebo pose for control")
    p.add_argument("--collect-data", action="store_true")
    p.add_argument("--expert-prob", type=float, default=0.,
                   help="teacher mixing for data collection only; evaluation uses zero")
    p.add_argument("--gui", action="store_true",
                   help="open the Gazebo GUI so the run can be watched live")
    p.add_argument("--keep-open", type=float, default=0.,
                   help="seconds to keep Gazebo visible after the run finishes")
    args = p.parse_args()
    if args.holdout_seed is not None and args.hex_holdout_seed is not None:
        p.error("choose either --holdout-seed or --hex-holdout-seed")
    if not 0 <= args.expert_prob <= 1 or (args.expert_prob and not args.collect_data):
        p.error("expert-prob requires collect-data and must be in [0, 1]")
    if (not 0 <= args.domain_id <= 200 or args.yaw_jitter < 0 or
            args.lidar_noise_std < 0 or not 0 <= args.lidar_dropout < 1 or
            args.odom_linear_scale <= 0 or args.odom_angular_scale <= 0):
        p.error("invalid domain, jitter, sensor noise, or odometry scale")
    rng = np.random.default_rng(args.seed)
    lidar_rng = np.random.default_rng(args.seed ^ 0x51A7)
    # Separate transport from any existing user simulation or controller.
    partition, robot_id = worker_identity(args.domain_id)
    os.environ["GZ_PARTITION"] = partition
    os.environ["GZ_IP"] = "127.0.0.1"
    os.environ["ROS_DOMAIN_ID"] = str(args.domain_id)
    from geometry_msgs.msg import Pose
    from sensor_msgs.msg import LaserScan
    from std_msgs.msg import Float64
    from ros_gz_interfaces.msg import Contacts
    import rclpy
    from rclpy.qos import qos_profile_sensor_data

    out = Path(args.output).resolve()
    out.mkdir(parents=True, exist_ok=True)
    generated = args.holdout_seed is not None or args.hex_holdout_seed is not None
    if args.hex_holdout_seed is not None:
        spec = generate_hex_holdout(args.hex_holdout_seed, args.holdout_size,
                                    args.holdout_cell)
    elif args.holdout_seed is not None:
        spec = generate_holdout(args.holdout_seed, args.holdout_size,
                                args.holdout_cell)
    else:
        spec = load_registry()[args.maze]
    args.maze = spec.name
    planner, geometry, goal = maze_setup(spec)
    path = planner.plan(spec.start_xy_local, goal)
    oriented = args.navigation == "auto" and not path
    route = None
    torch.set_num_threads(1)
    model_path = args.tracking_model if oriented else args.model
    model = TrackingPolicy() if oriented else LocalPolicy()
    model.load_state_dict(torch.load(model_path, map_location="cpu", weights_only=True))
    model.eval()
    processes, logs = [], []
    node, pubs = None, []
    scopes = {"truth": "Gazebo known-map + ground-truth pose, NOT SLAM",
              "command_odom": "Gazebo known-map + command dead reckoning, NOT SLAM",
              "scan_match": "Gazebo known-map + LiDAR scan-matched odometry, NOT SLAM"}
    scope = scopes[args.localization]
    result = {"scope": scope, "localization": args.localization,
              "maze": args.maze, "success": False, "geometry_collision": False,
              "navigation": "oriented" if oriented else "disk",
              "observation_dim": 42 if oriented else 38,
              "reason": "startup", "steps": 0, "overrides": 0,
              "contact_messages": 0, "wall_contacts": 0, "contact_pairs": [],
              "seed": args.seed, "domain_id": args.domain_id,
              "lidar_noise_std_m": args.lidar_noise_std,
              "lidar_dropout": args.lidar_dropout,
              "odom_linear_scale": args.odom_linear_scale,
              "odom_angular_scale": args.odom_angular_scale,
              "robot_id": robot_id, "partition": os.environ["GZ_PARTITION"],
              "expert_probability": args.expert_prob, "expert_steps": 0,
              "model_sha256": hashlib.sha256(Path(model_path).read_bytes()).hexdigest()}
    trace = []
    observations, labels = [], []
    stop_requested = False

    def interrupt(signum, frame):
        nonlocal stop_requested
        # Do not throw through rclpy's C++ callback boundary.
        stop_requested = True

    signal.signal(signal.SIGTERM, interrupt)
    signal.signal(signal.SIGINT, interrupt)

    def launch(cmd, name):
        log = (out / name).open("w")
        logs.append(log)
        proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        processes.append(proc)
        return proc

    def publish(v=0., w=0.):
        for pub, side in zip(pubs, (-1, 1, -1, 1)):
            pub.publish(Float64(data=float((v + side * w * C.WHEEL_SEP / 2) / C.WHEEL_RADIUS)))

    try:
        if not path and not oriented:
            raise RuntimeError("no_safe_path: footprint planning gate failed")
        world_name = f"nhom8_maze_{args.maze}"
        if not generated:
            world = ET.parse(C.WS_ROOT / "worlds" / f"{world_name}.sdf")
        else:
            from scripts.generate_maze_sdf_multi import build_standalone_text
            world = ET.ElementTree(ET.fromstring(build_standalone_text(spec, world_name)))
        ET.SubElement(world.getroot().find("world"), "plugin",
                      filename="gz-sim-contact-system", name="gz::sim::systems::Contact")
        world_path = out / "world.sdf"
        world.write(world_path)
        gz_cmd = ["gz", "sim", "-r"]
        if not args.gui:
            gz_cmd += ["-s", "--headless-rendering"]
        gz_cmd.append(str(world_path))
        launch(gz_cmd, "gazebo.log")
        robot = ET.fromstring((C.WS_ROOT / "one_robot.sdf").read_text().replace("ROBOT_ID", robot_id))
        robot_model = robot.find("model")
        for plugin in list(robot_model.findall("plugin")):
            if "velocity-control" in plugin.get("filename", ""):
                robot_model.remove(plugin)
        x, y = spec.start_xy_world
        spawn_yaw = spec.start_yaw + rng.uniform(-args.yaw_jitter, args.yaw_jitter)
        result["spawn_yaw"] = float(spawn_yaw)
        robot_model.find("pose").text = f"{x} {y} 0.024 0 0 {spawn_yaw}"
        contact_topics = []
        for link in robot_model.findall("link"):
            collision = link.find("collision")
            if collision is None:
                continue
            topic = f"/hybrid_contact_{robot_id}_" + link.get("name")
            contact_topics.append(topic)
            sensor = ET.SubElement(link, "sensor", name="audit_contact", type="contact")
            ET.SubElement(sensor, "always_on").text = "true"
            ET.SubElement(sensor, "update_rate").text = "100"
            contact = ET.SubElement(sensor, "contact")
            # gz-sim 8 Contact reads topic from <contact>, not <sensor>.
            ET.SubElement(contact, "topic").text = topic
            ET.SubElement(contact, "collision").text = collision.get("name")
        # Create acknowledges the queued request before reading the file.
        # Keep the exact model alongside the evidence for the whole run.
        sdf = out / "robot.sdf"
        ET.ElementTree(robot).write(sdf)
        deadline = time.monotonic() + 25
        spawned = False
        while time.monotonic() < deadline and not stop_requested:
            call = subprocess.run(["gz", "service", "-s", f"/world/{world_name}/create",
                "--reqtype", "gz.msgs.EntityFactory", "--reptype", "gz.msgs.Boolean",
                "--timeout", "2000", "--req", f'sdf_filename: "{sdf}"'],
                capture_output=True, text=True, timeout=4)
            if call.returncode == 0 and "data: true" in call.stdout:
                spawned = True
                break
            time.sleep(0.5)
        if not spawned:
            raise RuntimeError("Gazebo spawn service did not succeed")
        launch(["ros2", "run", "ros_gz_bridge", "parameter_bridge",
                f"/scan{robot_id}@sensor_msgs/msg/LaserScan[gz.msgs.LaserScan",
                f"/model/robot_{robot_id}/pose@geometry_msgs/msg/Pose[gz.msgs.Pose",
                *[f"{topic}@ros_gz_interfaces/msg/Contacts[gz.msgs.Contacts" for topic in contact_topics],
                *[f"/wheel_{side}_{robot_id}@std_msgs/msg/Float64]gz.msgs.Double" for side in ("fl", "fr", "rl", "rr")]], "bridge.log")
        rclpy.init()
        node = rclpy.create_node(f"hybrid_eval_{robot_id}")
        state = {"scan": None, "pose": None, "seq": 0, "scan_time": 0., "pose_time": 0.,
                 "scan_stamp": None}

        def scan_callback(msg):
            ranges = np.asarray(msg.ranges, dtype=np.float32)
            # +inf denotes no return, NaN and -inf remain invalid.
            ranges = np.where(np.isposinf(ranges), C.LIDAR_MAX, ranges)
            stamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
            ranges = corrupt_lidar(ranges, lidar_rng, args.lidar_noise_std, args.lidar_dropout)
            state.update(scan=ranges, seq=state["seq"] + 1, scan_time=time.monotonic(),
                         scan_stamp=stamp)

        def pose_callback(msg):
            q = msg.orientation
            yaw = math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z))
            lx, ly = spec.to_local((msg.position.x, msg.position.y))
            state.update(pose=np.array([lx, ly, yaw]), pose_time=time.monotonic())

        def contact_callback(msg):
            result["contact_messages"] += 1
            for contact in msg.contacts:
                pair = [contact.collision1.name, contact.collision2.name]
                if pair not in result["contact_pairs"] and len(result["contact_pairs"]) < 20:
                    result["contact_pairs"].append(pair)
                if "wall" in contact.collision1.name or "wall" in contact.collision2.name:
                    result["wall_contacts"] += 1

        node.create_subscription(LaserScan, f"/scan{robot_id}", scan_callback, qos_profile_sensor_data)
        node.create_subscription(Pose, f"/model/robot_{robot_id}/pose", pose_callback, qos_profile_sensor_data)
        for topic in contact_topics:
            node.create_subscription(Contacts, topic, contact_callback, qos_profile_sensor_data)
        pubs = [node.create_publisher(Float64, f"/wheel_{side}_{robot_id}", 10) for side in ("fl", "fr", "rl", "rr")]
        deadline = time.monotonic() + 30
        while time.monotonic() < deadline and not stop_requested and (state["scan"] is None or state["pose"] is None):
            rclpy.spin_once(node, timeout_sec=0.05)
        if state["scan"] is None or state["pose"] is None:
            raise RuntimeError("missing LiDAR or pose stream; see bridge/gazebo logs")
        if math.dist(state["pose"][:2], spec.start_xy_local) > 0.15:
            raise RuntimeError("spawn pose mismatch")
        odometry = CommandOdometry(state["pose"], args.odom_linear_scale,
                                   args.odom_angular_scale)
        odometry.update(state["scan_stamp"])
        last_scan_match = -math.inf
        result["scan_match_corrections"] = 0
        result["scan_match_attempts"] = 0
        result["scan_match_rejections"] = 0
        match_score = math.inf
        if oriented:
            route = OrientedRoute(spec, start_pose=odometry.pose)
            (out / "planned_route.json").write_text(json.dumps(route.poses.tolist()))
        begin, used = time.monotonic(), -1
        motion_pose, motion_time = state["pose"].copy(), begin
        override_streak = 0
        result["replans"] = 0
        result["reason"] = "timeout"
        while time.monotonic() - begin < args.seconds and not stop_requested:
            rclpy.spin_once(node, timeout_sec=0.01)
            now = time.monotonic()
            if now - min(state["scan_time"], state["pose_time"]) > 0.5:
                publish()
                if now - min(state["scan_time"], state["pose_time"]) > 5:
                    result["reason"] = "stale_sensors"
                    break
                continue
            if used == state["seq"]:
                continue
            used = state["seq"]
            true_pose, scan = state["pose"], state["scan"]
            control_pose = (true_pose.copy() if args.localization == "truth"
                            else odometry.update(state["scan_stamp"]))
            if args.localization == "scan_match" and state["scan_stamp"] - last_scan_match >= .5:
                matched_pose, match_score = scan_match_pose(
                    geometry, control_pose, scan,
                    valid_pose=lambda p: planner.valid(planner.cell(p[:2])))
                result["scan_match_attempts"] += 1
                if match_score <= SCAN_MATCH_MAX_RESIDUAL:
                    control_pose = matched_pose
                    result["scan_match_corrections"] += 1
                else:
                    result["scan_match_rejections"] += 1
                odometry.pose[:] = control_pose
                last_scan_match = state["scan_stamp"]
                result["last_scan_match_score"] = match_score
            if np.linalg.norm(true_pose[:2] - motion_pose[:2]) > 0.003 or abs(math.atan2(
                    math.sin(true_pose[2] - motion_pose[2]), math.cos(true_pose[2] - motion_pose[2]))) > 0.02:
                motion_pose, motion_time = true_pose.copy(), now
            elif now - motion_time > 10:
                if args.localization != "truth":
                    result["reason"] = "localization_stall"
                    break
                if trace and any(abs(t["v"]) > 0.02 or abs(t["w"]) > 0.2 for t in trace[-20:]):
                    result["reason"] = "no_motion_response"
                    break
            if result["wall_contacts"]:
                result["reason"] = "wall_contact"
                break
            if geometry.collides(true_pose):
                result["geometry_collision"], result["reason"] = True, "geometry_collision"
                break
            if math.dist(true_pose[:2], goal) < H.goal_tolerance:
                result["success"], result["reason"] = True, "goal"
                break
            if route is not None:
                reference, ref_command = route.reference(control_pose)
                obs = tracking_observation(scan, control_pose, reference, ref_command)
            else:
                obs = observation(scan, control_pose, planner.subgoal(control_pose[:2], path))
            v, w = model.command(obs)
            if args.collect_data:
                label = (tracking_teacher(obs[None]) if oriented else teacher(obs[None]))[0]
                observations.append(obs)
                labels.append(label)
                if rng.random() < args.expert_prob:
                    v, w = float(label[0] * (.1 if oriented else H.max_v)), float(label[1] * (.8 if oriented else H.max_w))
                    result["expert_steps"] += 1
            v, w, override = (filter_rectangular(geometry, control_pose, scan, v, w) if oriented
                              else filter_command(planner, control_pose, scan, v, w))
            override_streak = override_streak + 1 if override else 0
            if not oriented and override_streak and override_streak % 10 == 0:
                replanned = planner.plan(control_pose[:2], goal)
                if replanned:
                    path = replanned
                    result["replans"] += 1
            publish(v, w)
            odometry.set_command(v, w)
            result["steps"] += 1
            result["overrides"] += int(override)
            pose_error = float(np.linalg.norm(control_pose[:2] - true_pose[:2]))
            trace.append({"wall_seconds": now - begin, "pose": true_pose.tolist(),
                          "control_pose": control_pose.tolist(), "pose_error_m": pose_error,
                          "v": v, "w": w,
                          "scan_min": float(np.min(scan)), "override": override,
                          "scan_match_score": match_score})
            if result["steps"] % 50 == 0:
                print(f"step={result['steps']} pose={true_pose.round(3)} "
                      f"odom_error={pose_error:.3f} goal_distance={math.dist(true_pose[:2], goal):.2f}",
                      flush=True)
        result["seconds"] = time.monotonic() - begin
        if trace:
            result["final_pose_error_m"] = trace[-1]["pose_error_m"]
            result["max_pose_error_m"] = max(row["pose_error_m"] for row in trace)
    except KeyboardInterrupt:
        result["reason"] = "interrupted"
    except Exception as exc:
        result["reason"] = f"error: {exc}"
    finally:
        # A second termination signal must not interrupt simulator cleanup.
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        if node is not None:
            try:
                publish()
                rclpy.spin_once(node, timeout_sec=0.1)
                node.destroy_node()
                rclpy.shutdown()
            except Exception:
                pass  # Simulator/bridge cleanup must still happen.
        if args.gui and args.keep_open > 0:
            time.sleep(args.keep_open)
        for proc in reversed(processes):
            if proc.poll() is None:
                try:
                    os.killpg(proc.pid, signal.SIGTERM)
                except ProcessLookupError:
                    continue
                try:
                    proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    os.killpg(proc.pid, signal.SIGKILL)
        for log in logs:
            log.close()
        result["contact_monitor_observed"] = result["contact_messages"] > 0
        if stop_requested:
            result.update(success=False, reason="interrupted")
        if result["wall_contacts"]:
            result["success"] = False
            result["reason"] = "wall_contact"
        (out / "result.json").write_text(json.dumps(result, indent=2) + "\n")
        (out / "trajectory.json").write_text(json.dumps(trace) + "\n")
        if args.collect_data:
            np.savez_compressed(out / "demonstrations.npz",
                                observations=np.asarray(observations, dtype=np.float32).reshape(-1, result["observation_dim"]),
                                labels=np.asarray(labels, dtype=np.float32).reshape(-1, 2))
    print(json.dumps(result, indent=2), flush=True)
    return 0 if result["success"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
