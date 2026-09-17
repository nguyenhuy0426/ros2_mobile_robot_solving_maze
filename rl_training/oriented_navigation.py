"""Known-map rectangular-footprint route tracking with neural local actions."""
import math

import numpy as np
import torch
from torch import nn

from rl_training import config as C
from rl_training.audit_sigma_se2 import search, safe_pose
from rl_training.hybrid_navigation import H, integrate, wrap


class OrientedRoute:
    def __init__(self, spec, start_pose=None):
        # Reserve tracking error in the planned route, independently of the
        # tighter emergency command-check margin. Do not shrink the robot.
        result = search(spec, seconds=10, start_pose=start_pose, margin=.045)
        if not result["path_found"]:
            raise RuntimeError("no_safe_oriented_path")
        dense, commands = [], []
        for pose, (v, w, dt) in zip(result["path"], result["commands"]):
            if v < 0:
                raise RuntimeError("reverse route needs a bidirectional neural policy")
            dense.extend(integrate(pose, v, w, t) for t in np.linspace(0, dt, 16)[:-1])
            commands.extend([(v * .5, w * .5)] * 15)
        dense.append(np.asarray(result["path"][-1]))
        commands.append((0., 0.))
        self.poses = np.asarray(dense)
        self.commands = np.asarray(commands)
        self.index = 0

    def subgoal(self, xy):
        window = self.poses[self.index:self.index + 100, :2]
        self.index += int(np.argmin(np.linalg.norm(window - xy, axis=1)))
        target = self.poses[self.index, :2]
        for target in self.poses[self.index + 1:, :2]:
            if math.dist(xy, target) >= .07:
                break
        return target

    def reference(self, pose):
        window = self.poses[self.index:self.index + 100]
        cost = np.linalg.norm(window[:, :2] - pose[:2], axis=1) + .04 * np.abs(wrap(window[:, 2] - pose[2]))
        self.index += int(np.argmin(cost))
        return self.poses[self.index], self.commands[self.index]


def tracking_observation(scan, pose, reference, command):
    scan = np.asarray(scan)
    if scan.shape != (36,) or not np.isfinite(scan).all() or np.any(scan <= 0):
        raise ValueError("invalid LiDAR scan")
    dx, dy = reference[:2] - pose[:2]
    c, s = math.cos(pose[2]), math.sin(pose[2])
    angle = wrap(reference[2] - pose[2])
    return np.r_[np.clip(scan / C.LIDAR_MAX, 0, 1),
                 (c * dx + s * dy) / .2, (-s * dx + c * dy) / .2,
                 math.sin(angle), math.cos(angle), command[0] / .1, command[1] / .8].astype(np.float32)


def tracking_teacher(obs):
    ex, ey, sine, cosine, rv, rw = obs[:, 36:].T
    v = np.clip(rv * .1 * cosine + .5 * ex * .2, 0., .1)
    # During an in-place pivot, lateral error cannot be corrected by turning
    # towards the path: it can cancel the desired rotation and deadlock.
    lateral_gain = np.clip(rv / .3, 0., 1.)
    w = np.clip(rw * .8 + 2. * np.arctan2(sine, cosine) + 8. * ey * .2 * lateral_gain, -.8, .8)
    frontal = obs[:, [0, 1, 35]].min(axis=1) * C.LIDAR_MAX
    scale = np.clip((frontal - .1) / .2, 0., 1.)
    return np.column_stack((v / .1 * scale, w / .8 * scale)).astype(np.float32)


class TrackingPolicy(nn.Module):
    """42 inputs: LiDAR + relative reference pose and reference twist."""
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(42, 96), nn.Tanh(), nn.Linear(96, 96), nn.Tanh(), nn.Linear(96, 2))

    def forward(self, obs):
        return self.net(obs)

    @torch.inference_mode()
    def command(self, obs):
        out = self(torch.as_tensor(obs)[None])[0].cpu().numpy()
        return float(np.clip(out[0], 0, 1) * .1), float(np.clip(out[1], -1, 1) * .8)


def filter_rectangular(geometry, pose, scan, v, w):
    """Check map boxes and live LiDAR points against the swept rectangle.

    A 10 mm margin and interpolation allowance enclose intermediate poses.
    Stale data must be rejected by the caller. This is a simulation model
    check; sparse LiDAR does not certify arbitrary real-world obstacles.
    """
    scan = np.asarray(scan)
    if scan.shape != (C.N_RAYS,) or not np.isfinite([*pose, v, w, *scan]).all() or np.any(scan <= 0):
        return 0., 0., True
    angles = pose[2] + np.arange(C.N_RAYS) * 2 * np.pi / C.N_RAYS
    sensor = np.asarray(pose[:2]) + C.LIDAR_X_OFF * np.array([math.cos(pose[2]), math.sin(pose[2])])
    points = sensor + scan[:, None] * np.column_stack((np.cos(angles), np.sin(angles)))
    points = points[scan < C.LIDAR_MAX - 1e-3]
    for scale in (1., .5, .25, 0.):
        cv, cw = v * scale, w * scale
        horizon = H.reaction_time + max(abs(cv) / H.braking_accel, abs(cw) / 4.)
        times = np.linspace(0, horizon, max(2, math.ceil(horizon / .02) + 1))
        radius = math.hypot(C.CHASSIS_L / 2 + .01, C.WHEEL_SEP / 2 + .02)
        margin = .01 + (abs(cv) + radius * abs(cw)) * (times[1] - times[0]) / 2
        safe = True
        for t in times:
            future = integrate(pose, cv, cw, t)
            if not safe_pose(geometry, future, margin):
                safe = False
                break
            delta = points - future[:2]
            c, s = math.cos(future[2]), math.sin(future[2])
            if np.any((np.abs(delta[:, 0] * c + delta[:, 1] * s) <= C.CHASSIS_L / 2 + margin)
                      & (np.abs(-delta[:, 0] * s + delta[:, 1] * c) <= C.WHEEL_SEP / 2 + .01 + margin)):
                safe = False
                break
        if safe:
            return cv, cw, scale != 1.
    return 0., 0., True
