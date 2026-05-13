#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run.py  ─  GA-NN Mecanum Maze Controller
GA with neural network controller for solving a specific 5×5 maze.

Architecture: Parallel multi-maze evaluation with continuous queue.
Each genome is evaluated sequentially in one maze (no lidar cross-talk).

Usage:
  gz sim nhom8_multi5.sdf                        # Step 1: open Gazebo
  ./spawn_robot.sh 5                             # Step 2: spawn 5 robots
  python3 run.py --pop 50 --gen 200 --workers 5  # Step 3: train
  python3 run.py --load ga_checkpoints_v2/best.json --workers 5  # resume
  python3 run.py --run  ga_checkpoints_v2/best.json              # demo
  python3 run.py --info                          # print architecture
"""

import argparse
import json
import math
import queue
import subprocess
import os
import sys
import threading
import time
from datetime import datetime

import numpy as np
import rclpy
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from geometry_msgs.msg import Pose, Twist
from sensor_msgs.msg import LaserScan


# CONFIG  — tất cả hằng số ở đây

class MazeCfg:
    """
    Per-maze coordinates — tất cả mazes giống nhau về topology,
    chỉ khác nhau ở X offset (từ gen_multi_maze.py).
    """
    def __init__(self, robot_id: int,
                 start_x: float, start_y: float,
                 goal_x: float,  goal_y: float,
                 x_min: float,   x_max: float,
                 y_min: float,   y_max: float):
        self.robot_id = robot_id
        self.start_x  = start_x
        self.start_y  = start_y
        self.goal_x   = goal_x
        self.goal_y   = goal_y
        self.x_min    = x_min
        self.x_max    = x_max
        self.y_min    = y_min
        self.y_max    = y_max
        self.cx       = (x_min + x_max) / 2.0
        self.cy       = (y_min + y_max) / 2.0
        self.start_d  = _dist(start_x, start_y, goal_x, goal_y)

def _default_maze_configs(n: int) -> list:
    """
    Tạo N MazeCfg với X offset = 15m/maze.
    Dùng khi --workers N được chỉ định mà không có MAZE_CONFIGS tùy chỉnh.
    Maze gốc: x∈[2.0,4.5] → offset = 15.0m/maze (lidar 12m không với tới maze kế)
    """
    DY = 3.5
    configs = []
    for i in range(n):
        dy = -DY * i   # y đi xuống cho maze tiếp theo
        configs.append(MazeCfg(
            robot_id = i + 1,
            start_x  = 2.25,
            start_y  = -3.25 - dy,
            goal_x   = 4.25,
            goal_y   = -3.25 - dy,
            x_min    = 1.85,
            x_max    = 4.65,
            y_min    = -4.65 - dy,
            y_max    = -1.85 - dy,
        ))
    return configs

class Cfg:
    # Gazebo
    WORLD = "nhom8_mecanum"
    # Robot
    WHEEL_R = 0.024
    LX, LY  = 0.08, 0.09
    MAX_W   = 12.0

    # Lidar — 12 rays every 30° for full 360° coverage (no blind spots)
    N_RAYS    = 36
    LIDAR_MAX = 12.0
    OBS_IDX   = [0, 3, 6, 9, 12, 15, 18, 21, 24, 27, 30, 33]  # 12 rays
    ELIM_DIST = 0.08
    AIRBORNE_Z = 0.10

    # Network: 16 → 16(tanh) → 8(tanh) → 4  (444 weights)
    N_IN  = 16
    N_H1  = 16
    N_H2  = 8
    N_OUT = 4
    N_W   = N_IN*N_H1 + N_H1 + N_H1*N_H2 + N_H2 + N_H2*N_OUT + N_OUT  # 444

    # GA
    POP       = 50
    ELITE     = 5
    TOURN_K   = 3
    CX_RATE   = 0.70
    MUT_P     = 0.12
    MUT_STD   = 0.25
    W_CLIP    = 5.0
    INIT_S    = 0.5
    STAG_LIM  = 15
    STAG_FRAC = 0.40
    SCOUT_FRAC= 0.10   # fraction of pop randomly generated each gen

    # Episode
    MAX_STEPS = 600
    STEP_DT   = 0.20
    TELE_WAIT = 1.2
    PARK_Z    = -5.0
    GOAL_R    = 0.20

    # Fitness v8 (clean reward/penalty, no safety system)
    START_DIST  = 2.0
    P_COLL      = 300.0
    P_OOB       = 300.0
    P_TIMEOUT   = 50.0
    P_STATIONARY= 100.0
    TIME_COST   = 0.5
    R_PROGRESS  = 500.0
    R_BEST_PROG = 250.0
    R_GOAL      = 5000.0
    R_SPEED     = 3000.0
    R_CELLS     = 2.0
    CELL_SZ     = 0.25

    # Action smoothing EMA
    EMA_ALPHA = 0.7

    # Velocity clamps — allow reverse for dead-end recovery
    VX_MAX =  0.20
    VX_MIN = -0.10
    VY_MAX =  0.15
    WZ_MAX =  2.0

    # Logging
    LOG  = "ga_training_v2.log"
    CSV  = "ga_stats_v2.csv"
    CKPT = "ga_checkpoints_v2"

C = Cfg()

# NEURAL NETWORK  (NumPy only)

class MLP:
    """
    3-layer MLP. Genome = flat float32 array of length N_W.
    Forward: x(16) → tanh → tanh → clip[-1,1] → action(4)
    """

    def __init__(self, weights: np.ndarray):
        i = 0
        def take(shape):
            nonlocal i
            n = int(np.prod(shape))
            out = weights[i:i+n].reshape(shape).copy()
            i  += n
            return out
        self.W1 = take((C.N_IN, C.N_H1));  self.b1 = take((C.N_H1,))
        self.W2 = take((C.N_H1, C.N_H2));  self.b2 = take((C.N_H2,))
        self.W3 = take((C.N_H2, C.N_OUT)); self.b3 = take((C.N_OUT,))

    def forward(self, x: np.ndarray) -> np.ndarray:
        h = np.tanh(x @ self.W1 + self.b1)
        h = np.tanh(h @ self.W2 + self.b2)
        return np.clip(h @ self.W3 + self.b3, -1.0, 1.0).astype(np.float32)

    @staticmethod
    def to_twist(a: np.ndarray):
        """Mecanum kinematics: 4 wheel values → (vx, vy, wz)."""
        r, lx, ly, mw = C.WHEEL_R, C.LX, C.LY, C.MAX_W
        fl, fr = float(a[0])*mw, float(a[1])*mw
        rl, rr = float(a[2])*mw, float(a[3])*mw
        vx = r/4*(fl+fr+rl+rr)       # allow reverse
        vy = r/4*(-fl+fr+rl-rr)
        wz = r/(4*(lx+ly))*(-fl+fr-rl+rr)
        return vx, vy, wz


# ROBOT AGENT  (ROS2 I/O only — no fitness logic here)

class RobotAgent(Node):
    """ROS2 node for 1 robot in 1 maze. EMA smoothing, Z-tracking."""

    def __init__(self, maze: "MazeCfg"):
        super().__init__(f"ga_agent_{maze.robot_id}")
        self._lock  = threading.Lock()
        self._scan  = [C.LIDAR_MAX] * C.N_RAYS
        self._x, self._y, self._z = maze.start_x, maze.start_y, 0.064
        self._yaw = 0.0
        self.maze   = maze

        # EMA smoothing state (reset each episode)
        self._prev_vx  = 0.0
        self._prev_vy  = 0.0
        self._prev_wz  = 0.0

        qos = rclpy.qos.QoSProfile(
            reliability=rclpy.qos.ReliabilityPolicy.BEST_EFFORT,
            history=rclpy.qos.HistoryPolicy.KEEP_LAST, depth=1)

        rid = maze.robot_id
        self.create_subscription(LaserScan, f"/scan{rid}", self._cb_scan, qos)
        self.create_subscription(Pose, f"/model/robot_{rid}/pose", self._cb_pose, qos)
        self._pub = self.create_publisher(Twist, f"/model/robot_{rid}/cmd_vel", 10)
        self.get_logger().info(f"Agent robot_{rid} maze offset={maze.start_x - 2.25:.1f}m")

    def _cb_scan(self, msg: LaserScan):
        with self._lock:
            self._scan = [
                r if (math.isfinite(r) and r > 0.0) else C.LIDAR_MAX
                for r in msg.ranges
            ]

    def _cb_pose(self, msg: Pose):
        with self._lock:
            self._x   = msg.position.x
            self._y   = msg.position.y
            self._z   = msg.position.z      # ← NEW: track height
            self._yaw = 2.0 * math.atan2(msg.orientation.z, msg.orientation.w)

    def observe(self) -> np.ndarray:
        """16-dim observation: 12 lidar + goal_angle + goal_dist + heading."""
        with self._lock:
            scan = list(self._scan)
            x, y, yaw = self._x, self._y, self._yaw
        m = self.maze
        lidar = np.array(
            [min(scan[i], C.LIDAR_MAX)/C.LIDAR_MAX for i in C.OBS_IDX],
            dtype=np.float32)
        dx = m.goal_x - x
        dy = m.goal_y - y
        goal_angle = math.atan2(dy, dx) - yaw
        goal_angle = (goal_angle + math.pi) % (2*math.pi) - math.pi
        dist = _dist(x, y, m.goal_x, m.goal_y)
        return np.array([
            *lidar,                                    # [0:12]  12 lidar rays
            goal_angle / math.pi,                      # [12]    goal angle in robot frame
            min(dist, m.start_d) / m.start_d,          # [13]    normalized distance
            math.sin(yaw), math.cos(yaw),              # [14:16] heading
        ], dtype=np.float32)

    def act(self, action: np.ndarray):
        """Velocity clamping + EMA smoothing. No wall avoidance override."""
        vx, vy, wz = MLP.to_twist(action)

        # Velocity clamping
        vx = max(C.VX_MIN, min(C.VX_MAX, vx))
        vy = max(-C.VY_MAX, min(C.VY_MAX, vy))
        wz = max(-C.WZ_MAX, min(C.WZ_MAX, wz))

        # EMA smoothing
        a = C.EMA_ALPHA
        vx = a * vx + (1 - a) * self._prev_vx
        vy = a * vy + (1 - a) * self._prev_vy
        wz = a * wz + (1 - a) * self._prev_wz
        self._prev_vx, self._prev_vy, self._prev_wz = vx, vy, wz

        self._pub_twist(vx, vy, wz)

    def stop(self):
        self._pub_twist(0.0, 0.0, 0.0)

    def _pub_twist(self, vx, vy, wz):
        msg = Twist()
        msg.linear.x, msg.linear.y, msg.angular.z = float(vx), float(vy), float(wz)
        self._pub.publish(msg)

    @property
    def pose(self):
        """Returns (x, y, z, yaw)."""
        with self._lock: return self._x, self._y, self._z, self._yaw

    @property
    def scan_min(self):
        with self._lock: return min(self._scan)

    def reset_episode(self):
        """Reset per-episode state: scan, EMA."""
        with self._lock:
            self._scan = [C.LIDAR_MAX] * C.N_RAYS
        self._prev_vx = self._prev_vy = self._prev_wz = 0.0

    def reset_scan(self):
        """Legacy compat — calls reset_episode."""
        self.reset_episode()


# EPISODE  — evaluate one genome, return structured result

class EpResult:
    def __init__(self, start_dist: float = C.START_DIST):
        self.steps     = 0
        self.goal      = False
        self.collision = False
        self.airborne  = False
        self.oob       = False
        self.timeout   = False
        self.stationary = False
        self.cells     = 0
        self.best_d    = start_dist
        self.final_d   = start_dist
        self.fitness   = 0.0


def evaluate(agent: RobotAgent, mlp: MLP, max_steps: int) -> EpResult:
    """Evaluate one genome in agent's maze."""
    m      = agent.maze
    res    = EpResult(start_dist=m.start_d)
    best_d = m.start_d
    cells  = {_cell(m.start_x, m.start_y)}
    lcx, lcy = m.start_x, m.start_y

    check_steps = 60
    last_check_x, last_check_y = m.start_x, m.start_y

    for step in range(max_steps):
        t0     = time.perf_counter()
        obs    = agent.observe()
        action = mlp.forward(obs)
        agent.act(action)

        x, y, z, _ = agent.pose
        s_min    = agent.scan_min
        res.steps = step + 1

        # Airborne detection
        if z > C.AIRBORNE_Z:
            res.airborne  = True
            res.collision = True
            break

        # Out of bounds
        if not (m.x_min <= x <= m.x_max and m.y_min <= y <= m.y_max):
            res.oob = True;  break

        # Lidar collision
        if s_min < C.ELIM_DIST:
            res.collision = True;  break

        # Goal check
        d = _dist(x, y, m.goal_x, m.goal_y)
        if d < C.GOAL_R:
            res.goal  = True
            best_d    = min(best_d, d)
            break

        if d < best_d:
            best_d = d

        # Cell tracking
        if _dist(x, y, lcx, lcy) > C.CELL_SZ / 2:
            cells.add(_cell(x, y))
            lcx, lcy = x, y

        # Stationary check (every 60 steps = 12s)
        if step > 0 and step % check_steps == 0:
            if _dist(x, y, last_check_x, last_check_y) < 0.15:
                res.stationary = True
                break
            last_check_x, last_check_y = x, y

        rem = C.STEP_DT - (time.perf_counter() - t0)
        if rem > 0: time.sleep(rem)

    agent.stop()
    x, y, z, _ = agent.pose
    res.cells   = len(cells)
    if res.cells < 3:
        res.stationary = True
    if not res.goal and not res.collision and not res.oob and not res.stationary:
        res.timeout = True
    res.best_d   = best_d
    res.final_d  = _dist(x, y, m.goal_x, m.goal_y)
    res.fitness  = _fitness(res, m.start_d)
    return res


def _fitness(r: EpResult, start_d: float = C.START_DIST) -> float:
    """
    Fitness v8: clean reward/penalty structure. Minimize J (more negative = better).
    J = Penalties − Rewards.
    """
    progress  = max(0, start_d - r.final_d) / start_d   # 0-1 normalized
    best_prog = max(0, start_d - r.best_d)  / start_d   # 0-1 normalized

    # Penalties
    P = (C.P_COLL * r.collision +
         C.P_OOB * r.oob +
         C.P_TIMEOUT * r.timeout +
         C.P_STATIONARY * r.stationary +
         C.TIME_COST * r.steps)

    # Rewards
    R = (C.R_PROGRESS * progress +
         C.R_BEST_PROG * best_prog +
         C.R_CELLS * r.cells)

    if r.goal:
        R += C.R_GOAL + max(0, C.R_SPEED - r.steps * 5)

    return float(P - R)


# GENETIC ALGORITHM

class GA:
    """
    Minimization GA: tournament selection, uniform crossover,
    Gaussian mutation, elitism, diversity injection on stagnation.
    """

    def __init__(self, pop: int = C.POP, seed: int = 42):
        self.rng     = np.random.default_rng(seed)
        self.pop     = [self.rng.normal(0.0, C.INIT_S, C.N_W) for _ in range(pop)]
        self.fits    = [float("inf")] * pop
        self.gen     = 0
        self.best_g  = None
        self.best_f  = float("inf")
        self._stag   = 0
        self.history = []

    def set_fits(self, fits: list):
        self.fits = list(fits)
        bi = int(np.argmin(fits))
        if fits[bi] < self.best_f:
            self.best_f = fits[bi]
            self.best_g = self.pop[bi].copy()
            self._stag  = 0
        else:
            self._stag += 1
            # ← REMOVED: best_f regression (was causing quality decay)

        self.history.append({"gen": self.gen, "best": float(np.min(fits)),
                             "mean": float(np.mean(fits)),
                             "std": float(np.std(fits)), "stag": self._stag})

    def evolve(self, max_gens: int = 500):
        """Evolve with strong exploration: scouts + full-random injection."""
        order    = np.argsort(self.fits)
        new_pop  = [self.pop[i].copy() for i in order[:C.ELITE]]
        inject   = self._stag >= C.STAG_LIM
        n_inj    = int((len(self.pop)-C.ELITE)*C.STAG_FRAC) if inject else 0
        n_scouts = int((len(self.pop)-C.ELITE)*C.SCOUT_FRAC)  # random explorers

        # Rank Selection Preparation
        N = len(self.pop)
        ranks = np.zeros(N)
        for rank, idx in enumerate(order[::-1]):
            ranks[idx] = rank + 1
        probs = ranks / ranks.sum()

        def _select():
            idx = self.rng.choice(N, p=probs)
            return self.pop[idx].copy()

        # Mutation: floor at useful levels, burst on stagnation
        decay  = max(0.30, 1.0 - (self.gen / max(1, max_gens)))
        burst  = 2.0 if (5 <= self._stag < C.STAG_LIM) else 1.0
        mut_std_eff = max(0.10, C.MUT_STD * decay * burst)
        mut_p_eff   = max(0.12, min(0.6, C.MUT_P * (1.0 + self._stag / C.STAG_LIM)))

        remaining = len(self.pop) - C.ELITE
        for k in range(remaining):
            if inject and k < n_inj:
                # Full random — NOT based on best_g (escape the trap)
                child = self.rng.normal(0.0, C.INIT_S, C.N_W)
            elif k >= remaining - n_scouts:
                # Scouts: fully random genomes every generation
                child = self.rng.normal(0.0, C.INIT_S, C.N_W)
            else:
                p1, p2 = _select(), _select()
                child  = self._cx(p1,p2) if self.rng.random()<C.CX_RATE else p1.copy()
                child  = self._mut(child, mut_p_eff, mut_std_eff)
            child = np.clip(child, -C.W_CLIP, C.W_CLIP)
            new_pop.append(child)

        if inject:
            print(f"    ⚡ Diversity inject: {n_inj} RANDOM genomes (stag={self._stag})")
            self._stag = 0

        self.pop  = new_pop
        self.fits = [float("inf")] * len(self.pop)
        self.gen += 1

    def _cx(self, a, b):
        m = self.rng.random(C.N_W) < 0.5
        return np.where(m, a, b)

    def _mut(self, g, mut_p: float, mut_std: float):
        """Gaussian mutation with adaptive parameters."""
        m = self.rng.random(C.N_W) < mut_p
        return np.clip(g + m*self.rng.normal(0, mut_std, C.N_W), -C.W_CLIP, C.W_CLIP)

    def save(self, path: str):
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, "w") as f:
            json.dump({"gen": self.gen, "best_f": float(self.best_f),
                       "best_g": self.best_g.tolist() if self.best_g is not None else None,
                       "pop": [g.tolist() for g in self.pop],
                       "fits": [float(x) for x in self.fits],
                       "history": self.history}, f, indent=2)
        print(f"  ✓ {path}  gen={self.gen} best={self.best_f:.1f}")

    def load(self, path: str):
        with open(path) as f: d = json.load(f)
        self.gen    = d["gen"]
        self.best_f = d["best_f"]
        self.best_g = np.array(d["best_g"]) if d["best_g"] else None
        self.pop    = [np.array(g) for g in d["pop"]]
        self.fits   = d.get("fits", [float("inf")]*len(self.pop))
        self.history= d.get("history", [])
        print(f"  ✓ Loaded {path}  gen={self.gen} best={self.best_f:.1f}")


# LOGGER

class Logger:
    HDR = ("gen,best_J,mean_J,std_J,goals,cols,oobs,timeouts,"
           "best_best_prog,best_final_prog,best_d,best_cells,best_steps,ep_time,stag\n")

    def __init__(self):
        os.makedirs(C.CKPT, exist_ok=True)
        if not os.path.isfile(C.CSV):
            _fwrite(C.CSV, self.HDR)
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        _fwrite(C.LOG, f"\n{'═'*60}\nSession {ts}\n"
                f"  MODE=parallel  POP={C.POP}  N_W={C.N_W}  STEPS={C.MAX_STEPS}\n"
                f"  Net: {C.N_IN}→{C.N_H1}→{C.N_H2}→{C.N_OUT}  Weights={C.N_W}\n"
                f"  ELIM={C.ELIM_DIST}m  AIRBORNE_Z={C.AIRBORNE_Z}m\n"
                f"  Fitness v8: P_COLL={C.P_COLL}  R_PROGRESS={C.R_PROGRESS}  R_GOAL={C.R_GOAL}\n"
                f"  EMA(α={C.EMA_ALPHA}) + vel clamp vx∈[{C.VX_MIN},{C.VX_MAX}]\n"
                f"{'═'*60}\n")

    def log_gen(self, gen: int, fits: list, results: list, ep_t: float, ga: GA):
        br   = results[int(np.argmin(fits))]
        g    = sum(r.goal for r in results)
        c    = sum(r.collision for r in results)
        o    = sum(r.oob for r in results)
        to   = len(results) - g - c - o
        ts   = datetime.now().strftime("%H:%M:%S")
        bp   = C.START_DIST - br.best_d
        fp   = C.START_DIST - br.final_d
        _fwrite(C.LOG,
            f"[{ts}] Gen {gen:4d} | {ep_t:.1f}s | "
            f"best={min(fits):.1f} mean={float(np.mean(fits)):.1f} "
            f"std={float(np.std(fits)):.1f} stag={ga._stag}\n"
            f"  outcomes: goal={g} col={c} oob={o} timeout={to}\n"
            f"  best_genome: best_prog={bp:+.3f}m  final_prog={fp:+.3f}m  "
            f"best_d={br.best_d:.3f}m  cells={br.cells}  "
            f"{'★GOAL★' if br.goal else ''}\n")
        _fwrite(C.CSV,
            f"{gen},{min(fits):.2f},{float(np.mean(fits)):.2f},"
            f"{float(np.std(fits)):.2f},{g},{c},{o},{to},"
            f"{bp:.3f},{fp:.3f},{br.best_d:.3f},{br.cells},{br.steps},"
            f"{ep_t:.2f},{ga._stag}\n")

    def log_event(self, msg: str):
        ts = datetime.now().strftime("%H:%M:%S")
        _fwrite(C.LOG, f"[{ts}] {msg}\n")
        print(f"  [LOG] {msg}")

    def log_end(self, best_f: float, sec: float):
        _fwrite(C.LOG,
            f"\nEnd: best={best_f:.2f}  time={sec/60:.1f}min\n{'─'*60}\n")


# TRAINER  (Sequential evaluation + GA loop)

class Trainer:
    """
    GA Training loop với hỗ trợ parallel evaluation trên N mazes.

    ── Kiến trúc Parallel Batch ──────────────────────────────────────
    Với N workers (mazes), mỗi generation chạy ceil(POP/N) batches.
    Trong mỗi batch: N genomes được đánh giá SONG SONG,
    mỗi genome trong maze riêng của nó → không lidar cross-talk.

    POP=20, N=5:  4 batches × 5 parallel  =  ~85s/gen  (5× nhanh hơn)
    POP=20, N=1:  20 batches × 1          = ~427s/gen  (tuần tự)
    POP=20, N=20: 1 batch  × 20 parallel  =  ~21s/gen (nhanh nhất)

    ── Điều kiện để parallel đúng ────────────────────────────────────
    1. Mỗi maze là bản copy giống hệt → fitness tương đương
    2. Khoảng cách mazes ≥ 15m → lidar không thấy maze kế
    3. Mỗi robot có bridge riêng → topic không overlap
    4. evaluate() thread-safe: không share mutable state

    ── Cách dùng ─────────────────────────────────────────────────────
    # 1 maze (sequential, bản cũ):
    python3 ga_nn_controller.py --workers 1

    # 5 mazes (parallel 5×):
    gz sim nhom8_multi5.sdf      # maze SDF từ gen_multi_maze.py
    ./spawn_robot.sh 5           # spawn robot_1..5
    python3 ga_nn_controller.py --workers 5

    # 20 mazes (parallel 20× — 1 batch = 1 gen):
    gz sim nhom8_multi20.sdf
    ./spawn_robot.sh 20
    python3 ga_nn_controller.py --workers 20
    """

    def __init__(self, n_gen: int, steps: int, ga: GA,
                 maze_configs: list):
        """
        maze_configs: list of MazeCfg, length = number of parallel workers.
        """
        self.n_gen   = n_gen
        self.steps   = steps
        self.ga      = ga
        self.mazes   = maze_configs   # N MazeCfg objects
        self.n_w     = len(maze_configs)
        self.agents  = []
        self.logger  = Logger()
        self._exec   = None

    def setup(self):
        """Create N RobotAgent nodes, start executor, park all robots."""
        rclpy.init()
        self.agents = [RobotAgent(m) for m in self.mazes]
        n_threads   = self.n_w * 2 + 4
        self._exec  = MultiThreadedExecutor(num_threads=n_threads)
        for ag in self.agents:
            self._exec.add_node(ag)
        threading.Thread(target=self._exec.spin, daemon=True,
                         name="ros2_spin").start()

        print(f"  Waiting for Gazebo bridge (3s)...", end="", flush=True)
        time.sleep(3.0)
        print(" OK")
        # Park all robots underground so initial scans are clean
        self._park_all()
        time.sleep(0.5)
        print(f"  {self.n_w} worker(s) ready [v2].")

    def train(self):
        pop_n = len(self.ga.pop)
        batches = -(-pop_n // self.n_w)   # ceil division
        speedup = pop_n / batches
        _banner(
            f"GA-RL Training  POP={pop_n}  GEN={self.n_gen}  WORKERS={self.n_w}",
            f"  Network: {C.N_IN}→{C.N_H1}→{C.N_H2}→{C.N_OUT} ({C.N_W} weights)",
            f"  Fitness v8: R_GOAL={C.R_GOAL}  R_PROGRESS={C.R_PROGRESS}  P_COLL={C.P_COLL}",
            f"  EMA(α={C.EMA_ALPHA}) + clamp vx∈[{C.VX_MIN},{C.VX_MAX}]",
            f"  Mutation: adaptive decay + stagnation burst",
            f"  Maze DY spacing: {self.mazes[1].start_y - self.mazes[0].start_y:.1f}m"
              if self.n_w > 1 else "  Single maze mode",
        )
        t0 = time.time()

        for gen in range(self.n_gen):
            gen_t   = time.time()
            fits    = [None] * pop_n
            results = [None] * pop_n

            print(f"\n  Gen {self.ga.gen:4d} │ {pop_n} genomes × {self.n_w} workers "
                  f"(continuous queue mode)")

            # ── Continuous Queue Evaluation ─────────────────────────
            q = queue.Queue()
            for i in range(pop_n):
                q.put(i)

            def worker_thread(slot_idx, agent):
                while True:
                    try:
                        genome_i = q.get_nowait()
                    except queue.Empty:
                        break
                    
                    # 1. Teleport to start position
                    self._teleport_agent(agent)
                    agent.reset_scan()
                    time.sleep(C.TELE_WAIT)
                    
                    # 2. Evaluate genome
                    mlp = MLP(self.ga.pop[genome_i])
                    r = evaluate(agent, mlp, self.steps)
                    
                    # 3. Disappear immediately to free space / prepare next evaluation
                    self._park_agent(agent)
                    
                    fits[genome_i] = r.fitness
                    results[genome_i] = r
                    
                    bp   = agent.maze.start_d - r.best_d
                    fp   = agent.maze.start_d - r.final_d
                    icon = "🏁" if r.goal else "✈️" if r.airborne \
                           else "💥" if r.collision \
                           else "🚪" if r.oob else "⏱"
                    tag  = 'AIRBORNE' if r.airborne else 'HIT WALL' if r.collision else ''
                    print(f"    [{genome_i+1:2d}/{pop_n}|w{slot_idx+1}] {icon} "
                          f"J={r.fitness:8.1f}  "
                          f"bp={bp:+.2f}m  fp={fp:+.2f}m  "
                          f"cells={r.cells:2d}  steps={r.steps:3d}  {tag}",
                          flush=True)
                    q.task_done()

            # Start worker threads
            threads = []
            for slot, ag in enumerate(self.agents):
                t = threading.Thread(target=worker_thread, args=(slot, ag), daemon=True)
                threads.append(t)
                t.start()

            # Wait for all genes in population to be evaluated
            q.join()
            for t in threads: t.join()
                
            ep_t = time.time() - gen_t
            self.ga.set_fits(fits)
            self.logger.log_gen(self.ga.gen, fits, results, ep_t, self.ga)

            best_r = results[int(np.argmin(fits))]
            m0     = self.mazes[0]
            bp     = m0.start_d - best_r.best_d
            fp     = m0.start_d - best_r.final_d
            bar    = _pbar(-min(fits), C.R_GOAL)
            print(f"\n  Gen {self.ga.gen:4d} │ {bar} │ "
                  f"best={min(fits):.1f}  mean={float(np.mean(fits)):.1f}  "
                  f"stag={self.ga._stag}  time={ep_t:.0f}s")
            print(f"  Gen {self.ga.gen:4d} │ best_genome: "
                  f"best_prog={bp:+.3f}m  final_prog={fp:+.3f}m  "
                  f"{'★ GOAL!' if best_r.goal else f'dist={best_r.final_d:.2f}m'}")

            if self.ga._stag >= C.STAG_LIM:
                self.logger.log_event(f"Diversity inject gen={self.ga.gen}")
            self.ga.evolve(max_gens=self.n_gen)

            if (gen+1) % 5 == 0:
                self.ga.save(f"{C.CKPT}/gen_{self.ga.gen:04d}.json")
            if self.ga.best_g is not None:
                self.ga.save(f"{C.CKPT}/best.json")
            self.ga.save(f"{C.CKPT}/last_run.json")

        self.ga.save(f"{C.CKPT}/final.json")
        self.ga.save(f"{C.CKPT}/last_run.json")
        total = time.time() - t0
        self.logger.log_end(self.ga.best_f, total)
        _banner(f"Done!  best_J={self.ga.best_f:.1f}  time={total/60:.1f}min",
                f"Best checkpoint: {C.CKPT}/best.json",
                f"CSV: {C.CSV}")

    def demo(self, genome: np.ndarray):
        """Run best genome in maze 0."""
        mlp = MLP(genome)
        ag  = self.agents[0]
        m   = self.mazes[0]
        print(f"\n  Demo: ({m.start_x:.2f},{m.start_y:.2f}) → ({m.goal_x:.2f},{m.goal_y:.2f})")
        self._teleport_agent(ag)
        ag.reset_scan()
        time.sleep(C.TELE_WAIT)
        r = evaluate(ag, mlp, max_steps=5000)
        print(f"  {'GOAL!' if r.goal else 'col' if r.collision else 'OOB' if r.oob else 'timeout'}"
              f"  steps={r.steps}  cells={r.cells}  dist={r.final_d:.2f}m  J={r.fitness:.1f}")

    # ── Teleport helpers ─────────────────────────────────────────

    def _teleport_agent(self, agent: RobotAgent, yaw: float = 0.0):
        """Teleport agent to its maze's START position."""
        m  = agent.maze
        qz = math.sin(yaw/2); qw = math.cos(yaw/2)
        req = (f"name: 'robot_{m.robot_id}' "
               f"position {{ x: {m.start_x} y: {m.start_y} z: 0.064 }} "
               f"orientation {{ x: 0.0 y: 0.0 z: {qz:.6f} w: {qw:.6f} }}")
        try:
            subprocess.run(
                ["gz","service","-s",f"/world/{C.WORLD}/set_pose",
                 "--reqtype","gz.msgs.Pose","--reptype","gz.msgs.Boolean",
                 "--req",req,"--timeout","1000"],
                capture_output=True, timeout=2.0)
        except (subprocess.TimeoutExpired, FileNotFoundError):
            pass

    def _park_agent(self, agent: RobotAgent):
        """Park agent underground (scan clean for next eval)."""
        m  = agent.maze
        req = (f"name: 'robot_{m.robot_id}' "
               f"position {{ x: {m.start_x} y: {m.start_y} z: {C.PARK_Z} }} "
               f"orientation {{ x: 0.0 y: 0.0 z: 0.0 w: 1.0 }}")
        try:
            subprocess.run(
                ["gz","service","-s",f"/world/{C.WORLD}/set_pose",
                 "--reqtype","gz.msgs.Pose","--reptype","gz.msgs.Boolean",
                 "--req",req,"--timeout","1000"],
                capture_output=True, timeout=2.0)
        except (subprocess.TimeoutExpired, FileNotFoundError):
            pass

    def _park_all(self):
        """Park all workers simultaneously."""
        threads = [threading.Thread(target=self._park_agent, args=(ag,), daemon=True)
                   for ag in self.agents]
        for t in threads: t.start()
        for t in threads: t.join()
        time.sleep(0.1)

    def shutdown(self):
        for ag in self.agents:
            try: ag.stop()
            except: pass
        try: self._exec and self._exec.shutdown()
        except: pass
        for ag in self.agents:
            try: ag.destroy_node()
            except: pass
        try: rclpy.shutdown()
        except: pass


# UTILITIES

def _dist(x1,y1,x2,y2): return math.sqrt((x1-x2)**2+(y1-y2)**2)
def _cell(x,y): return (int(x/C.CELL_SZ), int(y/C.CELL_SZ))
def _wrap(a):
    while a > math.pi:  a -= 2*math.pi
    while a < -math.pi: a += 2*math.pi
    return a
def _fwrite(path, text):
    with open(path, "a") as f: f.write(text)
def _pbar(val, maxv, w=20):
    p = min(1.0, max(0.0, val/maxv))
    n = int(w*p)
    return "█"*n + "░"*(w-n)
def _banner(*lines):
    w = max(len(l) for l in lines) + 4
    print(f"\n  ╔{'═'*w}╗")
    for l in lines: print(f"  ║  {l:<{w-2}}║")
    print(f"  ╚{'═'*w}╝\n")


# MAIN

def main():
    ap = argparse.ArgumentParser(
        description="GA-RL Mecanum Maze — Parallel multi-maze evaluation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Sequential (1 maze, bản cũ):
  python3 ga_nn_controller.py --pop 20 --gen 500 --workers 1

  # Parallel 5× (cần nhom8_multi5.sdf + ./spawn_robot.sh 5):
  python3 ga_nn_controller.py --pop 20 --gen 500 --workers 5

  # Tiếp tục từ checkpoint:
  python3 ga_nn_controller.py --load ga_checkpoints/best.json --workers 5

  # Demo best genome:
  python3 ga_nn_controller.py --run  ga_checkpoints/best.json --workers 1

  # Tạo multi-maze SDF:
  python3 gen_multi_maze.py nhom8_maze.sdf --n 5 --out nhom8_multi5.sdf
""")
    ap.add_argument("--pop",     type=int, default=C.POP,      help="Population size")
    ap.add_argument("--gen",     type=int, default=200,         help="Generations")
    ap.add_argument("--steps",   type=int, default=C.MAX_STEPS, help="Steps per genome")
    ap.add_argument("--workers", type=int, default=1,
                    help="Number of parallel mazes/workers (default 1 = sequential)")
    ap.add_argument("--load",  type=str, default=None,  help="Resume from checkpoint")
    ap.add_argument("--fresh", action="store_true",     help="Start fresh (do not auto-resume from last_run.json)")
    ap.add_argument("--run",   type=str, default=None,  help="Demo best genome")
    ap.add_argument("--seed",  type=int, default=42)
    ap.add_argument("--info",  action="store_true",     help="Print architecture and exit")
    args = ap.parse_args()

    if args.info:
        _banner(
            f"Network: {C.N_IN}→{C.N_H1}(tanh)→{C.N_H2}(tanh)→{C.N_OUT}(clip)",
            f"Weights: {C.N_W}",
            "Inputs [0:12]  12 lidar rays ∈ [0,1] (every 30°)",
            "       [12]    goal angle in robot frame / π",
            "       [13]    normalized distance to goal",
            "       [14:16] heading: sin_yaw, cos_yaw",
            f"GA: POP={C.POP} ELITE={C.ELITE} MUT_P={C.MUT_P}(adaptive) MUT_STD={C.MUT_STD}(decay)",
            "Fitness v8: J = Penalties − Rewards (minimize)",
            f"  P_COLL={C.P_COLL}  P_OOB={C.P_OOB}  P_TIMEOUT={C.P_TIMEOUT}",
            f"  P_STATIONARY={C.P_STATIONARY}  TIME_COST={C.TIME_COST}/step",
            f"  R_PROGRESS={C.R_PROGRESS}  R_BEST_PROG={C.R_BEST_PROG}  R_GOAL={C.R_GOAL}",
            f"  R_SPEED={C.R_SPEED}  R_CELLS={C.R_CELLS}",
            f"Velocity: vx∈[{C.VX_MIN},{C.VX_MAX}]  EMA(α={C.EMA_ALPHA})",
            f"Scouts: {C.SCOUT_FRAC*100:.0f}% random/gen  Inject: {C.STAG_FRAC*100:.0f}% random@stag{C.STAG_LIM}",
        )
        return

    # ── Build maze configs ────────────────────────────────────────
    maze_configs = _default_maze_configs(args.workers)
    print(f"\n  Workers: {args.workers}")
    for m in maze_configs:
        print(f"    robot_{m.robot_id}: START({m.start_x:.2f},{m.start_y:.2f}) "
              f"GOAL({m.goal_x:.2f},{m.goal_y:.2f})  "
              f"bounds x∈[{m.x_min:.2f},{m.x_max:.2f}]")

    ga = GA(pop=args.pop, seed=args.seed)
    if args.load:
        ga.load(args.load)
    elif not args.fresh and os.path.exists(f"{C.CKPT}/last_run.json"):
        print(f"  Auto-resuming from {C.CKPT}/last_run.json")
        ga.load(f"{C.CKPT}/last_run.json")

    trainer = Trainer(n_gen=args.gen, steps=args.steps, ga=ga,
                      maze_configs=maze_configs)

    if args.run:
        if ga.best_g is None:
            print("  No best genome in checkpoint."); return
        try:
            trainer.setup()
            trainer.demo(ga.best_g)
        finally:
            trainer.shutdown()
        return

    try:
        trainer.setup()
        trainer.train()
    except KeyboardInterrupt:
        print("\n  Interrupted — saving...")
        ga.save(f"{C.CKPT}/last_run.json")
        ga.save(f"{C.CKPT}/interrupted_gen{ga.gen}.json")
    except Exception as e:
        print(f"\n  ERROR: {type(e).__name__}: {e}")
        import traceback; traceback.print_exc()
        ga.save(f"{C.CKPT}/error_gen{ga.gen}.json")
    finally:
        trainer.shutdown()


if __name__ == "__main__":
    main()
    # args = ap.parse_args()