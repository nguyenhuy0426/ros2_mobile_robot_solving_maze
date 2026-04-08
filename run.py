#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ga_nn_controller.py  ─  GA-RL Mecanum Maze Controller  v4
══════════════════════════════════════════════════════════

KIẾN TRÚC: Sequential Single-Agent Evaluation
═══════════════════════════════════════════════
VẤN ĐỀ "20 robots cùng lúc":
  Lidar robot_i quét thấy chassis của robot_j (cùng vị trí)
  → ranges < 0.05m → collision flag ngay step 0 → tất cả eliminate.

GIẢI PHÁP: Đánh giá tuần tự từng cá thể với 1 robot duy nhất
  for i in range(POP):
      teleport robot_1 → START
      r = run_episode(genome[i])   # đo fitness thực sự
      fitness[i] = r.J
  ga.evolve()

  → Không có lidar cross-contamination
  → Fitness measure chính xác
  → Đây là cách NEATRobot, OpenAI Gym, robotics RL research đều làm

Cách chạy:
  gz sim nhom8_maze.sdf                              # B1: mở Gazebo
  ./spawn_robot.sh 1                                 # B2: spawn 1 robot
  python3 ga_nn_controller.py --pop 20 --gen 500     # B3: train
  python3 ga_nn_controller.py --load best.json       # B4: tiếp tục
  python3 ga_nn_controller.py --run  best.json       # B5: demo
  python3 ga_nn_controller.py --info                 # xem kiến trúc
"""

import argparse
import json
import math
import os
import subprocess
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


# ══════════════════════════════════════════════════════════════════
# CONFIG  — tất cả hằng số ở đây
# ══════════════════════════════════════════════════════════════════

class Cfg:
    # Gazebo
    WORLD    = "nhom8_mecanum"
    ROBOT_ID = 1               # chỉ dùng 1 robot duy nhất

    # Mê cung (từ nhom8_maze.sdf)
    START_X, START_Y = 2.25, -3.25
    START_Z          = 0.064   # chassis z off ground
    START_YAW        = 0.0     # 0 rad = facing +x into maze
    GOAL_X,  GOAL_Y  = 4.25, -3.25
    GOAL_R           = 0.25    # goal radius (m)
    MAZE_X_MIN, MAZE_X_MAX = 1.85, 4.65
    MAZE_Y_MIN, MAZE_Y_MAX = -4.65, -1.85
    MAZE_CX, MAZE_CY        = 3.25, -3.25  # center of maze bounding box

    # Robot (one_robot.sdf)
    WHEEL_R = 0.024   # m
    LX, LY  = 0.08, 0.09
    MAX_W   = 8.0     # rad/s → vx_max = 0.192 m/s (safe for 50cm corridor)

    # Lidar (36 rays × 10°, range 0–12m)
    N_RAYS   = 36
    LIDAR_MAX = 12.0
    OBS_IDX  = [0, 4, 9, 13, 18, 22, 27, 31]   # 8 rays ~45° apart for observation
    FRONT_IDX = [34, 35, 0, 1, 2]               # ±20° front only — NOT side walls
    ELIM_DIST = 0.10    # lidar < this → collision → episode ends
    WALL_STOP = 0.15    # emergency: stop + rotate
    WALL_SLOW = 0.22    # warning: slow down vx
    ESCAPE_WZ = 1.2     # rad/s rotation when escaping wall

    # Neural Network: 16 → 32(tanh) → 16(tanh) → 4(clip)
    N_IN  = 16    # [lidar×8, dx_goal, dy_goal, dist, collision, sin_yaw, cos_yaw, x_rel, y_rel]
    N_H1  = 32
    N_H2  = 16
    N_OUT = 4     # [w_fl, w_fr, w_rl, w_rr] ∈ [-1,1]
    N_W   = N_IN*N_H1 + N_H1 + N_H1*N_H2 + N_H2 + N_H2*N_OUT + N_OUT  # 1140

    # Genetic Algorithm
    POP      = 20
    ELITE    = 2
    TOURN_K  = 3
    CX_RATE  = 0.70
    MUT_P    = 0.20    # probability per gene
    MUT_STD  = 0.40    # gaussian std
    W_CLIP   = 6.0
    INIT_S   = 0.6
    STAG_LIM  = 10     # gens without improvement before diversity injection
    STAG_FRAC = 0.40   # fraction of non-elite to randomize on injection

    # Episode
    MAX_STEPS = 200     # max steps per individual evaluation
    STEP_DT   = 0.10    # s (10 Hz)
    TELE_WAIT = 1.2     # s after teleport before episode starts
    PARK_Z    = -5.0    # underground z for parking between evaluations

    # ── Fitness v5: distance-first, zero noise ───────────────────────
    #
    # J = P_COLL×col + P_OOB×oob
    #   − R_BEST ×(START_D − best_d)    ← closest moment to goal
    #   − R_FINAL×(START_D − final_d)   ← final resting position
    #   − R_GOAL ×goal
    #
    # Properties:
    #   still robot      → J = 0       (neutral, moving robots dominate)
    #   backward robot   → J > 0       (penalized via negative progress terms)
    #   advances 1m      → J = −230    (strongly rewarded)
    #   advances + coll  → J = 80−230  (still rewarded for best_d)
    #   reaches goal     → J ≈ −2560   (dominant signal)
    # ──────────────────────────────────────────────────────────────────
    START_DIST = 2.0    # dist(START→GOAL) = 2.0m (computed manually)
    P_COLL  = 80.0
    P_OOB   = 120.0
    R_BEST  = 150.0   # reward per meter at closest point
    R_FINAL = 80.0    # reward per meter at final position
    R_GOAL  = 2000.0
    CELL_SZ = 0.25    # for logging only, not in fitness

    # Logging
    LOG  = "ga_training.log"
    CSV  = "ga_stats.csv"
    CKPT = "ga_checkpoints"


C = Cfg()


# ══════════════════════════════════════════════════════════════════
# NEURAL NETWORK  (NumPy only)
# ══════════════════════════════════════════════════════════════════

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
        vx = max(0.0, r/4*(fl+fr+rl+rr))   # no reverse
        vy = r/4*(-fl+fr+rl-rr)
        wz = r/(4*(lx+ly))*(-fl+fr-rl+rr)
        return vx, vy, wz


# ══════════════════════════════════════════════════════════════════
# ROBOT AGENT  (ROS2 I/O only — no fitness logic here)
# ══════════════════════════════════════════════════════════════════

class RobotAgent(Node):
    """
    Minimal ROS2 node: subscribe /scanN + /model/robot_N/pose,
    publish /model/robot_N/cmd_vel.
    Thread-safe via _lock.
    """

    def __init__(self):
        super().__init__("ga_agent")
        self._lock = threading.Lock()
        self._scan = [C.LIDAR_MAX] * C.N_RAYS
        self._x, self._y, self._yaw = C.START_X, C.START_Y, C.START_YAW

        qos = rclpy.qos.QoSProfile(
            reliability=rclpy.qos.ReliabilityPolicy.BEST_EFFORT,
            history=rclpy.qos.HistoryPolicy.KEEP_LAST, depth=1)

        self.create_subscription(
            LaserScan, f"/scan{C.ROBOT_ID}", self._cb_scan, qos)
        self.create_subscription(
            Pose, f"/model/robot_{C.ROBOT_ID}/pose", self._cb_pose, qos)
        self._pub = self.create_publisher(
            Twist, f"/model/robot_{C.ROBOT_ID}/cmd_vel", 10)
        self.get_logger().info("RobotAgent ready (single robot mode).")

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
            self._yaw = 2.0 * math.atan2(msg.orientation.z, msg.orientation.w)

    def observe(self) -> np.ndarray:
        """
        16-dim observation vector:
          [0:8]  lidar ∈ [0,1]        (8 rays ~45° apart)
          [8:10] goal relative dir     (dx/4, dy/4)
          [10]   dist to goal / 8
          [11]   collision flag        (1 if lidar_min < ELIM_DIST)
          [12:14] heading              (sin_yaw, cos_yaw)
          [14:16] position vs center   (x_rel/2, y_rel/2)
        """
        with self._lock:
            scan = list(self._scan)
            x, y, yaw = self._x, self._y, self._yaw

        lidar = np.array(
            [min(scan[i], C.LIDAR_MAX)/C.LIDAR_MAX for i in C.OBS_IDX],
            dtype=np.float32)
        dist  = _dist(x, y, C.GOAL_X, C.GOAL_Y)
        return np.array([
            *lidar,
            (C.GOAL_X - x) / 4.0,
            (C.GOAL_Y - y) / 4.0,
            min(dist, 8.0) / 8.0,
            1.0 if min(scan) < C.ELIM_DIST else 0.0,
            math.sin(yaw), math.cos(yaw),
            (x - C.MAZE_CX) / 2.0,
            (y - C.MAZE_CY) / 2.0,
        ], dtype=np.float32)

    def act(self, action: np.ndarray):
        """
        Publish action with SOFT wall enforcement.

        ── Lý do KHÔNG dùng hard escape rotation ──────────────────
        Code cũ: khi gần tường → ghi đè wz = ±ESCAPE_WZ (bỏ qua NN)
        Hậu quả: NN không bao giờ học cách quay (bị override liên tục)
                 Robot lắc qua lại: turn → clear wall → forward → hit wall → turn...

        Thiết kế mới:
          • wz LUÔN đến từ NN (NN tự học quay tránh tường)
          • Chỉ giảm/dừng vx khi tường ở phía trước
          • vy cũng bị dừng khi rất gần tường (mọi hướng)
          Collision penalty trong fitness dạy NN tránh tường.
        """
        with self._lock:
            scan = list(self._scan)

        min_all   = min(scan)
        min_front = min(scan[i] for i in C.FRONT_IDX)

        vx, vy, wz = MLP.to_twist(action)

        if min_all < C.WALL_STOP:
            # Rất gần tường bất kỳ hướng: dừng vx+vy, giữ wz của NN
            # NN tự xoay thoát — đây là kỹ năng NN cần học
            self._pub_twist(0.0, 0.0, wz)
            return

        if min_front < C.WALL_SLOW:
            # Tiếp cận tường phía trước: scale vx xuống mượt
            t   = (min_front - C.WALL_STOP) / (C.WALL_SLOW - C.WALL_STOP)
            vx *= max(0.0, t ** 1.5)   # power 1.5 = smooth braking

        self._pub_twist(vx, vy, wz)

    def stop(self):
        self._pub_twist(0.0, 0.0, 0.0)

    def _pub_twist(self, vx, vy, wz):
        msg = Twist()
        msg.linear.x, msg.linear.y, msg.angular.z = float(vx), float(vy), float(wz)
        self._pub.publish(msg)

    @property
    def pose(self):
        with self._lock: return self._x, self._y, self._yaw

    @property
    def scan_min(self):
        with self._lock: return min(self._scan)

    def reset_scan(self):
        """Clear stale scan from underground parking (PARK_Z=-5m)."""
        with self._lock:
            self._scan = [C.LIDAR_MAX] * C.N_RAYS


# ══════════════════════════════════════════════════════════════════
# EPISODE  — evaluate one genome, return structured result
# ══════════════════════════════════════════════════════════════════

class EpResult:
    """Metrics from one genome evaluation."""
    def __init__(self):
        self.steps     = 0
        self.goal      = False
        self.collision = False
        self.oob       = False
        self.cells     = 0        # unique cells (for logging only)
        self.best_d    = C.START_DIST   # minimum dist to goal seen (best moment)
        self.final_d   = C.START_DIST   # dist at episode end
        self.fitness   = 0.0


def evaluate(agent: RobotAgent, mlp: MLP, max_steps: int) -> EpResult:
    """
    Evaluate one genome: teleport already done by Trainer.

    Key design decisions:
    1. Track best_d (minimum distance ever) — rewards robot even if it
       collides AFTER getting close to goal.
    2. Cell counting only for LOGGING, not fitness.
    3. No yaw spread — always START_YAW (user request).
    4. wz comes entirely from NN (wall enforcement doesn't override wz).
    """
    res    = EpResult()
    best_d = C.START_DIST            # track closest distance achieved
    cells  = {_cell(C.START_X, C.START_Y)}
    # Track actual physical cells (require movement > CELL_SZ/2 to count new cell)
    last_cell_x, last_cell_y = C.START_X, C.START_Y

    for step in range(max_steps):
        t0     = time.perf_counter()
        obs    = agent.observe()
        action = mlp.forward(obs)
        agent.act(action)

        x, y, yaw = agent.pose
        s_min     = agent.scan_min
        res.steps  = step + 1

        # ── Terminate ──────────────────────────────────────────
        if not (C.MAZE_X_MIN <= x <= C.MAZE_X_MAX and
                C.MAZE_Y_MIN <= y <= C.MAZE_Y_MAX):
            res.oob = True
            break

        if s_min < C.ELIM_DIST:
            res.collision = True
            break

        d = _dist(x, y, C.GOAL_X, C.GOAL_Y)
        if d < C.GOAL_R:
            res.goal = True
            best_d = min(best_d, d)
            break

        # ── Track best distance ────────────────────────────────
        if d < best_d:
            best_d = d

        # ── Cell counting (LOG only) — require real movement ────
        # Threshold: only count new cell if moved > CELL_SZ/2 from last cell
        if _dist(x, y, last_cell_x, last_cell_y) > C.CELL_SZ / 2:
            cells.add(_cell(x, y))
            last_cell_x, last_cell_y = x, y

        rem = C.STEP_DT - (time.perf_counter() - t0)
        if rem > 0:
            time.sleep(rem)

    agent.stop()
    x, y, _ = agent.pose
    res.cells   = len(cells)
    res.best_d  = best_d
    res.final_d = _dist(x, y, C.GOAL_X, C.GOAL_Y)
    res.fitness = _fitness(res)
    return res


def _fitness(r: EpResult) -> float:
    """
    J = penalties − rewards   (minimize J, more negative = better robot)

    Thiết kế hoàn toàn dựa trên KHOẢNG CÁCH:
      best_prog  = START_D − best_d   ∈ [0, 2.0]  (robot đến gần nhất)
      final_prog = START_D − final_d  ∈ (-∞, 2.0] (âm nếu đi lùi)

    Tại sao dùng cả best_prog và final_prog?
      • best_prog:  robot bị collision sau khi tiến xa vẫn được reward
                    → GA có tín hiệu học cả kỹ năng "đi tới"
      • final_prog: robot cần DỪ LẠI GẦN GOAL, không chỉ đi qua rồi lùi
                    → áp lực duy trì vị trí tốt
    """
    best_prog  = C.START_DIST - r.best_d    # luôn ≥ 0
    final_prog = C.START_DIST - r.final_d   # có thể âm (đi lùi)

    P = C.P_COLL * r.collision + C.P_OOB * r.oob
    R = C.R_BEST * best_prog + C.R_FINAL * final_prog + C.R_GOAL * r.goal
    return float(P - R)


# ══════════════════════════════════════════════════════════════════
# GENETIC ALGORITHM
# ══════════════════════════════════════════════════════════════════

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
        self.history.append({"gen": self.gen, "best": float(np.min(fits)),
                             "mean": float(np.mean(fits)),
                             "std": float(np.std(fits)), "stag": self._stag})

    def evolve(self):
        order    = np.argsort(self.fits)
        new_pop  = [self.pop[i].copy() for i in order[:C.ELITE]]
        inject   = self._stag >= C.STAG_LIM
        n_inj    = int((len(self.pop)-C.ELITE)*C.STAG_FRAC) if inject else 0

        for k in range(len(self.pop)-C.ELITE):
            if inject and k < n_inj:
                base  = self.best_g if self.best_g is not None \
                        else self.rng.normal(0.0, C.INIT_S, C.N_W)
                child = np.clip(base + self.rng.normal(0.0, C.INIT_S*0.6, C.N_W),
                                -C.W_CLIP, C.W_CLIP)
            else:
                p1, p2 = self._tour(), self._tour()
                child  = self._cx(p1,p2) if self.rng.random()<C.CX_RATE else p1.copy()
                child  = self._mut(child)
            new_pop.append(child)

        if inject:
            print(f"    ⚡ Diversity inject: {n_inj} reset (stag={self._stag})")
            self._stag = 0

        self.pop  = new_pop
        self.fits = [float("inf")] * len(self.pop)
        self.gen += 1

    def _tour(self):
        idx = self.rng.choice(len(self.pop), size=C.TOURN_K, replace=False)
        return self.pop[idx[int(np.argmin([self.fits[i] for i in idx]))]].copy()

    def _cx(self, a, b):
        m = self.rng.random(C.N_W) < 0.5
        return np.where(m, a, b)

    def _mut(self, g):
        m = self.rng.random(C.N_W) < C.MUT_P
        return np.clip(g + m*self.rng.normal(0, C.MUT_STD, C.N_W), -C.W_CLIP, C.W_CLIP)

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


# ══════════════════════════════════════════════════════════════════
# LOGGER
# ══════════════════════════════════════════════════════════════════

class Logger:
    HDR = ("gen,best_J,mean_J,std_J,goals,cols,oobs,timeouts,"
           "best_best_prog,best_final_prog,best_d,best_cells,best_steps,ep_time,stag\n")

    def __init__(self):
        os.makedirs(C.CKPT, exist_ok=True)
        if not os.path.isfile(C.CSV):
            _fwrite(C.CSV, self.HDR)
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        _fwrite(C.LOG, f"\n{'═'*60}\nSession {ts}\n"
                f"  MODE=sequential  POP={C.POP}  N_W={C.N_W}  STEPS={C.MAX_STEPS}\n"
                f"  ELIM={C.ELIM_DIST}m  STOP={C.WALL_STOP}m  SLOW={C.WALL_SLOW}m\n"
                f"  Fitness v5: R_BEST={C.R_BEST}  R_FINAL={C.R_FINAL}  R_GOAL={C.R_GOAL}\n"
                f"  Wall: only vx suppressed, wz always from NN\n"
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


# ══════════════════════════════════════════════════════════════════
# TRAINER  (Sequential evaluation + GA loop)
# ══════════════════════════════════════════════════════════════════

class Trainer:
    """
    Main training loop — sequential single-agent evaluation.

    Per generation:
      for i in range(POP):
          teleport robot to START (with spread yaw)
          run_episode → EpResult → fitness[i]
          park robot underground (clean scan for next eval)
      GA.set_fits() → GA.evolve()
      checkpoint every 5 gens
    """

    def __init__(self, n_gen: int, steps: int, ga: GA):
        self.n_gen  = n_gen
        self.steps  = steps
        self.ga     = ga
        self.agent  = None
        self.logger = Logger()
        self._exec  = None

    def setup(self):
        rclpy.init()
        self.agent = RobotAgent()
        self._exec = MultiThreadedExecutor(num_threads=4)
        self._exec.add_node(self.agent)
        threading.Thread(target=self._exec.spin, daemon=True).start()
        print("  Waiting for Gazebo bridge (3s)...", end="", flush=True)
        time.sleep(3.0)
        print(" OK")
        # Park first so initial scan is clean (no ground reflections at z=-5)
        self._park()
        time.sleep(0.5)
        print("  Initial park complete — scan cleared.")

    def train(self):
        pop_n = len(self.ga.pop)
        _banner(
            f"GA-RL v5 Sequential  POP={pop_n}  GEN={self.n_gen}  STEPS={self.steps}",
            f"  Net: {C.N_IN}→{C.N_H1}→{C.N_H2}→{C.N_OUT}  Weights={C.N_W}",
            f"  Fitness: pure distance  R_BEST={C.R_BEST}  R_FINAL={C.R_FINAL}  R_GOAL={C.R_GOAL}",
            f"  Wall: only vx suppressed, wz ALWAYS from NN (no forced escape rotation)",
            f"  Yaw: fixed START_YAW={C.START_YAW:.2f} (straight into maze)",
            f"  START ({C.START_X},{C.START_Y}) → GOAL ({C.GOAL_X},{C.GOAL_Y})  dist={C.START_DIST}m",
        )

        t0 = time.time()

        for gen in range(self.n_gen):
            gen_t   = time.time()
            fits    = []
            results = []

            print(f"\n  Gen {self.ga.gen:4d} │ Evaluating {pop_n} genomes...")

            for idx in range(pop_n):
                mlp = MLP(self.ga.pop[idx])

                # Always use START_YAW — straight into maze, consistent evaluation
                self._teleport(C.START_X, C.START_Y, C.START_Z, C.START_YAW)
                self.agent.reset_scan()
                time.sleep(C.TELE_WAIT)

                r = evaluate(self.agent, mlp, self.steps)
                fits.append(r.fitness)
                results.append(r)

                best_prog  = C.START_DIST - r.best_d
                final_prog = C.START_DIST - r.final_d
                icon = "🏁" if r.goal else "💥" if r.collision else "🚪" if r.oob else "⏱"
                print(f"    [{idx+1:2d}/{pop_n}] {icon} "
                      f"J={r.fitness:8.1f}  "
                      f"best_prog={best_prog:+.2f}m  final_prog={final_prog:+.2f}m  "
                      f"cells={r.cells:3d}  steps={r.steps:3d}", flush=True)

                self._park()

            ep_t = time.time() - gen_t
            self.ga.set_fits(fits)
            self.logger.log_gen(self.ga.gen, fits, results, ep_t, self.ga)

            best_r = results[int(np.argmin(fits))]
            bar    = _pbar(-min(fits), C.R_GOAL)
            bp = C.START_DIST - best_r.best_d
            fp = C.START_DIST - best_r.final_d
            print(f"\n  Gen {self.ga.gen:4d} │ {bar} │ "
                  f"best={min(fits):.1f}  mean={float(np.mean(fits)):.1f}  "
                  f"stag={self.ga._stag}")
            print(f"  Gen {self.ga.gen:4d} │ best_genome: "
                  f"best_prog={bp:+.3f}m  final_prog={fp:+.3f}m  "
                  f"cells={best_r.cells}  "
                  f"{'★ GOAL!' if best_r.goal else f'dist={best_r.final_d:.2f}m'}")

            if self.ga._stag >= C.STAG_LIM:
                self.logger.log_event(f"Diversity inject at gen {self.ga.gen}")
            self.ga.evolve()

            if (gen+1) % 5 == 0:
                self.ga.save(f"{C.CKPT}/gen_{self.ga.gen:04d}.json")
            if self.ga.best_g is not None:
                self.ga.save(f"{C.CKPT}/best.json")

        self.ga.save(f"{C.CKPT}/final.json")
        total = time.time() - t0
        self.logger.log_end(self.ga.best_f, total)
        _banner(f"Done! best_J={self.ga.best_f:.1f}  time={total/60:.1f}min",
                f"Best: {C.CKPT}/best.json  |  CSV: {C.CSV}")

    def demo(self, genome: np.ndarray):
        """Run best genome once (no termination on collision — let it run)."""
        mlp = MLP(genome)
        print(f"\n  Demo: ({C.START_X},{C.START_Y}) → ({C.GOAL_X},{C.GOAL_Y})")
        self._teleport(C.START_X, C.START_Y, C.START_Z, C.START_YAW)
        self.agent.reset_scan()
        time.sleep(C.TELE_WAIT)
        r = evaluate(self.agent, mlp, max_steps=5000)
        print(f"  {'GOAL!' if r.goal else 'collision' if r.collision else 'OOB' if r.oob else 'timeout'}"
              f"  steps={r.steps}  cells={r.cells}  dist={r.final_d:.2f}m  J={r.fitness:.1f}")

    def _teleport(self, x, y, z, yaw):
        qz, qw = math.sin(yaw/2), math.cos(yaw/2)
        req = (f"name: 'robot_{C.ROBOT_ID}' "
               f"position {{ x: {x} y: {y} z: {z} }} "
               f"orientation {{ x: 0.0 y: 0.0 z: {qz:.6f} w: {qw:.6f} }}")
        try:
            subprocess.run(
                ["gz","service","-s",f"/world/{C.WORLD}/set_pose",
                 "--reqtype","gz.msgs.Pose","--reptype","gz.msgs.Boolean",
                 "--req",req,"--timeout","1000"],
                capture_output=True, timeout=2.0)
        except (subprocess.TimeoutExpired, FileNotFoundError):
            pass

    def _park(self):
        self._teleport(C.START_X, C.START_Y, C.PARK_Z, C.START_YAW)
        time.sleep(0.15)

    def shutdown(self):
        try: self.agent and self.agent.stop()
        except: pass
        try: self._exec and self._exec.shutdown()
        except: pass
        try: self.agent and self.agent.destroy_node()
        except: pass
        try: rclpy.shutdown()
        except: pass


# ══════════════════════════════════════════════════════════════════
# UTILITIES
# ══════════════════════════════════════════════════════════════════

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


# ══════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════

def main():
    ap = argparse.ArgumentParser(
        description="GA-RL Mecanum Maze — Sequential 1-robot evaluation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python3 ga_nn_controller.py --pop 20 --gen 500 --steps 200
  python3 ga_nn_controller.py --load ga_checkpoints/best.json --gen 200
  python3 ga_nn_controller.py --run  ga_checkpoints/best.json
  python3 ga_nn_controller.py --info
""")
    ap.add_argument("--pop",   type=int, default=C.POP,       help="Population size (default 20)")
    ap.add_argument("--gen",   type=int, default=200,          help="Generations (default 200)")
    ap.add_argument("--steps", type=int, default=C.MAX_STEPS,  help="Steps per genome (default 200)")
    ap.add_argument("--load",  type=str, default=None,         help="Resume from checkpoint JSON")
    ap.add_argument("--run",   type=str, default=None,         help="Demo best genome from checkpoint")
    ap.add_argument("--seed",  type=int, default=42,           help="Random seed")
    ap.add_argument("--info",  action="store_true",            help="Print architecture and exit")
    args = ap.parse_args()

    if args.info:
        _banner(
            f"Network: {C.N_IN} → {C.N_H1}(tanh) → {C.N_H2}(tanh) → {C.N_OUT}(clip)",
            f"Total weights: {C.N_W}",
            "Inputs:",
            "  [0:8]   8 lidar rays normalized to [0,1]",
            "  [8:10]  goal direction (dx/4, dy/4)",
            "  [10]    distance to goal / 8.0",
            "  [11]    collision flag (1 = lidar_min < ELIM_DIST)",
            "  [12:14] heading (sin_yaw, cos_yaw) — robot orientation",
            "  [14:16] position relative to maze center / 2.0",
            f"GA: POP={C.POP}  ELITE={C.ELITE}  MUT_P={C.MUT_P}  MUT_STD={C.MUT_STD}",
            f"    stagnation inject after {C.STAG_LIM} gens, reset {C.STAG_FRAC*100:.0f}% of non-elite",
            "Fitness (minimize J = penalties − rewards):",
            f"  Penalties: collision×{C.P_COLL}  OOB×{C.P_OOB}  still×{C.P_STILL}/step",
            f"  Rewards:   cells×{C.R_CELL}  dist×{C.R_DIST}  angle×{C.R_ANGLE}/step  goal×{C.R_GOAL}",
        )
        return

    ga = GA(pop=args.pop, seed=args.seed)
    if args.load:
        ga.load(args.load)

    trainer = Trainer(n_gen=args.gen, steps=args.steps, ga=ga)

    if args.run:
        if ga.best_g is None:
            print("  No best genome in checkpoint.")
            return
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
        print("\n  Interrupted — saving checkpoint...")
        ga.save(f"{C.CKPT}/interrupted_gen{ga.gen}.json")
    except Exception as e:
        print(f"\n  ERROR: {type(e).__name__}: {e}")
        import traceback; traceback.print_exc()
        ga.save(f"{C.CKPT}/error_gen{ga.gen}.json")
    finally:
        trainer.shutdown()


if __name__ == "__main__":
    main()