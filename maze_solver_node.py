"""
ga_maze_solver.py
=================
Thuật toán Di Truyền (GA) giải mê cung với robot mecanum 4 bánh.

Chromosome = trọng số mạng neural nhỏ:
    Input  : 36 tia LiDAR (normalized) + góc đến goal + khoảng cách đến goal
    Hidden : 24 neurons (tanh)
    Output : 4 vận tốc bánh xe ∈ [-1, 1]

PIPELINE:
    [lidar 36] + [goal_angle, goal_dist]
          ↓  NeuralController.forward()
    [w_fl, w_fr, w_rl, w_rr] ∈ [-1,1]
          ↓  MecanumKinematics.compute()
    (vx, vy, wz)  →  clip vx≥0  →  publish cmd_vel

Yêu cầu:
    pip install numpy
    pip install gz-transport  (chỉ khi dùng Gazebo thật)

Chạy simulation (test không cần Gazebo):
    python3 ga_maze_solver.py --sim --pop 30 --generations 50

Chạy với Gazebo (spawn N robot trước bằng spawn_robot.sh):
    python3 ga_maze_solver.py --robots 10 --pop 50 --generations 100 --goal 8.0 5.0
"""

import numpy as np
import time
import math
import argparse
import json
from dataclasses import dataclass, field
from typing import List, Tuple, Optional
from copy import deepcopy

# ── Thử import gz-transport ────────────────────────────────────────────────
try:
    from gz.transport13 import Node as GzNode
    import gz.msgs.laserscan_pb2 as LaserScanMsg
    import gz.msgs.pose_v_pb2    as PoseMsg
    import gz.msgs.twist_pb2     as TwistMsg
    import gz.msgs.double_pb2    as DoubleMsg
    GZ_AVAILABLE = True
except ImportError:
    print("[WARN] gz-transport not found. Dùng --sim để chạy mà không cần Gazebo.")
    GZ_AVAILABLE = False

# HẰNG SỐ
LIDAR_SECTORS   = 36       # phải bằng <samples> trong SDF
LIDAR_MAX_RANGE = 12.0     # m
LIDAR_DANGER    = 0.25     # m — sát tường: phạt nặng
LIDAR_NEAR      = 0.50     # m — gần tường: phạt nhẹ

WHEEL_RADIUS    = 0.024    # m  (từ SDF)
HALF_WHEELBASE  = 0.08     # lx
HALF_SEPARATION = 0.09     # ly
MAX_WHEEL_VEL   = 15.0     # rad/s ([-1,1] map sang [-MAX, +MAX])

MAX_VX          = 0.50     # m/s tối đa
MAX_VY          = 0.30
MAX_WZ          = 1.20

EPISODE_STEPS   = 300      # bước/episode
STEP_DT         = 0.1      # giây/bước (sync lidar 10Hz)
GOAL_RADIUS     = 0.40     # m — coi là đến đích

STUCK_WINDOW    = 30       # steps để kiểm tra stuck
STUCK_THRESH    = 0.10     # m di chuyển tối thiểu trong window


# MECANUM KINEMATICS
class MecanumKinematics:
    """
    4 bánh → (vx, vy, wz).   Ràng buộc KHÔNG LÙI: clip vx ≥ 0.

    Sơ đồ (nhìn từ trên):
      FL (+x,+y) ──── FR (+x,-y)
           |                |
      RL (-x,+y) ──── RR (-x,-y)

    Forward kinematics mecanum:
      vx = r/4 * ( w_fl + w_fr + w_rl + w_rr )
      vy = r/4 * (-w_fl + w_fr + w_rl - w_rr )
      wz = r/(4*(lx+ly)) * (-w_fl + w_fr - w_rl + w_rr )
    """
    def __init__(self):
        self.r  = WHEEL_RADIUS
        self.lx = HALF_WHEELBASE
        self.ly = HALF_SEPARATION

    def compute(self, action: np.ndarray) -> Tuple[float, float, float]:
        w  = np.clip(action, -1.0, 1.0) * MAX_WHEEL_VEL
        r, lx, ly = self.r, self.lx, self.ly
        vx = r/4 * (w[0] + w[1] + w[2] + w[3])
        vy = r/4 * (-w[0] + w[1] + w[2] - w[3])
        wz = r / (4*(lx+ly)) * (-w[0] + w[1] - w[2] + w[3])
        vx = max(0.0, vx)                          # KHÔNG LÙI
        return (float(np.clip(vx,  0.0,  MAX_VX)),
                float(np.clip(vy, -MAX_VY, MAX_VY)),
                float(np.clip(wz, -MAX_WZ, MAX_WZ)))

# NEURAL CONTROLLER  (chromosome = flattened weights)
class NeuralController:
    """
    Input  (38): 36 lidar rays (norm) + goal_angle/π + goal_dist/MAX_RANGE
    Hidden (24): tanh
    Output  (4): tanh → [-1,1]  (wheel velocities)

    Chromosome size: 38*24 + 24 + 24*4 + 4 = 1036
    """
    IN, HID, OUT = LIDAR_SECTORS + 2, 24, 4
    CHROM_SIZE   = IN*HID + HID + HID*OUT + OUT   # 1036

    def __init__(self, weights: Optional[np.ndarray] = None):
        if weights is None:
            weights = np.random.randn(self.CHROM_SIZE) * 0.5
        self.set_weights(weights)

    def set_weights(self, w: np.ndarray):
        self.weights = w.copy()
        i = 0
        s = self.IN*self.HID
        self.W1 = w[i:i+s].reshape(self.IN, self.HID); i += s
        self.b1 = w[i:i+self.HID];                      i += self.HID
        s = self.HID*self.OUT
        self.W2 = w[i:i+s].reshape(self.HID, self.OUT); i += s
        self.b2 = w[i:i+self.OUT]

    def forward(self, lidar_norm: np.ndarray,
                goal_angle: float, goal_dist: float) -> np.ndarray:
        x = np.concatenate([lidar_norm,
                             [goal_angle / math.pi],
                             [min(goal_dist / LIDAR_MAX_RANGE, 1.0)]])
        h = np.tanh(x @ self.W1 + self.b1)
        return np.tanh(h @ self.W2 + self.b2)

    def get_weights(self) -> np.ndarray:
        return self.weights.copy()

# ROBOT ENVIRONMENT (Gazebo)
@dataclass
class RobotState:
    x: float = 0.0
    y: float = 0.0
    yaw: float = 0.0
    lidar: np.ndarray = field(
        default_factory=lambda: np.full(LIDAR_SECTORS, LIDAR_MAX_RANGE))


class RobotEnv:
    """Giao tiếp với 1 robot Gazebo qua gz-transport."""

    def __init__(self, robot_id: int, goal: Tuple[float, float]):
        self.id    = robot_id
        self.goal  = goal
        self.state = RobotState()
        self.kin   = MecanumKinematics()
        self._got_lidar = False
        self._got_pose  = False

        if GZ_AVAILABLE:
            self.node = GzNode()
            self.node.subscribe(f"/scan{robot_id}",
                                self._cb_lidar, LaserScanMsg.LaserScan)
            self.node.subscribe(f"/model/robot_{robot_id}/pose",
                                self._cb_pose, PoseMsg.Pose_V)
            self._pub_vel = self.node.advertise(
                f"/model/robot_{robot_id}/cmd_vel", TwistMsg.Twist)
            self._pub_fl  = self.node.advertise(
                f"/wheel_fl_{robot_id}", DoubleMsg.Double)
            self._pub_fr  = self.node.advertise(
                f"/wheel_fr_{robot_id}", DoubleMsg.Double)
            self._pub_rl  = self.node.advertise(
                f"/wheel_rl_{robot_id}", DoubleMsg.Double)
            self._pub_rr  = self.node.advertise(
                f"/wheel_rr_{robot_id}", DoubleMsg.Double)

    def _cb_lidar(self, msg):
        r = np.array(msg.ranges)
        r = np.where(np.isfinite(r), r, LIDAR_MAX_RANGE)
        self.state.lidar = r
        self._got_lidar  = True

    def _cb_pose(self, msg):
        if not msg.pose:
            return
        p = msg.pose[0]
        self.state.x = p.position.x
        self.state.y = p.position.y
        q = p.orientation
        siny = 2*(q.w*q.z + q.x*q.y)
        cosy = 1 - 2*(q.y*q.y + q.z*q.z)
        self.state.yaw = math.atan2(siny, cosy)
        self._got_pose  = True

    def goal_polar(self) -> Tuple[float, float]:
        """(angle_in_robot_frame, distance) đến goal."""
        dx  = self.goal[0] - self.state.x
        dy  = self.goal[1] - self.state.y
        d   = math.hypot(dx, dy)
        ang = math.atan2(dy, dx) - self.state.yaw
        ang = (ang + math.pi) % (2*math.pi) - math.pi
        return ang, d

    def reached_goal(self) -> bool:
        return self.goal_polar()[1] < GOAL_RADIUS

    def send_action(self, action: np.ndarray):
        if not GZ_AVAILABLE:
            return
        w = np.clip(action, -1.0, 1.0)
        for pub, val in zip(
            [self._pub_fl, self._pub_fr, self._pub_rl, self._pub_rr], w
        ):
            m = DoubleMsg.Double(); m.data = float(val) * MAX_WHEEL_VEL
            pub.publish(m)
        vx, vy, wz = self.kin.compute(w)
        t = TwistMsg.Twist()
        t.linear.x = vx; t.linear.y = vy; t.angular.z = wz
        self._pub_vel.publish(t)

    def stop(self):
        if GZ_AVAILABLE:
            self.send_action(np.zeros(4))

    def wait_data(self, timeout=3.0):
        t0 = time.time()
        while not (self._got_lidar and self._got_pose):
            if time.time()-t0 > timeout:
                break
            time.sleep(0.05)


# SIMULATION ENV (không cần Gazebo — dùng để test)
class SimRobotEnv(RobotEnv):
    """
    Môi trường giả lập 2D: tường = hình vuông 10×10 m.
    Lidar: ray-cast analytic đến 4 tường.
    """
    WALL = 5.0   # bán kính maze từ tâm

    def __init__(self, robot_id: int, goal: Tuple[float, float],
                 start: Tuple[float, float] = (0.0, 0.0)):
        self.id    = robot_id
        self.goal  = goal
        self.kin   = MecanumKinematics()
        self._reset(start)

    def _reset(self, start=(0.0, 0.0)):
        self.state = RobotState(x=start[0], y=start[1], yaw=0.0)
        self._update_lidar()

    def _update_lidar(self):
        rays = np.zeros(LIDAR_SECTORS)
        W = self.WALL
        for i in range(LIDAR_SECTORS):
            ang = 2*math.pi*i/LIDAR_SECTORS + self.state.yaw
            dx, dy = math.cos(ang), math.sin(ang)
            dists = []
            if abs(dx) > 1e-9:
                d = (W - self.state.x)/dx if dx>0 else (-W - self.state.x)/dx
                if d > 0: dists.append(d)
            if abs(dy) > 1e-9:
                d = (W - self.state.y)/dy if dy>0 else (-W - self.state.y)/dy
                if d > 0: dists.append(d)
            rays[i] = min(dists) if dists else LIDAR_MAX_RANGE
        self.state.lidar = np.clip(rays, 0, LIDAR_MAX_RANGE)

    def send_action(self, action: np.ndarray):
        vx, vy, wz = self.kin.compute(action)
        cy, sy = math.cos(self.state.yaw), math.sin(self.state.yaw)
        self.state.x   += (vx*cy - vy*sy) * STEP_DT
        self.state.y   += (vx*sy + vy*cy) * STEP_DT
        self.state.yaw  = (self.state.yaw + wz*STEP_DT + math.pi) % (2*math.pi) - math.pi
        self._update_lidar()

    def stop(self): pass
    def wait_data(self, timeout=0): pass


# FITNESS FUNCTION
#
#  f = Σ_steps [
#        - P_step                         (time pressure, mỗi bước)
#        + W_progress * (prev_d - new_d)  (tiến gần goal)
#        + W_explore  * dist_moved        (tổng quãng đường)
#        + wall_penalty(lidar)            (tránh tường)
#        + still_penalty(vx)             (tránh đứng im)
#     ]
#     + R_goal  (nếu đến đích)
#     + P_stuck (nếu kẹt, kết thúc sớm)
#
WEIGHTS = dict(
    goal_bonus  = 500.0,   # thưởng một lần khi về đích
    time_bonus  = 2.0,     # thêm mỗi bước còn dư khi về đích (đến nhanh hơn = +++)
    progress    = 10.0,    # mỗi bước tiến gần goal
    explore     = 0.5,     # mỗi đơn vị khoảng cách đã đi
    wall_danger = -5.0,    # lidar < LIDAR_DANGER
    wall_near   = -1.0,    # lidar < LIDAR_NEAR
    stuck       = -80.0,   # phát hiện kẹt (kết thúc episode)
    step        = -0.1,    # mỗi bước
    still       = -2.0,    # vx ≈ 0 (đứng im)
)


def evaluate(controller: NeuralController, env: RobotEnv,
             start: Tuple[float, float] = (0.0, 0.0)) -> float:
    if isinstance(env, SimRobotEnv):
        env._reset(start)
    env.wait_data()

    W = WEIGHTS
    fitness     = 0.0
    prev_d      = env.goal_polar()[1]
    total_moved = 0.0
    pos_hist    = [(env.state.x, env.state.y)]

    for step in range(EPISODE_STEPS):
        st = env.state
        lidar_n = st.lidar / LIDAR_MAX_RANGE
        ang, dist = env.goal_polar()

        action = controller.forward(lidar_n, ang, dist)
        env.send_action(action)
        if GZ_AVAILABLE and not isinstance(env, SimRobotEnv):
            time.sleep(STEP_DT)

        # ── Rewards / Penalties ─────────────────────────────────────────
        fitness += W["step"]

        # Progress
        new_ang, new_dist = env.goal_polar()
        fitness += W["progress"] * (prev_d - new_dist)

        # Exploration
        nx, ny = env.state.x, env.state.y
        moved   = math.hypot(nx - pos_hist[-1][0], ny - pos_hist[-1][1])
        total_moved += moved
        fitness += W["explore"] * total_moved

        # Wall penalty
        min_r = float(np.min(st.lidar))
        if min_r < LIDAR_DANGER:
            fitness += W["wall_danger"]
        elif min_r < LIDAR_NEAR:
            fitness += W["wall_near"]

        # Still penalty (robot không tiến)
        vx, _, _ = env.kin.compute(action)
        if vx < 0.02:
            fitness += W["still"]

        # Stuck detection
        pos_hist.append((nx, ny))
        if len(pos_hist) > STUCK_WINDOW:
            ox, oy = pos_hist[-STUCK_WINDOW]
            if math.hypot(nx-ox, ny-oy) < STUCK_THRESH:
                fitness += W["stuck"]
                env.stop()
                return fitness

        prev_d = new_dist

        # Goal reached
        if env.reached_goal():
            time_bonus = W["goal_bonus"] + (EPISODE_STEPS - step) * W["time_bonus"]
            fitness += time_bonus
            env.stop()
            return fitness

    env.stop()
    return fitness


# GENETIC ALGORITHM
@dataclass
class GAConfig:
    pop_size       : int   = 50
    n_generations  : int   = 100
    elite_frac     : float = 0.10     # tỉ lệ giữ nguyên (elitism)
    crossover_prob : float = 0.80
    mutation_prob  : float = 0.15     # xác suất đột biến mỗi gen
    mutation_sigma : float = 0.20     # σ ban đầu
    tournament_k   : int   = 4        # tournament size
    sigma_decay    : float = 0.995    # giảm σ mỗi thế hệ
    sigma_min      : float = 0.02


class GeneticAlgorithm:
    """
    ┌──────────────────────────────────────────────────────────────┐
    │  Khởi tạo quần thể N(0,0.5) ngẫu nhiên                     │
    │  ─────────────────────────────────────────────────────────  │
    │  Lặp mỗi thế hệ:                                            │
    │    1. Đánh giá fitness tất cả cá thể                        │
    │    2. Sắp xếp giảm dần                                      │
    │    3. Elitism: copy top-k vào thế hệ mới (không thay đổi)  │
    │    4. Tournament Selection: chọn 2 cha mẹ                   │
    │    5. BLX-α Crossover (hoặc Uniform Crossover 50/50)        │
    │    6. Gaussian Mutation (σ giảm dần qua các thế hệ)        │
    │    7. Lặp 4-6 cho đến đủ pop_size                           │
    │    8. Log, save checkpoint                                   │
    └──────────────────────────────────────────────────────────────┘
    """

    def __init__(self, cfg: GAConfig, goal: Tuple[float, float],
                 robot_ids: List[int], use_sim: bool = True):
        self.cfg      = cfg
        self.use_sim  = use_sim
        self.n_elite  = max(1, int(cfg.pop_size * cfg.elite_frac))
        csz           = NeuralController.CHROM_SIZE

        self.envs = [
            SimRobotEnv(rid, goal) if use_sim else RobotEnv(rid, goal)
            for rid in robot_ids
        ]
        self.pop = [NeuralController(np.random.randn(csz)*0.5)
                    for _ in range(cfg.pop_size)]
        self.best_ctrl   : Optional[NeuralController] = None
        self.best_fitness: float = -1e9
        self.history     : List[dict] = []

    # ── Chọn lọc tự nhiên: Tournament ────────────────────────────────────
    def _tournament(self, fits: np.ndarray) -> int:
        """
        Chọn ngẫu nhiên k cá thể, trả về index cái tốt nhất.
        Tăng k → áp lực chọn lọc tăng → hội tụ nhanh hơn nhưng dễ mất đa dạng.
        """
        cands = np.random.choice(len(fits), self.cfg.tournament_k, replace=False)
        return int(cands[np.argmax(fits[cands])])

    # ── Lai ghép: BLX-α + Uniform ────────────────────────────────────────
    def _crossover(self, p1: np.ndarray, p2: np.ndarray,
                   alpha: float = 0.3) -> Tuple[np.ndarray, np.ndarray]:
        """
        BLX-α (Blend Crossover):
          Con nằm trong khoảng mở rộng [min - α*d, max + α*d].
          Tốt hơn single-point cho không gian liên tục.
        Uniform Crossover:
          Mỗi gen được chọn ngẫu nhiên từ cha/mẹ.
          Tăng khả năng khám phá.
        """
        if np.random.rand() > self.cfg.crossover_prob:
            return p1.copy(), p2.copy()
        if np.random.rand() < 0.5:                     # BLX-α
            lo = np.minimum(p1, p2)
            hi = np.maximum(p1, p2)
            d  = hi - lo
            c1 = np.random.uniform(lo - alpha*d, hi + alpha*d)
            c2 = np.random.uniform(lo - alpha*d, hi + alpha*d)
        else:                                           # Uniform
            mask = np.random.rand(len(p1)) < 0.5
            c1   = np.where(mask, p1, p2)
            c2   = np.where(mask, p2, p1)
        return c1, c2

    # ── Đột biến: Adaptive Gaussian ──────────────────────────────────────
    def _mutate(self, chrom: np.ndarray) -> np.ndarray:
        """
        Gaussian mutation thích nghi:
          - mutation_prob: mỗi gen có xác suất bị nhiễu
          - σ giảm dần qua thế hệ (khai thác dần thay vì khám phá)
          - Creep mutation nhỏ trên toàn bộ gen (fine-tune)
        """
        c = chrom.copy()
        s = self.cfg.mutation_sigma
        mask = np.random.rand(len(c)) < self.cfg.mutation_prob
        c[mask] += np.random.randn(mask.sum()) * s
        # Creep: nhiễu rất nhỏ trên tất cả gen (0.5% mỗi gen)
        creep = np.random.rand(len(c)) < 0.005
        c[creep] += np.random.randn(creep.sum()) * s * 0.1
        return c

    # ── Đánh giá quần thể ────────────────────────────────────────────────
    def _eval_all(self) -> np.ndarray:
        fits = np.full(self.cfg.pop_size, -1e9)
        n    = len(self.envs)
        for i, ind in enumerate(self.pop):
            env  = self.envs[i % n]
            fits[i] = evaluate(ind, env)
        return fits

    # ── Vòng lặp chính ───────────────────────────────────────────────────
    def run(self) -> NeuralController:
        print(f"\n{'='*58}")
        print(f"  GA MAZE SOLVER   pop={self.cfg.pop_size}"
              f"   gen={self.cfg.n_generations}"
              f"   {'SIM' if self.use_sim else 'GAZEBO'}")
        print(f"  Chromosome size: {NeuralController.CHROM_SIZE}"
              f"   Lidar sectors: {LIDAR_SECTORS}")
        print(f"{'='*58}\n")

        for gen in range(self.cfg.n_generations):
            t0   = time.time()
            fits = self._eval_all()
            idx  = np.argsort(fits)[::-1]

            gen_best = fits[idx[0]]
            gen_avg  = fits.mean()

            if gen_best > self.best_fitness:
                self.best_fitness = gen_best
                self.best_ctrl    = deepcopy(self.pop[idx[0]])

            elapsed = time.time() - t0
            self.history.append({
                "gen": gen+1, "best": round(float(gen_best), 2),
                "avg": round(float(gen_avg), 2),
                "sigma": round(self.cfg.mutation_sigma, 5)
            })
            print(f"  Gen {gen+1:4d}  best={gen_best:8.1f}  "
                  f"avg={gen_avg:7.1f}  σ={self.cfg.mutation_sigma:.4f}"
                  f"  ({elapsed:.1f}s)")

            # ── Tạo thế hệ mới ──────────────────────────────────────────
            new_pop = [deepcopy(self.pop[idx[i]]) for i in range(self.n_elite)]

            while len(new_pop) < self.cfg.pop_size:
                i1 = self._tournament(fits)
                i2 = self._tournament(fits)
                c1, c2 = self._crossover(self.pop[i1].get_weights(),
                                         self.pop[i2].get_weights())
                new_pop.append(NeuralController(self._mutate(c1)))
                if len(new_pop) < self.cfg.pop_size:
                    new_pop.append(NeuralController(self._mutate(c2)))

            self.pop = new_pop[:self.cfg.pop_size]

            # Adaptive sigma decay
            self.cfg.mutation_sigma = max(
                self.cfg.sigma_min,
                self.cfg.mutation_sigma * self.cfg.sigma_decay
            )

            if (gen + 1) % 10 == 0:
                self._save(f"checkpoint_gen{gen+1}.json")

        print(f"\n  ✓ Hoàn thành. Best fitness = {self.best_fitness:.1f}")
        self._save("ga_best.json")
        return self.best_ctrl

    def _save(self, path: str):
        data = {
            "best_fitness": float(self.best_fitness),
            "weights": self.best_ctrl.get_weights().tolist()
                       if self.best_ctrl else [],
            "history": self.history,
            "config": {
                "lidar_sectors": LIDAR_SECTORS,
                "hidden_dim": NeuralController.HID,
            }
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        print(f"    → {path}")

    def load(self, path: str):
        with open(path) as f:
            d = json.load(f)
        w = np.array(d["weights"])
        self.best_ctrl    = NeuralController(w)
        self.best_fitness = d["best_fitness"]
        # Thay thế cá thể đầu tiên bằng best đã load để tiếp tục
        self.pop[0] = deepcopy(self.best_ctrl)
        print(f"    ← Loaded {path} (fitness={self.best_fitness:.1f})")


# MAIN
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--robots",      type=int,   default=5)
    ap.add_argument("--pop",         type=int,   default=50)
    ap.add_argument("--generations", type=int,   default=100)
    ap.add_argument("--goal",   nargs=2, type=float, default=[8.0, 5.0],
                    metavar=("GX","GY"))
    ap.add_argument("--load",        type=str,   default=None)
    ap.add_argument("--sim",         action="store_true",
                    help="Chế độ giả lập, không cần Gazebo")
    args = ap.parse_args()

    use_sim = args.sim or not GZ_AVAILABLE

    cfg = GAConfig(
        pop_size       = args.pop,
        n_generations  = args.generations,
        elite_frac     = 0.10,
        crossover_prob = 0.80,
        mutation_prob  = 0.15,
        mutation_sigma = 0.20,
        tournament_k   = 4,
        sigma_decay    = 0.995,
        sigma_min      = 0.02,
    )

    ga = GeneticAlgorithm(cfg,
                          goal=tuple(args.goal),
                          robot_ids=list(range(1, args.robots+1)),
                          use_sim=use_sim)
    if args.load:
        ga.load(args.load)

    ga.run()


if __name__ == "__main__":
    main()