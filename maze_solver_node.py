#!/usr/bin/env python3
"""
maze_solver_v6.py  —  Mê cung 5×5 | LiDAR-first | BFS khám phá | Mecanum omni-drive
Điểm khác biệt so với v5:

  [LIDAR-PRIMARY]
    - LiDAR bắt buộc: quét 4 sector (E/N/W/S) tại mỗi ô → cập nhật bản đồ tường
    - WALL_THRESH = 0.27 m (CELL/2 = 0.25 m + buffer nhỏ)
    - 500 sample/360° → mỗi sector ±20° ~ 56 sample → độ phân giải cao
    - Không dùng bản đồ cứng (hardcoded) — robot tự khám phá hoàn toàn qua LiDAR

  [BFS KHÁM PHÁ]
    - Tường = None (chưa biết): BFS coi là có thể đi qua (optimistic)
    - Tường = True (LiDAR xác nhận): BFS bỏ qua
    - Khi đang di chuyển: LiDAR front-sector làm phanh khẩn cấp
    - Phanh → cập nhật tường True → lùi → quét lại → BFS mới

  [MECANUM OMNI-DRIVE]
    - Không cần quay hướng trước khi di chuyển (không có state TURN)
    - _move_to(tx,ty): tính vận tốc robot-frame từ world-frame → strafe trực tiếp
    - Giảm thời gian di chuyển; robot vẫn giữ heading trong khi strafe

  [CELL & TỌA ĐỘ]
    - CELL = 0.5 m (đúng theo SDF), khoảng cách tâm ô = 0.5 m
    - OX = 2.0, OY = -2.0 (tự hiệu chỉnh từ odom spawn đầu tiên)
    - START world (2.25, -3.25) → cell (2,0)
    - GOAL  world (4.25, -3.25) → cell (2,4)

  [DEBUG]
    - In toàn bộ lịch sử ô đã thăm mỗi khi đến ô mới
    - In khoảng cách LiDAR từng sector tại mỗi ô dừng
    - Vẽ ASCII maze sau mỗi lần quét (tường đã biết / chưa biết / đã thăm / path)
"""
import rclpy
import math
import numpy as np
from rclpy.node import Node
from collections import deque
from enum import Enum, auto
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy

#  MAZE CONFIG
ROWS, COLS   = 5, 5
CELL         = 0.5            # m — chiều rộng 1 ô (theo SDF)

START_W      = (2.25, -3.25)  # world coords của ô START
GOAL_W       = (4.25, -3.25)  # world coords của ô GOAL

# Cell index (0-indexed row,col):  (2,0) = START,  (2,4) = GOAL
START_CELL   = (2, 0)
GOAL_CELL    = (2, 4)

# Hướng: 0=E  1=N  2=W  3=S
DNAME = ['E', 'N', 'W', 'S']
DYAW  = [0.0,  math.pi/2, math.pi, -math.pi/2]
DDR   = [ 0,   -1,  0,  +1]   # delta row
DDC   = [+1,    0, -1,   0]   # delta col
OPP   = [ 2,    3,  0,   1]   # opposite direction

#  MOTION PARAMS
LIN_SPD      = 0.2    # m/s nominal
LIN_SPD_MIN  = 0.2    # m/s tối thiểu (slow-down khi gần đích)
KP_ANG       = 4.5     # gain hiệu chỉnh heading khi move
ANG_MAX      = 1.2     # rad/s max angular
ARRIVE_R     = 0.055   # m  bán kính "đã đến"

#  LIDAR PARAMS
LIDAR_HALF_DEG = 20     # ° — nửa góc mỗi sector  → tổng 40°/sector
WALL_THRESH    = 0.27   # m — tường nếu min_range < WALL_THRESH
BRAKE_DIST     = 0.2   # m — phanh khẩn cấp
SCAN_SETTLE    = 10     # timer-ticks chờ ổn định trước khi quét (10×50ms=0.5s)
BACKUP_DIST    = 0.1   # m — lùi sau khi phanh

class State(Enum):
    WAIT   = auto()
    SCAN   = auto()
    PLAN   = auto()
    MOVE   = auto()
    BACKUP = auto()
    DONE   = auto()

def wrap(a):
    while a >  math.pi: a -= 2 * math.pi
    while a < -math.pi: a += 2 * math.pi
    return a

def in_bounds(r, c):
    return 0 <= r < ROWS and 0 <= c < COLS

#  COORDINATE HELPERS  (dùng OX, OY được set sau calibrate)
def c2w(r, c, ox, oy):
    return ox + (c + 0.5) * CELL, oy - (r + 0.5) * CELL

def w2c(x, y, ox, oy):
    col = int(round((x - ox) / CELL - 0.5))
    row = int(round(-(y - oy) / CELL - 0.5))
    return (row, col)


#  NODE
class MazeSolverV6(Node):

    def __init__(self):
        super().__init__('maze_solver_v6')
        self.declare_parameter('robot_id', 1)
        rid = self.get_parameter('robot_id').value

        qos = QoSProfile(depth=5,
                         reliability=QoSReliabilityPolicy.BEST_EFFORT,
                         history=QoSHistoryPolicy.KEEP_LAST)
        self.pub = self.create_publisher(Twist, f'/model/robot_{rid}/cmd_vel', 10)
        self.create_subscription(LaserScan, f'/scan{rid}',                  self._cb_scan, qos)
        self.create_subscription(Odometry,  f'/model/robot_{rid}/odometry', self._cb_odom, qos)

        # Sensor state
        self._scan    = np.array([])
        self._odom_ok = False
        self.wx = self.wy = self.yaw = 0.0

        # Origin (calibrated from first odom)
        self._ox = START_W[0] - (START_CELL[1] + 0.5) * CELL   # = 2.0
        self._oy = START_W[1] + (START_CELL[0] + 0.5) * CELL   # = -2.0

        # Wall map: True=wall, False=open, None=unknown
        self.walls = {(r, c): {d: None for d in range(4)}
                      for r in range(ROWS) for c in range(COLS)}
        self._set_border_walls()

        # Navigation
        self.cur  = START_CELL
        self.goal = GOAL_CELL
        self.path = []

        # Visit history — danh sách ô theo thứ tự thăm để debug
        self.visited     = [self.cur]
        self.visited_set = {self.cur}

        # Motion state
        self.state    = State.WAIT
        self._settle  = 0
        self._tgt     = self.cur
        self._tgt_dir = 0
        self._tgt_w   = c2w(*self.cur, self._ox, self._oy)
        self._bk_wx   = 0.0
        self._bk_wy   = 0.0
        self.step     = 0

        self.get_logger().info(
            f'  maze_solver_v6  —  LiDAR-first + BFS\n'
            f'  START={self.cur}  GOAL={self.goal}\n'
            f'  CELL={CELL}m  WALL_THRESH={WALL_THRESH}m\n'
            f'  Sector: ±{LIDAR_HALF_DEG}° (~{2*LIDAR_HALF_DEG} deg / sector)\n'
        )

        self.create_timer(0.05, self._loop)

    # ── Border walls 
    def _set_border_walls(self):
        for r in range(ROWS):
            for c in range(COLS):
                if r == 0:        self.walls[(r, c)][1] = True   # North border
                if r == ROWS - 1: self.walls[(r, c)][3] = True   # South border
                if c == 0:        self.walls[(r, c)][2] = True   # West border
                if c == COLS - 1: self.walls[(r, c)][0] = True   # East border

    # ── Callbacks ───
    def _cb_scan(self, msg: LaserScan):
        raw = np.array(msg.ranges, dtype=np.float32)
        self._scan = np.where(
            np.isfinite(raw) & (raw > 0.05) & (raw < 12.0), raw, 12.0)

    def _cb_odom(self, msg: Odometry):
        p = msg.pose.pose.position
        q = msg.pose.pose.orientation
        self.wx  = p.x
        self.wy  = p.y
        self.yaw = math.atan2(
            2.0 * (q.w * q.z + q.x * q.y),
            1.0 - 2.0 * (q.y ** 2 + q.z ** 2))
        if not self._odom_ok:
            self._odom_ok = True
            # Tinh chỉnh origin từ vị trí spawn thực tế
            self._ox = self.wx - (START_CELL[1] + 0.5) * CELL
            self._oy = self.wy + (START_CELL[0] + 0.5) * CELL
            self._tgt_w = c2w(*self.cur, self._ox, self._oy)
            self.get_logger().info(
                f'[CALIB] spawn=({self.wx:.4f},{self.wy:.4f})  '
                f'OX={self._ox:.4f} OY={self._oy:.4f}  '
                f'START_world={c2w(*START_CELL, self._ox, self._oy)}  '
                f'GOAL_world={c2w(*GOAL_CELL, self._ox, self._oy)}')

    # ── LiDAR sector 

    def _sector_min(self, world_yaw):
        """Min range trong sector ±LIDAR_HALF_DEG° quanh hướng world_yaw."""
        if not len(self._scan):
            return 12.0
        n   = len(self._scan)
        rel = wrap(world_yaw - self.yaw)
        ci  = int((math.degrees(rel) % 360) / 360 * n) % n
        hw  = max(1, int(LIDAR_HALF_DEG / 360 * n))   # ~28 samples cho 500/360°
        idx = np.array([(ci + k) % n for k in range(-hw, hw + 1)])
        return float(np.min(self._scan[idx]))

    def _scan_walls_at_cur(self):
        """
        Quét LiDAR 4 hướng tại ô hiện tại.
        Cập nhật wall map của ô này VÀ ô hàng xóm tương ứng.
        """
        readings = {}
        for d in range(4):
            dist = self._sector_min(DYAW[d])
            has_wall = bool(dist < WALL_THRESH)
            readings[d] = (dist, has_wall)
            self.walls[self.cur][d] = has_wall
            nr, nc = self.cur[0] + DDR[d], self.cur[1] + DDC[d]
            if in_bounds(nr, nc):
                self.walls[(nr, nc)][OPP[d]] = has_wall

        log = '  '.join(
            f'{DNAME[d]}={readings[d][0]:.3f}m{"█" if readings[d][1] else "·"}'
            for d in range(4))
        self.get_logger().info(f'[LIDAR] cell={self.cur}  yaw={math.degrees(self.yaw):.0f}°  {log}')

    #BFS (optimistic: None = có thể đi)
    def _bfs(self, src, dst):
        q    = deque([(src, [src])])
        seen = {src}
        while q:
            (r, c), path = q.popleft()
            for d in range(4):
                nr, nc = r + DDR[d], c + DDC[d]
                nb = (nr, nc)
                if not in_bounds(nr, nc) or nb in seen:
                    continue
                if self.walls[(r, c)][d] is True:   # tường đã xác nhận
                    continue
                seen.add(nb)
                new_path = path + [nb]
                if nb == dst:
                    return new_path[1:]
                q.append((nb, new_path))
        return None

    #Motion 
    def _stop(self):
        self.pub.publish(Twist())

    def _move_to(self, tx, ty, speed):
        """
        Mecanum omni-drive: di chuyển thẳng đến (tx,ty) không cần quay trước.
        Tính vận tốc robot-frame từ hướng world-frame.
        """
        dx, dy = tx - self.wx, ty - self.wy
        dist   = math.hypot(dx, dy)
        if dist < 0.002:
            self._stop()
            return

        target_yaw = math.atan2(dy, dx)
        ux, uy     = dx / dist, dy / dist

        cos_y = math.cos(self.yaw)
        sin_y = math.sin(self.yaw)
        vxr   =  cos_y * ux * speed + sin_y * uy * speed
        vyr   = -sin_y * ux * speed + cos_y * uy * speed

        # Giữ heading về phía đích để sector front LiDAR luôn canh đúng hướng
        wz = float(np.clip(KP_ANG * wrap(target_yaw - self.yaw), -ANG_MAX, ANG_MAX))

        t = Twist()
        t.linear.x  = float(vxr)
        t.linear.y  = float(vyr)
        t.angular.z = wz
        self.pub.publish(t)

    #Visit logger 
    def _register_arrive(self):
        if self.cur not in self.visited_set:
            self.visited.append(self.cur)
            self.visited_set.add(self.cur)
        wx_c, wy_c = c2w(*self.cur, self._ox, self._oy)
        self.get_logger().info(
            f'[ARRIVE] step={self.step:3d}  cell={self.cur}  '
            f'world=({self.wx:.3f},{self.wy:.3f})  '
            f'cell_center=({wx_c:.3f},{wy_c:.3f})\n'
            f'[HISTORY] visited={self.visited}')

    #Main FSM 
    def _loop(self):
        if not self._odom_ok or not len(self._scan):
            return

        #WAIT 
        if self.state == State.WAIT:
            self._stop()
            self.state = State.SCAN

        #SCAN: dừng xe, đợi ổn định, quét LiDAR, lập BFS ─
        elif self.state == State.SCAN:
            self._stop()
            self._settle += 1
            if self._settle < SCAN_SETTLE:
                return

            self._scan_walls_at_cur()
            self._print_maze()

            self.path = self._bfs(self.cur, self.goal)
            if self.path:
                self.get_logger().info(f'[BFS ✓] {[self.cur] + self.path}  ({len(self.path)} bước)')
            else:
                self.get_logger().error(f'[BFS ✗] Không tìm được đường từ {self.cur} đến {self.goal}!')

            self._settle = 0
            self.state   = State.PLAN

        #PLAN: chọn bước tiếp theo trong BFS path ─
        elif self.state == State.PLAN:
            if self.cur == self.goal:
                self._stop()
                self.get_logger().info(
                    f'\n╔══════════════════════════════════╗\n'
                    f'║  ✓ ĐẾN ĐÍCH!  steps={self.step}\n'
                    f'║  LỊCH SỬ ĐI: {self.visited}\n'
                    f'╚══════════════════════════════════╝')
                self._print_maze()
                self.state = State.DONE
                return

            if not self.path:
                self.get_logger().error(f'[BẾ TẮC] Không có đường từ {self.cur}')
                self._stop()
                self.state = State.DONE
                return

            tgt = self.path.pop(0)
            dr  = tgt[0] - self.cur[0]
            dc  = tgt[1] - self.cur[1]
            d   = next(i for i in range(4) if DDR[i] == dr and DDC[i] == dc)

            self._tgt     = tgt
            self._tgt_dir = d
            self._tgt_w   = c2w(*tgt, self._ox, self._oy)
            self.step    += 1

            self.get_logger().info(
                f'[PLAN] step={self.step}  {self.cur}→{tgt}  '
                f'dir={DNAME[d]}  target_world={self._tgt_w}')
            self.state = State.MOVE

        #MOVE: mecanum omni-drive, LiDAR brake 
        elif self.state == State.MOVE:
            tx, ty = self._tgt_w
            dist   = math.hypot(tx - self.wx, ty - self.wy)

            # Đến nơi
            if dist < ARRIVE_R:
                self._stop()
                self.cur = self._tgt
                self._register_arrive()
                self._settle = 0
                self.state   = State.SCAN
                return

            # Phanh khẩn cấp: LiDAR phát hiện vật cản phía trước
            front = self._sector_min(DYAW[self._tgt_dir])
            if front < BRAKE_DIST:
                self._stop()
                self.get_logger().warn(
                    f'[BRAKE] dir={DNAME[self._tgt_dir]}  front={front:.3f}m < {BRAKE_DIST}m  '
                    f'→ đánh dấu tường tại {self.cur}[{DNAME[self._tgt_dir]}]')
                self.walls[self.cur][self._tgt_dir] = True
                nr, nc = self.cur[0] + DDR[self._tgt_dir], self.cur[1] + DDC[self._tgt_dir]
                if in_bounds(nr, nc):
                    self.walls[(nr, nc)][OPP[self._tgt_dir]] = True
                self.path  = []
                self._bk_wx, self._bk_wy = self.wx, self.wy
                self.state = State.BACKUP
                return

            # Giảm tốc khi gần đích ô
            ratio = max(LIN_SPD_MIN / LIN_SPD, min(1.0, dist / (CELL * 0.5)))
            speed = LIN_SPD * ratio
            self._move_to(tx, ty, speed)

        #BACKUP: lùi sau khi bị phanh 
        elif self.state == State.BACKUP:
            backed = math.hypot(self.wx - self._bk_wx, self.wy - self._bk_wy)
            if backed >= BACKUP_DIST:
                self._stop()
                self.get_logger().info('[BACKUP] Xong → quét lại + BFS mới')
                self._settle = 0
                self.state   = State.SCAN
            else:
                t = Twist()
                t.linear.x = -0.10
                self.pub.publish(t)

        #DONE 
        elif self.state == State.DONE:
            self._stop()

    #ASCII maze ──
    def _print_maze(self):
        ps         = set(self.path)
        start_cell = self.visited[0] if self.visited else None

        def wchar(val):
            if val is True:  return '│'
            if val is False: return ' '
            return '░'

        def hchar(val):
            if val is True:  return '──'
            if val is False: return '  '
            return '░░'

        lines = ['']
        lines.append('  ┌' + '──┬' * (COLS - 1) + '──┐')

        for r in range(ROWS):
            row = '  '
            for c in range(COLS):
                row += wchar(self.walls[(r, c)][2])
                if   (r, c) == self.cur:          row += 'R '
                elif (r, c) == self.goal:         row += 'G '
                elif (r, c) == start_cell:        row += 'S '
                elif (r, c) in ps:                row += '* '
                elif (r, c) in self.visited_set:  row += '· '
                else:                             row += '  '
            row += wchar(self.walls[(r, COLS - 1)][0])
            lines.append(row)

            if r < ROWS - 1:
                sep = '  ├'
                for c in range(COLS):
                    sep += hchar(self.walls[(r, c)][3])
                    sep += '┼' if c < COLS - 1 else '┤'
                lines.append(sep)

        lines.append('  └' + '──┴' * (COLS - 1) + '──┘')
        lines.append(f'  R={self.cur}  G={self.goal}  step={self.step}')
        lines.append(f'  path_planned={[self.cur] + self.path}')
        lines.append(f'  visited_history={self.visited}')
        for line in lines:
            self.get_logger().info(line)

#  ENTRY
def main(args=None):
    rclpy.init(args=args)
    node = MazeSolverV6()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node._stop()
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()