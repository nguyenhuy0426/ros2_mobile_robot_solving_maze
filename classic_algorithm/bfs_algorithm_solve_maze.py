#!/usr/bin/env python3
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

# ── CẤU HÌNH ────────────────────────────────────────────────────────────────
ROWS, COLS   = 5, 5
CELL         = 0.5
START_W      = (2.25, -3.25)
GOAL_W       = (2.25, -0.25)
START_CELL   = (2, 0)        # ô xuất phát (row, col)
GOAL_CELL    = (2, 4)        # ô đích (row, col)

# Hướng: 0=E  1=N  2=W  3=S
DNAME = ['E', 'N', 'W', 'S']
DYAW  = [0.0,  math.pi/2, math.pi, -math.pi/2]
DDR   = [ 0,   -1,  0,  +1]   # delta row theo hướng
DDC   = [+1,    0, -1,   0]   # delta col theo hướng
OPP   = [ 2,    3,  0,   1]   # hướng ngược lại

LIN_SPD      = 0.2
LIN_SPD_MIN  = 0.2
KP_ANG       = 4.5            # PID giữ hướng khi di chuyển
ANG_MAX      = 1.2            # rad/s tối đa
ARRIVE_R     = 0.055          # m — bán kính "đã đến" tâm ô

LIDAR_HALF_DEG = 20           # ° — nửa góc quét mỗi hướng
WALL_THRESH    = 0.27         # m — LiDAR < giá trị này → có tường
BRAKE_DIST     = 0.2          # m — phanh khẩn cấp
SCAN_SETTLE    = 10           # ticks chờ LiDAR ổn định (10×50ms = 0.5s)
BACKUP_DIST    = 0.1          # m — lùi sau khi phanh

# Vòng đời trạng thái: WAIT → SCAN → PLAN → MOVE → (BACKUP →) SCAN → ... → DONE
class State(Enum):
    WAIT   = auto()
    SCAN   = auto()
    PLAN   = auto()
    MOVE   = auto()
    BACKUP = auto()
    DONE   = auto()

def wrap(a):
    # Chuẩn hoá góc về [-π, π]
    while a >  math.pi: a -= 2 * math.pi
    while a < -math.pi: a += 2 * math.pi
    return a

def in_bounds(r, c):
    return 0 <= r < ROWS and 0 <= c < COLS

def c2w(r, c, ox, oy):
    # Chuyển (row, col) → tọa độ world tâm ô
    return ox + (c + 0.5) * CELL, oy - (r + 0.5) * CELL

def w2c(x, y, ox, oy):
    # Chuyển tọa độ world → (row, col)
    col = int(round((x - ox) / CELL - 0.5))
    row = int(round(-(y - oy) / CELL - 0.5))
    return (row, col)

# ── NODE ────────────────────────────────────────────────────────────────────
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

        # Dữ liệu cảm biến
        self._scan    = np.array([])
        self._odom_ok = False
        self.wx = self.wy = self.yaw = 0.0

        # Origin lưới — tính từ vị trí spawn thực tế (calibrate lần đầu)
        self._ox = START_W[0] - (START_CELL[1] + 0.5) * CELL
        self._oy = START_W[1] + (START_CELL[0] + 0.5) * CELL

        # Bản đồ tường: walls[(r,c)][d] = True/False/None
        self.walls = {(r, c): {d: None for d in range(4)}
                      for r in range(ROWS) for c in range(COLS)}
        self._set_border_walls()  # viền ngoài luôn là tường

        self.cur  = START_CELL
        self.goal = GOAL_CELL
        self.path = []            # BFS path hiện tại (danh sách ô còn lại)

        # Lịch sử thăm — dùng để debug và in bản đồ
        self.visited     = [self.cur]
        self.visited_set = {self.cur}

        # Trạng thái điều khiển
        self.state    = State.WAIT
        self._settle  = 0
        self._tgt     = self.cur
        self._tgt_dir = 0
        self._tgt_w   = c2w(*self.cur, self._ox, self._oy)
        self._bk_wx   = 0.0       # vị trí odom khi bắt đầu BACKUP
        self._bk_wy   = 0.0
        self.step     = 0

        self.get_logger().info(
            f'  maze_solver_v6  —  LiDAR-first + BFS\n'
            f'  START={self.cur}  GOAL={self.goal}\n'
            f'  CELL={CELL}m  WALL_THRESH={WALL_THRESH}m\n'
            f'  Sector: ±{LIDAR_HALF_DEG}° (~{2*LIDAR_HALF_DEG} deg / sector)\n'
        )
        self.create_timer(0.05, self._loop)

    # ── BORDER ───────────────────────────────────────────────────────────────

    def _set_border_walls(self):
        # Gán tường cố định cho 4 cạnh ngoài lưới
        for r in range(ROWS):
            for c in range(COLS):
                if r == 0:        self.walls[(r, c)][1] = True   # Bắc
                if r == ROWS - 1: self.walls[(r, c)][3] = True   # Nam
                if c == 0:        self.walls[(r, c)][2] = True   # Tây
                if c == COLS - 1: self.walls[(r, c)][0] = True   # Đông

    # ── CALLBACKS ────────────────────────────────────────────────────────────

    def _cb_scan(self, msg: LaserScan):
        raw = np.array(msg.ranges, dtype=np.float32)
        # Lọc giá trị lỗi, thay bằng 12.0 m
        self._scan = np.where(
            np.isfinite(raw) & (raw > 0.05) & (raw < 12.0), raw, 12.0)

    def _cb_odom(self, msg: Odometry):
        p = msg.pose.pose.position
        q = msg.pose.pose.orientation
        self.wx  = p.x
        self.wy  = p.y
        # Chuyển quaternion → yaw
        self.yaw = math.atan2(
            2.0 * (q.w * q.z + q.x * q.y),
            1.0 - 2.0 * (q.y ** 2 + q.z ** 2))
        if not self._odom_ok:
            # Lần đầu nhận odom → calibrate origin từ vị trí spawn thực tế
            self._odom_ok = True
            self._ox = self.wx - (START_CELL[1] + 0.5) * CELL
            self._oy = self.wy + (START_CELL[0] + 0.5) * CELL
            self._tgt_w = c2w(*self.cur, self._ox, self._oy)
            self.get_logger().info(
                f'[CALIB] spawn=({self.wx:.4f},{self.wy:.4f})  '
                f'OX={self._ox:.4f} OY={self._oy:.4f}  '
                f'START_world={c2w(*START_CELL, self._ox, self._oy)}  '
                f'GOAL_world={c2w(*GOAL_CELL, self._ox, self._oy)}')

    # ── LIDAR ────────────────────────────────────────────────────────────────

    def _sector_min(self, world_yaw):
        # Khoảng cách nhỏ nhất trong cung ±LIDAR_HALF_DEG° theo world_yaw
        if not len(self._scan):
            return 12.0
        n   = len(self._scan)
        rel = wrap(world_yaw - self.yaw)
        ci  = int((math.degrees(rel) % 360) / 360 * n) % n
        hw  = max(1, int(LIDAR_HALF_DEG / 360 * n))
        idx = np.array([(ci + k) % n for k in range(-hw, hw + 1)])
        return float(np.min(self._scan[idx]))

    def _scan_walls_at_cur(self):
        # Quét LiDAR 4 hướng, cập nhật tường cho ô hiện tại và ô hàng xóm
        readings = {}
        for d in range(4):
            dist = self._sector_min(DYAW[d])
            has_wall = bool(dist < WALL_THRESH)
            readings[d] = (dist, has_wall)
            self.walls[self.cur][d] = has_wall
            # Truyền thông tin tường sang ô hàng xóm (cạnh chung)
            nr, nc = self.cur[0] + DDR[d], self.cur[1] + DDC[d]
            if in_bounds(nr, nc):
                self.walls[(nr, nc)][OPP[d]] = has_wall

        log = '  '.join(
            f'{DNAME[d]}={readings[d][0]:.3f}m{"█" if readings[d][1] else "·"}'
            for d in range(4))
        self.get_logger().info(f'[LIDAR] cell={self.cur}  yaw={math.degrees(self.yaw):.0f}°  {log}')

    # ── BFS ──────────────────────────────────────────────────────────────────

    def _bfs(self, src, dst):
        """
        BFS trên bản đồ đã biết — tìm đường ngắn nhất từ src đến dst.
        Ô chưa biết (wall=None) được coi là đi được (optimistic).
        Trả về danh sách ô cần đi (không gồm src), hoặc None nếu bế tắc.
        """
        q    = deque([(src, [src])])
        seen = {src}
        while q:
            (r, c), path = q.popleft()
            for d in range(4):
                nr, nc = r + DDR[d], c + DDC[d]
                nb = (nr, nc)
                if not in_bounds(nr, nc) or nb in seen:
                    continue
                if self.walls[(r, c)][d] is True:   # bỏ qua nếu có tường
                    continue
                seen.add(nb)
                new_path = path + [nb]
                if nb == dst:
                    return new_path[1:]  # trả về path bỏ qua src
                q.append((nb, new_path))
        return None  # không tìm được đường

    # ── MOTION ───────────────────────────────────────────────────────────────

    def _stop(self):
        self.pub.publish(Twist())

    def _move_to(self, tx, ty, speed):
        # Mecanum drive: di chuyển thẳng đến (tx,ty) không cần quay trước
        dx, dy = tx - self.wx, ty - self.wy
        dist   = math.hypot(dx, dy)
        if dist < 0.002:
            self._stop()
            return

        target_yaw = math.atan2(dy, dx)
        ux, uy     = dx / dist, dy / dist

        # Chuyển vector world-frame → robot-frame
        cos_y = math.cos(self.yaw)
        sin_y = math.sin(self.yaw)
        vxr   =  cos_y * ux * speed + sin_y * uy * speed
        vyr   = -sin_y * ux * speed + cos_y * uy * speed
        # PID giữ heading để LiDAR luôn canh đúng hướng ô đích
        wz = float(np.clip(KP_ANG * wrap(target_yaw - self.yaw), -ANG_MAX, ANG_MAX))

        t = Twist()
        t.linear.x  = float(vxr)
        t.linear.y  = float(vyr)
        t.angular.z = wz
        self.pub.publish(t)

    def _register_arrive(self):
        # Ghi nhận đến ô mới vào lịch sử thăm
        if self.cur not in self.visited_set:
            self.visited.append(self.cur)
            self.visited_set.add(self.cur)
        wx_c, wy_c = c2w(*self.cur, self._ox, self._oy)
        self.get_logger().info(
            f'[ARRIVE] step={self.step:3d}  cell={self.cur}  '
            f'world=({self.wx:.3f},{self.wy:.3f})  '
            f'cell_center=({wx_c:.3f},{wy_c:.3f})\n'
            f'[HISTORY] visited={self.visited}')

    # ── MAIN LOOP ─────────────────────────────────────────────────────────────

    def _loop(self):
        if not self._odom_ok or not len(self._scan):
            return

        # ── WAIT: chờ cảm biến sẵn sàng, chuyển sang SCAN ───────────────────
        if self.state == State.WAIT:
            self._stop()
            self.state = State.SCAN

        # ── SCAN: dừng xe, đợi ổn định, quét tường, chạy BFS ────────────────
        elif self.state == State.SCAN:
            self._stop()
            self._settle += 1
            if self._settle < SCAN_SETTLE:
                return

            self._scan_walls_at_cur()
            self._print_maze()

            # Chạy BFS để tìm đường ngắn nhất đến GOAL
            self.path = self._bfs(self.cur, self.goal)
            if self.path:
                self.get_logger().info(f'[BFS ✓] {[self.cur] + self.path}  ({len(self.path)} bước)')
            else:
                self.get_logger().error(f'[BFS ✗] Không tìm được đường từ {self.cur} đến {self.goal}!')

            self._settle = 0
            self.state   = State.PLAN

        # ── PLAN: chọn ô tiếp theo từ BFS path, tính hướng và tọa độ đích ───
        elif self.state == State.PLAN:
            if self.cur == self.goal:
                self._stop()
                self.get_logger().info(
                    f'  ĐẾN ĐÍCH!  steps={self.step}\n'
                    f'  LỊCH SỬ ĐI: {self.visited}\n')
                self._print_maze()
                self.state = State.DONE
                return

            if not self.path:
                self.get_logger().error(f'[BẾ TẮC] Không có đường từ {self.cur}')
                self._stop()
                self.state = State.DONE
                return

            # Lấy ô kế tiếp trong path, xác định hướng di chuyển
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

        # ── MOVE: mecanum drive đến tâm ô, kiểm tra ARRIVE và phanh ─────────
        elif self.state == State.MOVE:
            tx, ty = self._tgt_w
            dist   = math.hypot(tx - self.wx, ty - self.wy)

            # ① Đã vào bán kính ARRIVE_R → coi là đến ô, quét lại
            if dist < ARRIVE_R:
                self._stop()
                self.cur = self._tgt
                self._register_arrive()
                self._settle = 0
                self.state   = State.SCAN
                return

            # ② LiDAR phát hiện vật cản → phanh, đánh dấu tường, lùi
            front = self._sector_min(DYAW[self._tgt_dir])
            if front < BRAKE_DIST:
                self._stop()
                self.get_logger().warn(
                    f'[BRAKE] dir={DNAME[self._tgt_dir]}  front={front:.3f}m < {BRAKE_DIST}m  '
                    f'→ đánh dấu tường tại {self.cur}[{DNAME[self._tgt_dir]}]')
                # Ghi tường cả 2 phía ô (không bao giờ bị ghi đè)
                self.walls[self.cur][self._tgt_dir] = True
                nr, nc = self.cur[0] + DDR[self._tgt_dir], self.cur[1] + DDC[self._tgt_dir]
                if in_bounds(nr, nc):
                    self.walls[(nr, nc)][OPP[self._tgt_dir]] = True
                self.path  = []   # xoá path cũ, BFS sẽ tính lại sau BACKUP
                self._bk_wx, self._bk_wy = self.wx, self.wy
                self.state = State.BACKUP
                return

            # ③ Giảm tốc khi gần tâm ô, đi thẳng bằng mecanum
            ratio = max(LIN_SPD_MIN / LIN_SPD, min(1.0, dist / (CELL * 0.5)))
            speed = LIN_SPD * ratio
            self._move_to(tx, ty, speed)

        # ── BACKUP: lùi sau phanh, sau đó quét lại + BFS mới ─────────────────
        elif self.state == State.BACKUP:
            backed = math.hypot(self.wx - self._bk_wx, self.wy - self._bk_wy)
            if backed >= BACKUP_DIST:
                self._stop()
                self.get_logger().info('[BACKUP] Xong → quét lại + BFS mới')
                self._settle = 0
                self.state   = State.SCAN  # quét lại để BFS tính đường mới
            else:
                t = Twist()
                t.linear.x = -0.10
                self.pub.publish(t)

        # ── DONE ──────────────────────────────────────────────────────────────
        elif self.state == State.DONE:
            self._stop()

    # ── DEBUG ASCII MAZE ─────────────────────────────────────────────────────

    def _print_maze(self):
        ps         = set(self.path)
        start_cell = self.visited[0] if self.visited else None

        def wchar(val):
            if val is True:  return '│'
            if val is False: return ' '
            return '░'   # chưa biết

        def hchar(val):
            if val is True:  return '──'
            if val is False: return '  '
            return '░░'  # chưa biết

        lines = ['']
        lines.append('  ┌' + '──┬' * (COLS - 1) + '──┐')

        for r in range(ROWS):
            row = '  '
            for c in range(COLS):
                row += wchar(self.walls[(r, c)][2])
                if   (r, c) == self.cur:          row += 'R '   # robot
                elif (r, c) == self.goal:         row += 'G '   # đích
                elif (r, c) == start_cell:        row += 'S '   # xuất phát
                elif (r, c) in ps:                row += '* '   # path BFS
                elif (r, c) in self.visited_set:  row += '· '   # đã thăm
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

# ── ENTRY ────────────────────────────────────────────────────────────────────

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