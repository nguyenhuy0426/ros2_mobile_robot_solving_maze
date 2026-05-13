#!/usr/bin/env python3
"""maze_solver.py - DFS + LiDAR | 5x5 grid 0.5m/cell | nhom8_mecanum"""

import rclpy, math
import numpy as np
from rclpy.node import Node
from collections import defaultdict
from enum import Enum, auto
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy

# ── CONFIG ──────────────────────────────────────────────────────────────────
ROWS, COLS   = 5, 5
CELL         = 0.50
START        = (2, 0)      # ô (2,0) 
GOAL         = (2, 4)      # ô (2,4) 

LIN_SPD      = 0.50        # m/s - tốc độ di chuyển thẳng
ANG_MAX      = 1.20        # rad/s - tốc độ quay tối đa
KP_FWD       = 3.5         # hệ số PID giữ hướng khi đi thẳng
KP_TURN      = 6.0         # hệ số PID quay tại chỗ
YAW_TOL      = 0.04        # rad - sai số yaw chấp nhận được

# ARRIVE dùng khoảng cách tương đối từ điểm xuất phát (không dùng world frame)
# Robot phải di chuyển >= ARRIVE_DIST mới được tính là đã vào ô mới
ARRIVE_DIST  = 0.43        # m = 86% cell, đảm bảo robot qua giữa ô mới

WALL_DIST    = 0.26        # m - LiDAR < threshold = có tường
BRAKE_DIST   = 0.14        # m - cự ly phanh khẩn cấp
LIDAR_HALF   = 20          # ° - nửa góc quét mỗi hướng
SCAN_SETTLE  = 8           # ticks chờ LiDAR ổn định
SCAN_AVG     = 4           # số scan lấy trung bình
BACKUP_DIST  = 0.12        # m - lùi sau khi brake
BACKUP_SPD   = 0.10        # m/s - tốc độ lùi

# Hướng: 0=E(Đông) 1=N(Bắc) 2=W(Tây) 3=S(Nam)
DNAME = ['E', 'N', 'W', 'S']
DYAW  = [0.0, math.pi/2, math.pi, -math.pi/2]
DDR   = [0, -1, 0, +1]   # delta row
DDC   = [+1, 0, -1, 0]   # delta col

class St(Enum):
    SCAN=auto(); PLAN=auto(); TURN=auto(); MOVE=auto(); BACK=auto(); DONE=auto()

def wrap(a):
    while a >  math.pi: a -= 2*math.pi
    while a < -math.pi: a += 2*math.pi
    return a

# ── NODE ────────────────────────────────────────────────────────────────────
class MazeSolver(Node):

    def __init__(self):
        super().__init__('maze_solver')
        self.declare_parameter('robot_id', 1)
        rid = self.get_parameter('robot_id').value

        qos = QoSProfile(depth=5,
                         reliability=QoSReliabilityPolicy.BEST_EFFORT,
                         history=QoSHistoryPolicy.KEEP_LAST)
        self.pub = self.create_publisher(Twist, f'/model/robot_{rid}/cmd_vel', 10)
        self.create_subscription(LaserScan, f'/scan{rid}',                  self._on_scan, qos)
        self.create_subscription(Odometry,  f'/model/robot_{rid}/odometry', self._on_odom, qos)

        # --- sensor state ---
        self._raw   = np.array([])
        self._sbuf  = []        # rolling buffer (scan_avg x 4 directions)
        self._scnt  = 0
        self._ook   = False
        self.wx = self.wy = self.yaw = 0.0

        # --- maze state ---
        # walls[(r,c)][d]: True=tường, False=mở, None=chưa biết
        # Quy tắc: KHÔNG BAO GIỜ ghi đè True bằng False
        self.walls   = defaultdict(lambda: {d: None for d in range(4)})
        self.visited = {START}
        self.stack   = []       # DFS backtrack stack
        self.cur     = START
        self.tgt     = None
        self.tdir    = 0
        self.going_back = False

        # --- control state ---
        self.state   = St.SCAN
        self._sctick = 0
        self._move_ox = self._move_oy = 0.0  # odom khi bắt đầu MOVE (relative arrive)
        self._bkx = self._bky = 0.0           # odom khi bắt đầu BACK
        self.step    = 0
        self._brake_count = 0                  # đếm brake liên tiếp cùng hướng

        self.get_logger().info(
            f'╔══════════════════════════════════════════════╗\n'
            f'║  MazeSolver DFS | cell={CELL}m | speed={LIN_SPD}m/s\n'
            f'║  START={START}  GOAL={GOAL}\n'
            f'╚══════════════════════════════════════════════╝')
        self.create_timer(0.05, self._loop)

    # ── CALLBACKS ────────────────────────────────────────────────────────────

    def _on_scan(self, msg):
        r = np.array(msg.ranges, dtype=np.float32)
        self._raw  = np.where(np.isfinite(r) & (r > msg.range_min), r, msg.range_max)
        self._scnt += 1
        if len(self._raw):
            self._sbuf.append(tuple(self._sector(DYAW[d]) for d in range(4)))
            if len(self._sbuf) > SCAN_AVG * 4:
                self._sbuf.pop(0)

    def _on_odom(self, msg):
        p = msg.pose.pose.position
        q = msg.pose.pose.orientation
        self.wx  = p.x
        self.wy  = p.y
        self.yaw = math.atan2(2*(q.w*q.z + q.x*q.y), 1 - 2*(q.y**2 + q.z**2))
        self._ook = True

    # ── LIDAR ────────────────────────────────────────────────────────────────

    def _sector(self, world_yaw):
        """Khoảng cách nhỏ nhất trong cung ±LIDAR_HALF° theo hướng world_yaw."""
        if not len(self._raw):
            return float('inf')
        n   = len(self._raw)
        rel = wrap(world_yaw - self.yaw)
        deg = math.degrees(rel) % 360
        ci  = int(deg / 360 * n) % n
        hw  = max(1, int(LIDAR_HALF / 360 * n))
        idx = np.array([(ci + k) % n for k in range(-hw, hw+1)])
        return float(np.min(self._raw[idx]))

    def _avg_walls(self):
        """Trung bình SCAN_AVG lần đọc → dict {dir: bool}."""
        if len(self._sbuf) < SCAN_AVG:
            return {d: self._sector(DYAW[d]) < WALL_DIST for d in range(4)}
        recent = self._sbuf[-SCAN_AVG:]
        return {d: float(np.mean([x[d] for x in recent])) < WALL_DIST
                for d in range(4)}

    # ── MOTION ───────────────────────────────────────────────────────────────

    def _stop(self):
        self.pub.publish(Twist())

    def _cmd(self, vx, wz):
        t = Twist()
        t.linear.x  = float(vx)
        t.angular.z = float(wz)
        self.pub.publish(t)

    # ── WALL MERGE ───────────────────────────────────────────────────────────

    def _merge_walls(self, scanned):
        """
        Merge kết quả scan vào knowledge base.
        Quy tắc: True (tường) không bao giờ bị ghi đè.
        """
        for d in range(4):
            cur_val = self.walls[self.cur][d]
            if cur_val == True:
                continue                       # đã biết là tường, giữ nguyên
            if scanned[d] == True:
                self.walls[self.cur][d] = True  # phát hiện tường mới
            elif cur_val is None:
                self.walls[self.cur][d] = False  # lần đầu scan, ghi mở

    # ── DFS ──────────────────────────────────────────────────────────────────

    def _nb(self, r, c):
        return [(d, (r+DDR[d], c+DDC[d]))
                for d in range(4)
                if 0 <= r+DDR[d] < ROWS and 0 <= c+DDC[d] < COLS]

    def _dfs_next(self):
        """
        Trả về (dir, target_cell, is_backtrack) hoặc None nếu bế tắc.
        Ưu tiên: open confirmed → unknown → backtrack.
        Không quay về START.
        """
        w = self.walls[self.cur]

        # 1) Tiến: ô chưa thăm, tường đã xác nhận mở
        for d, nb in self._nb(*self.cur):
            if nb not in self.visited and w[d] == False and nb != START:
                return d, nb, False

        # 2) Tiến: ô chưa thăm, tường chưa biết (thử xem)
        for d, nb in self._nb(*self.cur):
            if nb not in self.visited and w[d] is None and nb != START:
                return d, nb, False

        # 3) Backtrack về ô trước trong stack
        if self.stack:
            prev = self.stack[-1]
            for d, nb in self._nb(*self.cur):
                if nb == prev and w[d] != True:
                    return d, nb, True

        return None

    # ── TRANSITIONS ──────────────────────────────────────────────────────────

    def _to_scan(self):
        self._sctick = 0
        self._sbuf.clear()
        self.state = St.SCAN

    def _arrive(self):
        """Xử lý khi robot đến ô mới."""
        if self.going_back:
            if self.stack:
                self.stack.pop()
        else:
            self.stack.append(self.cur)
        self.cur = self.tgt
        self.visited.add(self.cur)
        self._brake_count = 0
        self.get_logger().info(
            f'[ARRIVE] → {self.cur}  step={self.step}  '
            f'visited={len(self.visited)}/{ROWS*COLS}  stack_d={len(self.stack)}')

    # ── MAIN LOOP ─────────────────────────────────────────────────────────────

    def _loop(self):
        if not self._ook or self._scnt < 5:
            return

        # ── SCAN ──────────────────────────────────────────────────────────────
        if self.state == St.SCAN:
            self._stop()
            self._sctick += 1
            if self._sctick < SCAN_SETTLE:
                return
            w = self._avg_walls()
            self._merge_walls(w)
            # Truyền thông tin tường sang ô hàng xóm
            for d, nb in self._nb(*self.cur):
                opp = (d + 2) % 4
                if self.walls[nb][opp] is None:
                    self.walls[nb][opp] = self.walls[self.cur][d]
            ws = '  '.join(
                f'{DNAME[d]}:{"█" if self.walls[self.cur][d] else ("?" if self.walls[self.cur][d] is None else "·")}'
                for d in range(4))
            self.get_logger().info(
                f'[SCAN]  cell={self.cur}  [{ws}]  visited={len(self.visited)}/{ROWS*COLS}')
            self._print_maze()
            self.state = St.PLAN

        # ── PLAN ──────────────────────────────────────────────────────────────
        elif self.state == St.PLAN:
            if self.cur == GOAL:
                self._stop()
                self.get_logger().info(f'[DONE] ✓ GOAL reached!  total_steps={self.step}')
                self._print_maze()
                self.state = St.DONE
                return

            result = self._dfs_next()

            if result is None:
                # Bế tắc: tất cả hướng đều là tường hoặc đã thăm
                self.get_logger().error(
                    f'[STUCK] cell={self.cur}  walls={dict(self.walls[self.cur])}  stack={self.stack}')
                if self.stack:
                    dropped = self.stack.pop()
                    self.get_logger().warn(f'[STUCK] force-pop {dropped} → re-scan')
                    self._to_scan()
                else:
                    self.get_logger().error('[STUCK] stack rỗng — mê cung không giải được!')
                return

            d, tgt, going_back = result
            self.tdir       = d
            self.tgt        = tgt
            self.going_back = going_back
            self.step      += 1

            arrow = '↩' if going_back else '→'
            self.get_logger().info(
                f'[PLAN]  step={self.step:3d}  {self.cur}{arrow}{tgt}  '
                f'dir={DNAME[d]}  back={going_back}')
            self.state = St.TURN

        # ── TURN ──────────────────────────────────────────────────────────────
        elif self.state == St.TURN:
            err = wrap(DYAW[self.tdir] - self.yaw)
            if abs(err) < YAW_TOL:
                self._stop()
                # Ghi điểm xuất phát để tính ARRIVE theo khoảng cách tương đối
                self._move_ox = self.wx
                self._move_oy = self.wy
                self.get_logger().info(
                    f'[TURN]  ✓ aligned {DNAME[self.tdir]}  yaw={math.degrees(self.yaw):.1f}°')
                self.state = St.MOVE
            else:
                wz = float(np.clip(KP_TURN * err, -ANG_MAX, ANG_MAX))
                self._cmd(0.0, wz)

        # ── MOVE ──────────────────────────────────────────────────────────────
        elif self.state == St.MOVE:
            # Khoảng cách đã di chuyển từ điểm xuất phát (relative, không cần world frame)
            moved = math.hypot(self.wx - self._move_ox, self.wy - self._move_oy)

            # ① Đã đến giữa ô mới → ARRIVE
            if moved >= ARRIVE_DIST:
                self._stop()
                self._arrive()
                if self.cur == GOAL:
                    self.get_logger().info(f'[DONE] ✓ GOAL reached!  total_steps={self.step}')
                    self._print_maze()
                    self.state = St.DONE
                else:
                    self._to_scan()
                return

            # ② Vật cản phía trước → phanh khẩn cấp
            front = self._sector(self.yaw)
            if front < BRAKE_DIST:
                self._stop()
                self._brake_count += 1
                self.get_logger().warn(
                    f'[BRAKE] front={front:.3f}m  moved={moved:.3f}m  '
                    f'brake_count={self._brake_count}  '
                    f'→ wall {DNAME[self.tdir]} from {self.cur}')
                # Đánh dấu tường cả 2 phía (không bao giờ bị overwrite)
                self.walls[self.cur][self.tdir] = True
                self.walls[self.tgt][(self.tdir+2) % 4] = True
                self._bkx = self.wx
                self._bky = self.wy
                self.state = St.BACK
                return

            # ③ Đi thẳng với hiệu chỉnh yaw
            err_yaw = wrap(DYAW[self.tdir] - self.yaw)
            # Tăng tốc dần khi bắt đầu, duy trì tốc độ ổn định
            frac = min(1.0, moved / 0.15 + 0.5)
            vx   = LIN_SPD * max(0.5, frac)
            wz   = float(np.clip(KP_FWD * err_yaw, -0.6, 0.6))
            self._cmd(vx, wz)

        # ── BACK ──────────────────────────────────────────────────────────────
        elif self.state == St.BACK:
            backed = math.hypot(self.wx - self._bkx, self.wy - self._bky)
            if backed >= BACKUP_DIST:
                self._stop()
                self.get_logger().info(
                    f'[BACK]  ✓ backed={backed:.3f}m  → PLAN (wall {DNAME[self.tdir]} kept)')
                # ✓ Chuyển thẳng sang PLAN — KHÔNG scan lại để tránh ghi đè tường
                self.state = St.PLAN
            else:
                self._cmd(-BACKUP_SPD, 0.0)

        # ── DONE ──────────────────────────────────────────────────────────────
        elif self.state == St.DONE:
            self._stop()

    # ── DEBUG ASCII MAZE ─────────────────────────────────────────────────────

    def _print_maze(self):
        def wchar(r, c, d):
            v = self.walls[(r, c)][d]
            return True if v else False

        lines = ['']
        # Top border
        top = '  ┌'
        for c in range(COLS):
            top += '──' + ('┬' if c < COLS-1 else '┐')
        lines.append(top)

        for r in range(ROWS):
            # Cell row
            row = '  '
            for c in range(COLS):
                row += '│' if wchar(r, c, 2) else ' '
                if   (r,c) == self.cur:     row += 'R '
                elif (r,c) == GOAL:         row += 'G '
                elif (r,c) == START:        row += 'S '
                elif (r,c) in self.visited: row += '· '
                else:                       row += '  '
            row += '│' if wchar(r, COLS-1, 0) else ' '
            lines.append(row)

            # Horizontal separator between rows
            if r < ROWS - 1:
                sep = '  ├'
                for c in range(COLS):
                    sep += '──' if wchar(r, c, 3) else '  '
                    sep += '┼' if c < COLS-1 else '┤'
                lines.append(sep)

        # Bottom border
        bot = '  └'
        for c in range(COLS):
            bot += '──' + ('┴' if c < COLS-1 else '┘')
        lines.append(bot)

        lines.append(
            f'  stack_path={self.stack}\n'
            f'  odom=({self.wx:.2f},{self.wy:.2f})  yaw={math.degrees(self.yaw):.0f}°  '
            f'step={self.step}  state={self.state.name}')
        for l in lines:
            self.get_logger().info(l)


# ── ENTRY ────────────────────────────────────────────────────────────────────

def main(args=None):
    rclpy.init(args=args)
    node = MazeSolver()
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