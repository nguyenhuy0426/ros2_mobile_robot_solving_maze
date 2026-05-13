# CLAUDE.md — AI Assistant Context

File này cung cấp ngữ cảnh cho AI assistant (Claude) khi làm việc với codebase `classic_algorithm`.

---

## Tổng quan dự án

Dự án giải mê cung 5×5 trong Gazebo Harmonic + ROS2 Jazzy bằng robot mecanum bánh toàn hướng. Hai thuật toán được triển khai song song: **DFS online** và **BFS online**, cả hai đều khám phá bản đồ thực tế qua LiDAR (không biết trước bản đồ).

**Môi trường:** Ubuntu 24.04, ROS2 Jazzy, Gazebo Harmonic, Python 3.10+, NumPy.

---

## Cấu trúc file quan trọng

```
classic_algorithm/
├── nhom8_maze.sdf                   # World Gazebo — mê cung 5×5
├── one_robot.sdf                    # Template robot (placeholder ROBOT_ID)
├── spawn_robot.sh                   # Spawn N robots + bridge + TF
├── dfs_algorithm_solve_maze.py      # Solver DFS
└── bfs_algorithm_solve_maze.py      # Solver BFS (class MazeSolverV6)
```

---

## Quy ước tọa độ

### Lưới logic (row, col)

```
(0,0) ── (0,4)
  │          │      Hàng tăng → Nam (y giảm trong world)
  │          │      Cột tăng  → Đông (x tăng trong world)
(4,0) ── (4,4)
```

### Hướng

```python
DNAME = ['E', 'N', 'W', 'S']       # index: 0=E, 1=N, 2=W, 3=S
DYAW  = [0.0, π/2, π, -π/2]        # world yaw tương ứng
DDR   = [0, -1,  0, +1]            # delta row
DDC   = [+1,  0, -1,  0]           # delta col
OPP   = [2,   3,  0,  1]           # hướng ngược lại (BFS)
```

### Chuyển đổi cell ↔ world (BFS)

```python
def c2w(r, c, ox, oy):
    return ox + (c + 0.5) * CELL, oy - (r + 0.5) * CELL

def w2c(x, y, ox, oy):
    col = int(round((x - ox) / CELL - 0.5))
    row = int(round(-(y - oy) / CELL - 0.5))
    return (row, col)
```

Origin `(ox, oy)` được calibrate từ vị trí spawn thực tế lần đầu nhận odometry.

---

## Cấu hình hai solver

### BFS — `MazeSolverV6`

```python
ROWS, COLS   = 5, 5
CELL         = 0.5           # m
START_CELL   = (2, 0)
GOAL_CELL    = (2, 4)
START_W      = (2.25, -3.25) # world coords START

LIN_SPD      = 0.2           # m/s
ARRIVE_R     = 0.055         # m — bán kính đã đến tâm ô
WALL_THRESH  = 0.27          # m — LiDAR < giá trị này → tường
BRAKE_DIST   = 0.2           # m — phanh khẩn cấp
SCAN_SETTLE  = 10            # ticks × 50ms = 0.5s chờ ổn định
BACKUP_DIST  = 0.1           # m — lùi sau phanh
LIDAR_HALF_DEG = 20          # ° — nửa góc quét mỗi sector
```

**Vòng đời state machine:**
```
WAIT → SCAN → PLAN → MOVE → BACKUP → SCAN (lặp lại) → DONE
```

### DFS — `MazeSolver`

```python
ROWS, COLS   = 5, 5
CELL         = 0.50
START        = (2, 0)
GOAL         = (2, 4)

LIN_SPD      = 0.50          # m/s (nhanh hơn BFS)
ARRIVE_DIST  = 0.43          # m — 86% cell (tính tương đối từ điểm xuất phát)
WALL_DIST    = 0.26          # m — ngưỡng tường
BRAKE_DIST   = 0.14          # m — phanh
SCAN_AVG     = 4             # số scan lấy trung bình (giảm nhiễu)
YAW_TOL      = 0.04          # rad — sai số yaw chấp nhận
```

**Vòng đời state machine:**
```
SCAN → PLAN → TURN → MOVE → BACK → SCAN (lặp lại) → DONE
```

---

## Bản đồ tường (Wall Map)

Cả hai solver dùng cấu trúc:

```python
walls[(r, c)][d]  # d = 0..3
# None  = chưa biết (chưa scan)
# True  = có tường (không bao giờ bị ghi đè bởi False)
# False = mở
```

**Quy tắc quan trọng:** `True` (tường) là bất biến — không bao giờ bị overwrite, dù scan sau cho kết quả khác.

Khi scan tại ô `(r,c)` hướng `d`, kết quả được truyền sang ô hàng xóm:
```python
walls[(nr, nc)][OPP[d]] = walls[(r, c)][d]
```

---

## Thuật toán core

### BFS (`_bfs`)

```python
def _bfs(self, src, dst):
    # BFS trên wall map hiện tại
    # wall=None  → đi được (optimistic)
    # wall=True  → chặn
    # wall=False → đi được
    # Trả về list ô từ src+1 đến dst, hoặc None
```

BFS chạy lại **mỗi lần** sau khi scan tại ô mới → luôn tìm đường ngắn nhất trên bản đồ cập nhật nhất.

### DFS (`_dfs_next`)

```python
def _dfs_next(self):
    # Trả về (dir, target_cell, is_backtrack) hoặc None
    # Ưu tiên 1: ô chưa thăm + wall=False
    # Ưu tiên 2: ô chưa thăm + wall=None (thử)
    # Ưu tiên 3: backtrack → stack[-1]
```

Stack DFS lưu lộ trình để backtrack vật lý (robot thực sự di chuyển ngược lại).

---

## LiDAR

**Phần cứng:** 36 tia, 10°/tia, range 0.05–12 m, 10 Hz, topic `/scanN`.

**Sector min:**
```python
def _sector_min(self, world_yaw):
    # Lấy min trong cung ±LIDAR_HALF_DEG° theo world_yaw
    # Tính index tia dựa trên yaw robot hiện tại
    rel = wrap(world_yaw - self.yaw)
    ci  = int((degrees(rel) % 360) / 360 * n) % n
    hw  = max(1, int(LIDAR_HALF_DEG / 360 * n))   # ~4 tia cho ±20°
```

DFS thêm **rolling average** `SCAN_AVG=4` lần scan để giảm nhiễu.

---

## Mecanum Drive (BFS)

BFS dùng mecanum holonomic — di chuyển thẳng đến tâm ô không cần quay trước:

```python
# Chuyển world-frame → robot-frame
vxr =  cos(yaw)*ux*speed + sin(yaw)*uy*speed   # forward
vyr = -sin(yaw)*ux*speed + cos(yaw)*uy*speed   # strafe
wz  = clip(KP_ANG * wrap(target_yaw - yaw), -ANG_MAX, ANG_MAX)  # heading
```

DFS dùng differential-style: quay tại chỗ (TURN) → tiến thẳng (MOVE), chỉ dùng `linear.x` và `angular.z`.

---

## ROS2 Interface

```python
# Publishers
pub = create_publisher(Twist, f'/model/robot_{rid}/cmd_vel', 10)

# Subscribers (QoS: BEST_EFFORT)
create_subscription(LaserScan, f'/scan{rid}',                  cb_scan, qos)
create_subscription(Odometry,  f'/model/robot_{rid}/odometry', cb_odom, qos)

# Timer
create_timer(0.05, _loop)   # 20 Hz control loop
```

**Parameter:**
```bash
ros2 run ... --ros-args -p robot_id:=1
```

---

## Spawn script

```bash
./spawn_robot.sh <N> [x] [y] [z]
```

- Tạo SDF riêng cho mỗi robot: `sed "s/ROBOT_ID/${i}/g" one_robot.sdf`
- Offset Y: `Y_i = Y + (i-1) * 3.5` — tránh va chạm khi spawn nhiều robot
- Spawn song song, chờ tất cả xong mới tiếp tục
- Khởi động `ros_gz_bridge` cho 4 topic mỗi robot
- Khởi động 2 `static_transform_publisher` mỗi robot (world→chassis, chassis→lidar)
- Health check sau 5s

---

## Các lỗi thường gặp

### Robot không di chuyển sau spawn

```bash
# Kiểm tra bridge còn sống không
cat /tmp/bridge_log_1.txt
# Kiểm tra topic
ros2 topic hz /scan1
ros2 topic hz /model/robot_1/odometry
```

### BFS không tìm được đường (`[BFS ✗]`)

Nguyên nhân thường: tất cả ô hàng xóm bị `wall=True` do false positive LiDAR.
- Giảm `WALL_THRESH` (0.27 → 0.24)
- Tăng `SCAN_SETTLE` để LiDAR ổn định hơn

### Robot bị phanh liên tục (`[BRAKE]` lặp)

- Tăng `BRAKE_DIST` nếu tường thực sự gần
- Giảm `BRAKE_DIST` nếu false positive từ nhiễu LiDAR

### DFS bị stuck (`[STUCK]`)

Stack rỗng mà không đến được GOAL → mê cung không có đường, hoặc tất cả hướng bị đánh dấu sai là tường. Kiểm tra log `[BRAKE]` trước đó.

### Calibration sai (robot đi lệch ô)

`_ox`, `_oy` được calibrate từ vị trí spawn thực tế. Nếu spawn lệch so với config, robot sẽ hiểu sai vị trí. Kiểm tra log `[CALIB]` khi khởi động.

---

## Điểm mở rộng

Nếu cần thêm tính năng, các hàm chính cần sửa:

| Tính năng | Hàm liên quan |
|---|---|
| Đổi thuật toán tìm đường | `_bfs()` / `_dfs_next()` |
| Thay đổi cách phát hiện tường | `_scan_walls_at_cur()` / `_avg_walls()` |
| Thay đổi kinematics | `_move_to()` / `_cmd()` |
| Thêm trạng thái mới | `State` / `St` enum + `_loop()` |
| Thay đổi kích thước mê cung | `ROWS`, `COLS`, `CELL` + cập nhật `START`, `GOAL` |

---

## Ghi chú quan trọng cho AI

1. **Hai solver dùng cùng START/GOAL** — Cả BFS và DFS đều dùng `(2,0)` làm ô xuất phát và `(2,4)` làm đích. Điều này cho phép so sánh trực tiếp hiệu năng của hai thuật toán trên cùng một mê cung.

2. **`wall=None` khác `wall=False`** — BFS cho phép đi qua `None` (optimistic). Khi sửa logic tìm đường, không được bỏ sót trường hợp `None`.

3. **True không bao giờ bị ghi đè** — đây là quy tắc bất biến của wall map. Không thay đổi logic này trừ khi có lý do rõ ràng.

4. **DFS có bước TURN** mà BFS không có — vì DFS dùng differential drive, cần quay về đúng hướng trước. BFS dùng mecanum nên di chuyển trực tiếp.

5. **ARRIVE khác nhau** — BFS dùng `ARRIVE_R` (bán kính từ tâm ô trong world), DFS dùng `ARRIVE_DIST` (khoảng cách tương đối từ điểm xuất phát MOVE).

6. **`_ox`, `_oy`** chỉ có trong BFS — dùng để chuyển đổi cell↔world. DFS không cần vì tính arrive theo khoảng cách tương đối.

7. **Spawn script dùng `ROBOT_ID` là placeholder** trong SDF — `sed` replace tại runtime. Không hardcode robot ID trong SDF.
