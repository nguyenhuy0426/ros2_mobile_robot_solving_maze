# Classic Algorithm Maze Solver — nhom8_mecanum

Robot mecanum bánh toàn hướng giải mê cung 5×5 trong Gazebo bằng hai thuật toán cổ điển: **DFS** và **BFS**, kết hợp LiDAR khám phá thực tế (online exploration).

---

## Mục lục

- [Tổng quan](#tổng-quan)
- [Cấu trúc dự án](#cấu-trúc-dự-án)
- [Yêu cầu hệ thống](#yêu-cầu-hệ-thống)
- [Cài đặt](#cài-đặt)
- [Chạy nhanh](#chạy-nhanh)
- [Chi tiết từng file](#chi-tiết-từng-file)
- [Cấu hình mê cung](#cấu-hình-mê-cung)
- [Cấu hình robot](#cấu-hình-robot)
- [Thuật toán](#thuật-toán)
- [ROS2 Topics](#ros2-topics)
- [Debug & quan sát](#debug--quan-sát)
- [Tham số tuning](#tham-số-tuning)

---

## Tổng quan

```
Gazebo Sim (nhom8_mecanum)
        │
        ├── nhom8_maze.sdf      ← mê cung 5×5, tường cố định
        ├── one_robot.sdf       ← template robot_N (mecanum + LiDAR)
        │
ros_gz_bridge
        │
        ├── /scanN              ← LaserScan 36 tia
        ├── /model/robot_N/odometry
        └── /model/robot_N/cmd_vel
                │
        Python Node (ROS2)
                │
                ├── dfs_algorithm_solve_maze.py   ← DFS + backtrack
                └── bfs_algorithm_solve_maze.py   ← BFS online
```

Robot nhận LiDAR + Odometry, xây dựng bản đồ tường theo thời gian thực, lập kế hoạch đường đi và gửi lệnh Twist điều khiển mecanum.

---

## Cấu trúc dự án

```
classic_algorithm/
├── nhom8_maze.sdf                      # World Gazebo: mê cung 5×5
├── one_robot.sdf                       # Template SDF robot (placeholder ROBOT_ID)
├── spawn_robot.sh                      # Script spawn N robot + bridge + TF
├── dfs_algorithm_solve_maze.py         # Solver thuật toán DFS
└── bfs_algorithm_solve_maze.py         # Solver thuật toán BFS
```

---

## Yêu cầu hệ thống

| Thành phần | Phiên bản |
|---|---|
| OS | Ubuntu 24.04 |
| ROS2 | Jazzy |
| Gazebo | Harmonic (gz-sim 8) |
| Python | 3.10+ |
| ros-jazzy-ros-gz-bridge | latest |
| ros-jazzy-ros-gz-sim | latest |
| numpy | ≥ 1.24 |

---

## Cài đặt

```bash
# 1. Clone / copy project vào workspace ROS2
cd ~/ros2_ws/src
git clone <repo_url> classic_algorithm

# 2. Build workspace
cd ~/ros2_ws
colcon build --packages-select classic_algorithm
source install/setup.bash

# 3. Cấp quyền script
chmod +x src/classic_algorithm/spawn_robot.sh
```

---

## Chạy nhanh

### Bước 1 — Khởi động Gazebo với mê cung

```bash
gz sim classic_algorithm/nhom8_maze.sdf
```

### Bước 2 — Spawn robot + bridge (terminal mới)

```bash
# Spawn 1 robot
./spawn_robot.sh 1

# Hoặc spawn nhiều robot (test song song)
./spawn_robot.sh 5
```

Script tự động:
- Spawn `robot_1` .. `robot_N` vào world
- Khởi động `ros_gz_bridge` cho từng robot
- Khởi động `static_transform_publisher` (world → chassis → lidar)

### Bước 3 — Chạy solver (terminal mới)

```bash
# Thuật toán BFS
ros2 run classic_algorithm bfs_algorithm_solve_maze.py --ros-args -p robot_id:=1

# Hoặc thuật toán DFS
ros2 run classic_algorithm dfs_algorithm_solve_maze.py --ros-args -p robot_id:=1
```

### Dừng

```bash
# Ctrl+C ở terminal spawn_robot.sh → dừng toàn bộ bridge + TF
```

---

## Chi tiết từng file

### `nhom8_maze.sdf`

World Gazebo chứa mê cung 5×5 ô, kích thước mỗi ô 0.5m × 0.5m. Tường là các box tĩnh màu xanh dương (collision + visual). Hệ tọa độ xoay 90° CCW so với lưới logic.

```
World name : nhom8_mecanum
Tâm xoay   : X=3.25, Y=-3.25
START        = (2, 0)
GOAL         = (2, 4)
```

### `one_robot.sdf`

Template SDF robot mecanum 4 bánh. Placeholder `ROBOT_ID` được thay bằng số thực khi spawn. Các thành phần:

| Thành phần | Thông số |
|---|---|
| Chassis | 0.26 × 0.155 × 0.08 m, mass 1.5 kg |
| Bánh (×4) | cylinder radius=0.024 m, length=0.020 m |
| LiDAR | 36 tia, 10°/tia, range 0.05–12 m, 10 Hz |
| Spawn height | z = 0.024 m (đáy bánh chạm đất z=0) |
| Plugin điều khiển | VelocityControl (nhận Twist) |

### `spawn_robot.sh`

Script bash spawn N robot song song. Dùng offset `DY_STEP = 3.5 m` theo trục Y để các robot không chồng lên nhau.

```bash
./spawn_robot.sh <N> [x] [y] [z]

# Mặc định: x=2.25, y=-3.25, z=0.024
./spawn_robot.sh 1        # 1 robot
./spawn_robot.sh 5        # 5 robots song song
```

Sau khi chạy, kiểm tra:

```bash
ros2 topic echo /scan1 --once
ros2 topic echo /model/robot_1/odometry --once
```

### `bfs_algorithm_solve_maze.py`

Node ROS2 giải mê cung bằng **BFS online**. Robot mecanum — không cần quay trước khi tiến.

**Vòng đời:** `WAIT → SCAN → PLAN → MOVE → (BACKUP →) SCAN → ... → DONE`

| Tham số | Giá trị mặc định |
|---|---|
| `robot_id` | 1 |
| START_CELL | (2, 0) |
| GOAL_CELL | (2, 4) |
| LIN_SPD | 0.2 m/s |
| WALL_THRESH | 0.27 m |
| ARRIVE_R | 0.055 m |

### `dfs_algorithm_solve_maze.py`

Node ROS2 giải mê cung bằng **DFS online + backtrack stack**. Robot phải quay về đúng hướng trước khi tiến (không phải mecanum full holonomic).

**Vòng đời:** `SCAN → PLAN → TURN → MOVE → (BACK →) SCAN → ... → DONE`

| Tham số | Giá trị mặc định |
|---|---|
| `robot_id` | 1 |
| START | (2, 0) |
| GOAL | (2, 4) |
| LIN_SPD | 0.50 m/s |
| WALL_DIST | 0.26 m |
| ARRIVE_DIST | 0.43 m |

---

## Cấu hình mê cung

```
Lưới: 5 hàng × 5 cột, ô 0.5m × 0.5m
Hướng: 0=E(Đông)  1=N(Bắc)  2=W(Tây)  3=S(Nam)

  ┌──┬──┬──┬──┬──┐
  │  │  │  │  │  │
  ├──┼──┼──┼──┼──┤
  │  │  │  │  │  │
  ├──┼──┼──┼──┼──┤
  │S │  │  │  │G │   ← CẢ HAI solver: START=(2,0), GOAL=(2,4)
  ├──┼──┼──┼──┼──┤
  │  │  │  │  │  │
  ├──┼──┼──┼──┼──┤
  │  │  │  │  │  │
  └──┴──┴──┴──┴──┘

Ký hiệu in terminal:
  R = Robot hiện tại
  G = Goal
  S = Start
  · = Đã thăm
  * = Path BFS hiện tại (chỉ BFS)
  █ = Có tường (xác nhận)
  ░ = Chưa biết
```

---

## Cấu hình robot

### Mecanum kinematics

Robot BFS sử dụng mecanum drive — di chuyển theo bất kỳ hướng nào không cần quay:

```python
# World-frame velocity → Robot-frame velocity
vxr =  cos(yaw) * ux * speed + sin(yaw) * uy * speed
vyr = -sin(yaw) * ux * speed + cos(yaw) * uy * speed
```

Robot DFS dùng differential-style — quay tại chỗ về đúng hướng trước, rồi tiến thẳng.

### LiDAR sectors

Mỗi lần quét, LiDAR lấy giá trị nhỏ nhất trong cung ±20° cho mỗi trong 4 hướng:

```
E (0°)  : tia 340°–20°   → phát hiện tường Đông
N (90°) : tia 70°–110°   → phát hiện tường Bắc
W (180°): tia 160°–200°  → phát hiện tường Tây
S (270°): tia 250°–290°  → phát hiện tường Nam
```

---

## Thuật toán

### BFS Online

```
Mỗi khi đến ô mới:
  1. SCAN  — quét LiDAR 4 hướng, cập nhật wall map
  2. PLAN  — chạy BFS từ ô hiện tại đến GOAL
             (ô chưa biết = None → coi là đi được / optimistic)
  3. MOVE  — mecanum drive đến tâm ô kế tiếp
  4. Nếu LiDAR phát hiện tường bất ngờ → BACKUP → SCAN lại
```

BFS đảm bảo **đường ngắn nhất** trên bản đồ đã biết tại thời điểm đó.

### DFS Online

```
Mỗi khi đến ô mới:
  1. SCAN  — quét LiDAR, cập nhật wall map (trung bình 4 lần scan)
  2. PLAN  — DFS chọn bước tiếp theo:
             Ưu tiên 1: ô chưa thăm + tường xác nhận mở
             Ưu tiên 2: ô chưa thăm + tường chưa biết
             Ưu tiên 3: backtrack về ô trước (stack)
  3. TURN  — quay robot về đúng hướng (PID)
  4. MOVE  — tiến thẳng ARRIVE_DIST = 0.43m
  5. Nếu bị phanh → đánh dấu tường → BACK → PLAN lại
```

DFS khám phá sâu, có thể đi nhiều bước hơn BFS nhưng đơn giản hơn.

---

## ROS2 Topics

| Topic | Type | Chiều |
|---|---|---|
| `/scanN` | `sensor_msgs/LaserScan` | Gazebo → Python |
| `/model/robot_N/odometry` | `nav_msgs/Odometry` | Gazebo → Python |
| `/model/robot_N/cmd_vel` | `geometry_msgs/Twist` | Python → Gazebo |
| `/model/robot_N/pose` | `geometry_msgs/Pose` | Gazebo → (monitor) |

---

## Debug & quan sát

```bash
# Xem scan của robot 1
ros2 topic echo /scan1 --once

# Xem odometry
ros2 topic echo /model/robot_1/odometry --once

# Gửi lệnh điều khiển thủ công
ros2 topic pub /model/robot_1/cmd_vel geometry_msgs/msg/Twist \
  '{linear: {x: 0.2, y: 0.0}, angular: {z: 0.0}}'

# Xem log node
ros2 node list
ros2 node info /maze_solver_v6
```

Bản đồ ASCII được in ra terminal mỗi khi robot đến ô mới hoặc phát hiện tường mới.

---

## Tham số tuning

### BFS (`bfs_algorithm_solve_maze.py`)

```python
WALL_THRESH    = 0.27   # Tăng nếu bỏ sót tường, giảm nếu false positive
BRAKE_DIST     = 0.2    # Tăng để phanh sớm hơn
ARRIVE_R       = 0.055  # Tăng nếu robot không nhận ra đã đến đích
SCAN_SETTLE    = 10     # Tăng nếu LiDAR chưa ổn định khi scan
LIDAR_HALF_DEG = 20     # Tăng để quét rộng hơn (nhưng dễ false positive hơn)
LIN_SPD        = 0.2    # Tốc độ tiến (m/s)
```

### DFS (`dfs_algorithm_solve_maze.py`)

```python
WALL_DIST    = 0.26     # Ngưỡng tường
BRAKE_DIST   = 0.14     # Ngưỡng phanh khẩn cấp
ARRIVE_DIST  = 0.43     # Khoảng cách coi là đến ô (86% cell)
YAW_TOL      = 0.04     # Sai số yaw chấp nhận khi quay (rad)
LIN_SPD      = 0.50     # Tốc độ tiến
SCAN_AVG     = 4        # Số lần scan lấy trung bình (giảm nhiễu)
```
