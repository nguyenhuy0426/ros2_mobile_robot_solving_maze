# Explore-then-Exit (v4) — Thiết kế & Lý do

## Bối cảnh

Pipeline cũ (goal-reaching) bị bế tắc: robot đứng tại ô start nhiều giờ. Nguyên
nhân đã chẩn đoán ở các phiên trước (thưởng lệch, entropy SAC sụp, mê cung thủng
biên — đã vá). Người dùng đổi bài toán sang dạng "khám phá" như 2 repo tham khảo:
robot chạy liên tục khắp nơi, quét vật cản, tạo map 2D, xong map mới tìm đường ra.

## Tham khảo từ 2 repo

- **DRL-Robot-Navigation-ROS2** (SAC, Gazebo Classic): obs 20-bin lidar + goal
  polar; reward = forward-speed bonus − turning penalty − clearance penalty khi
  min_scan < 1.35 m; terminal khi collision; dt 0.1 s. Cơ chế "chậm lại gần vật
  cản" của họ là penalty proximity — đã kế thừa ý tưởng nhưng gắn thêm với tốc độ
  (penalty ∝ −v·(1−clear/thresh)) vì bài toán của chúng ta cấm chạm tường tuyệt
  đối, không phải chỉ đến đích.
- **Mobile-Robot-Navigation-Using-DRL-and-ROS** (TD7, ROS2 Humble + Gazebo
  Classic): obs 20-bin lidar + agent state; reward = goal +100 / collision −100
  / clearance `(min_laser−1)/2` nếu < 1 m / action shaping `v/2 − |ω|/2`;
  terminal goal < 0.3 m hoặc collision < 0.35 m, max 500 bước, dt 0.1 s.
  **SLAM (slam_toolbox) trong repo này chỉ là demo — không được policy tiêu
  thụ; exploration hoàn toàn do policy RL tự thực hiện.** Hai điểm này khớp
  đúng thiết kế v4 của chúng ta: map là sản phẩm/telemetry, policy chỉ nhìn
  lidar + odom + coverage; không có random-walk/frontier nào thay policy.
  Khác biệt có chủ đích: clearance của họ KHÔNG gắn với tốc độ, của chúng ta
  có (−v·(1−clear/thresh)) vì yêu cầu "gần tường phải đi chậm"; timeout của
  họ dùng done=0 để bootstrap — SB3 đã tự lo việc này (handle_timeout_termination).

## Thiết kế v4

| Thành phần | Quyết định |
|---|---|
| Mê cung | 5×5, ô **0.75 m** (từ 0.50), tường 0.03, outer 3.93 m; `scripts/generate_maze_sdf.py` → `worlds/nhom8_maze75.sdf`; topology tường trong giữ nguyên bản 0.50 (verify BFS 25 ô nối đầy đủ) |
| Cửa ra | Biên Đông hàng 2 mở 1 ô (thiếu `wx_5_2`) — đích phase EXIT |
| Phase EXPLORE | +8/ô mới (25 ô), +20 khi đủ 25 ô; −0.02/bước; stuck −0.10/bước sau 20 bước dịch chuyển < 5 cm |
| Phase EXIT | potential shaping λ=2.0 theo geodesic tới cửa (Dijkstra chỉ dùng cho reward, policy không thấy); +100 khi ra qua cửa |
| Cứng | Va tường = terminal −30; ra cửa sớm = terminal −30; ra ngoài biên = terminal −30 |
| Chậm gần tường | r += −2.0·v·(1−clear/0.35) theo clearance phương trước ±45° — đi nhanh sát tường đắt, bò chậm gần như miễn phí |
| Obs 150-dim | 4×36 lidar stack + odom [x,y,cos yaw,sin yaw] + [coverage_frac, phase_flag] |
| Action 2-dim | [bánh trái, bánh phải] ∈ (−1,1)² diff-drive (robot thực không có lực ngang) |
| Map 2D | `occ_map.py`: trace 36 tia → grid 5 cm (83×83), free/occupied; publish `/map` (OccupancyGrid, latched); lưu PNG `explore_maps/ep_XXXX.png` cuối mỗi episode |
| Coverage | `CoverageTracker(pitch=0.78, wall_offset=0.03)` — lưới ô THẬT của mê cung (bán kính pitch gồm cả bề dày tường), không phải lưới 0.75 thuần (bug đã bắt bằng test) |

## Vì sao robot không còn đứng im

- Đứng im: −0.12/bước (thời gian + stuck) → 1500 bước ≈ −180; lao vào tường chỉ −30.
  Không còn phương án "đứng im là rẻ nhất" (nguyên nhân chính của v1).
- **Stall-termination (bổ sung sau quan sát thực tế):** với γ=0.99, dòng phạt
  nhỏ giọt still có PV ≈ −12 < −30 của va tường → policy đã học "sống sót bằng
  cách đứng im trôi hết episode" (356 episode: timeout với 4–6 ô). Sửa:
  `StallMonitor` — 4 cửa sổ 20 bước liên tiếp dịch chuyển < 8 cm (~8 s đứng
  thực sự) → kết thúc episode với phạt va tường. Đứng im giờ = −30 NHANH hơn
  cả va tường, không còn là phương án an toàn dưới chiết khấu.
- Thưởng phủ ô dày đặc (+8/ô) cho tín hiệu học ngay từ bước random đầu.
- Ô 0.75 m → hành lang 0.69 m vs robot 0.26 m: dư biên cho physics.

## Sửa lỗi tường hở (3 cm seams)

Bộ sinh SDF ban đầu đặt tường bắt đầu sau bề dày tường (`gy + t`) thay vì trải
kín cả pitch → mọi khớp nối giữa 2 đoạn tường thẳng hàng hở 3 cm (nhìn rõ trong
Gazebo). Đã sửa: mỗi tường trải **đầy đủ pitch** `[g(j), g(j)+pitch]` → các đoạn
kề nhau chia sẻ đầu mút, khớp T/corners kín tuyệt đối. Verify bằng test
`test_border_walls_sealed`: rasterize mỗi 1 cm trên toàn bộ các dải biên,
mọi điểm phải nằm trong hộp tường (trừ cửa ra). Lỗ hổng duy nhất còn lại trên
biên là **cửa ra chủ đích** (biên Đông hàng 2) — bắt buộc cho mục tiêu "tìm
đường ra mê cung".

## Kiểm chứng đã làm

- 37/37 unit + integration test (topology BFS, mapper, obs, safe-speed, exit
  logic, field); integration end-to-end bằng sensor giả: full path 25/25 ô →
  phase EXIT → success; early-exit −30; collision −30; SAC build + predict + step.
- Trên Gazebo thật: world 35 tường + cửa ra đúng chỗ; robot spawn (2.405, −2.535);
  robot di chuyển qua ô kế bên ngay episode đầu; `/map` publish đúng;
  PNG map thể hiện đúng hình học tường vùng đã quét.

## Theo dõi

```bash
tail -f sac_explore_logs/explore_monitor.csv          # coverage/success mỗi episode
./rl_venv/bin/tensorboard --logdir sac_explore_tensorboard
rviz2   # thêm Map display, topic /map, Fixed Frame: map
```

Kỳ vọng: vài chục episode đầu collision_rate cao là bình thường; tín hiệu học
sạch — coverage tăng dần, collision giảm, rồi map_complete và exitsuccess.
