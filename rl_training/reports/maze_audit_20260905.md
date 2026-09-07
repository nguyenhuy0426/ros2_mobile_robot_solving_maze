# Chẩn đoán robot không giải được mê cung — 2026-09-05

## Kết luận

Không có bằng chứng rằng chỉ tăng thời gian huấn luyện hoặc tốc độ bánh xe sẽ
giải quyết vấn đề. Log cho thấy các trainer chạy chồng thời gian, dữ liệu bị
trộn, episode thường chết rất sớm, còn policy thiếu bộ nhớ hành trình.
Ưu tiên là dữ liệu điều khiển đúng → học khám phá có bộ nhớ → đo thời gian giải.

Đã sửa các lỗi vận hành và thêm một biến thể bộ nhớ để thử nghiệm. Chưa chạy
benchmark Gazebo với bản sửa, nên chưa có số đo cải thiện success hoặc thời gian
giải. Tại lúc kiểm tra không thấy tiến trình trainer/Gazebo đang chạy.

## 1. Bằng chứng từ log thực tế

Nguồn: `sac_explore_multi_logs/*.csv`, `analysis.log`, và các event TensorBoard
trong `sac_explore_multi_tensorboard/SAC_*`. Số liệu và SHA-256 của CSV được lưu
ở `../analysis_outputs/maze_audit_20260905/evidence.json`; bảng theo maze ở
`../analysis_outputs/maze_audit_20260905/episode_report.txt`.

Đây là nguồn lịch sử được chụp trước đợt dọn dẹp ngày 2026-09-06. Các raw
checkpoint/log cũ đã được xóa sau khi tạo `evidence.json`; bản báo cáo và số liệu
tóm tắt là bản lưu còn lại để không nhầm chúng với run mới.

| Tập log | Episode | Thành công | Độ phủ trung bình | Cao nhất |
|---|---:|---:|---:|---:|
| Run cũ, CSV `explore_multi.20260904-001134.csv` | 3.110 | 0 | 6,69% | 44% = 11/25 vùng |
| CSV hiện tại `explore_multi.csv` | 28.290 | 0 | 6,46% | 36% = 9/25 vùng |

CSV hiện tại: 68,99% collision; 20,02% stall; 10,99% out-of-bounds;
1 early-exit; 0 timeout. Độ dài episode trung vị 16 bước, dài nhất 388 bước.
Có 4.404 episode va chạm trong tối đa 3 bước. Không episode nào đạt đủ 25 vùng
để chuyển sang EXIT. Đây là nhiệm vụ **thăm đủ vùng rồi mới thoát**, khó hơn chỉ
tìm một đường ra cửa.

### Đối chiếu runtime ngày 2026-09-06

Trong một world sạch với robot `delta_1`, tôi đo lệnh bốn bánh đồng thời ở
20 rad/s: vận tốc đo được khoảng 0,48 m/s, phù hợp bán kính bánh 0,024 m. Run
smoke 300 bước có bộ nhớ vị trí tạo 2 episode: coverage 2/25 vùng mỗi episode,
collision 100%, quãng đường thực 1,39 m và 1,69 m. Pose monitor thấy robot quay
lại gần đúng start `(-11.49, -10.22)` ngay sau reset. Đây giải thích vì sao
nhìn GUI sau một episode thường thấy ô xuất phát dù CSV vừa ghi coverage của
episode trước. Các số 2–11 trong CSV là zone count; không phải số ô liên tiếp.

Gazebo GUI tạo được cửa sổ nhưng framebuffer X11 trong môi trường này chụp toàn
màu đen; vì vậy không dùng ảnh màn hình để khẳng định vị trí. Pose topic
`/model/robot_1/pose`, scan và `/stats` được đọc trực tiếp; RTF runtime của
world sạch xấp xỉ 1,0. Visual text SDF gây lỗi `Geometry type 11` đã được bỏ
khỏi `one_robot.sdf`; lần khởi động sau không còn lỗi đó.

### Hai trainer cùng hoạt động: nguyên nhân vận hành ưu tiên cao nhất

| TensorBoard | PID trong tên event | Khoảng có dữ liệu (giờ Việt Nam) | Bước cuối |
|---|---:|---|---:|
| SAC_4 | 209623 | 04/09 00:41 → 14:24 | 115.258 |
| SAC_5 | 210644 | 04/09 00:42 → 05/09 00:53 | 333.229 |
| SAC_6 | 1002232 | 04/09 15:28 → 05/09 00:53 | 240.734 |

SAC_4 chồng SAC_5; sau đó SAC_6 lại chồng SAC_5. CSV có **6.880 lần bộ đếm bước
giảm**, bắt đầu từ cặp `624 → 13`, rồi `650 → 26`, liên tục xen kẽ đến cuối.
Đây không phải một policy học liên tục trong toàn bộ số giờ đó.

Code trainer dùng cùng `/wheel_*_<id>`, `/scan<id>`, pose robot và dịch vụ
teleport; đường dẫn log/checkpoint trước sửa cũng dùng chung. Vì vậy hai trainer
với cấu hình robot mặc định sẽ tranh lệnh và reset cùng robot. Transition của
policy A có thể chứa chuyển động/reset do policy B tạo ra: replay không còn
phản ánh đúng hành động A. Đây là cơ chế lỗi rõ ràng, không thể chữa bằng tăng
thưởng. Event/CSV xác nhận chạy chồng và trộn log; không còn command-line hoặc
trace bánh xe lịch sử để định lượng chính xác bao nhiêu va chạm do tranh lệnh.

### Timing và sensor có thể tiếp tục làm sai dữ liệu

`analysis.log` có RTF biến động mạnh: các mẫu cuối khoảng 0,030–0,943.
`_wait_fresh_step()` trước sửa chờ tối đa 2 giây rồi vẫn trả dữ liệu cũ.
Ở RTF 0,03, một chu kỳ LiDAR 0,1 giây mô phỏng cần khoảng 3,3 giây thực nếu
RTF ổn định. Timeout 2 giây có thể tạo transition chưa có quan sát mới.

`speed = distance / EXPL_DT` vẫn giả định bước luôn là 0,1 giây mô phỏng.
Trong thực tế việc chờ theo wall clock, snapshot lần lượt và reset chưa bảo đảm
điều này. Trước sửa, robot còn giữ lệnh khi SAC cập nhật mạng nếu không có
episode kết thúc. Tăng số gradient update lúc đó cũng tăng thời gian robot
chạy không nhận lệnh mới.

Không suy ra “RTF thấp khiến không đủ quãng đường trong 1.500 bước” chỉ từ RTF:
nếu mỗi bước chờ đúng một scan mới, 1.500 bước vẫn tương ứng khoảng 150 giây
mô phỏng. Vấn đề là bước stale và thời lượng điều khiển không được kiểm soát.

## 2. Nút thắt thuật toán

1. **Thiếu bộ nhớ dài hạn.** MLP nhận 4×36 LiDAR, vị trí/hướng, tỷ lệ coverage
   và phase. Hai lịch sử thăm các nhánh khác nhau nhưng có cùng số vùng đã thăm
   có thể cho cùng observation hiện tại. Policy không biết nhánh nào đã duyệt.
   Bản đồ occupancy được xây/lưu nhưng không đưa vào policy. Tăng thời gian học
   không bổ sung thông tin đang thiếu trong observation.
2. **EXIT chưa được học từ trải nghiệm thành công.** Reward thoát nằm sau điều
   kiện đủ 25 vùng, trong khi cao nhất mới 9 vùng ở log mới. Không nên hiểu
   “điểm thưởng tăng” là “sắp giải được mê cung”.
3. **Tỷ lệ cập nhật thấp khi tăng robot.** 13 env tạo 13 transition mỗi vector
   step, nhưng `gradient_steps=1` chỉ cập nhật mạng một lần: khoảng 1/13 update
   trên mỗi transition mới. Tăng robot thu dữ liệu không tự động tăng tương ứng
   lượng học. Đây là yếu tố cần thử đối chứng, không phải bảo đảm tăng tốc.
4. **Các sửa v6 đã có trước audit.** Gamma 0,997, target entropy −1, phạt dựa
   trên clearance cả vòng LiDAR, elite theo `(coverage, return)` đã có sẵn.
   Không đề xuất lại như giải pháp chưa được áp dụng. `target_entropy=-1`
   cũng không phải sàn entropy coefficient; coefficient cuối SAC_5 ~0,0179,
   SAC_6 ~0,0172 chưa chứng minh entropy collapse là nguyên nhân chính run mới.

`elites.json` còn global coverage 0,36 trong khi entry ortho_4 giữ 0,44 từ run
cũ. Bảng chứa di sản khác phiên; không dùng global score để tuyên bố checkpoint
đã được đánh giá tốt trên cả 13 maze. Elite vẫn chỉ là snapshot lúc có một
episode tốt, không phải kết quả evaluation nhiều seed.

## 3. Thay đổi đã triển khai

- `training_session.py`: khóa tiến trình bằng `flock`, giữ suốt phiên trainer.
  Trainer thứ hai dùng entrypoint đã sửa bị từ chối trước khi tạo env. Khóa
  không ngăn script khác hoặc binary cũ cố ý điều khiển cùng topic.
- `train_explore_multi.py`: mặc định mỗi phiên ghi vào
  `sac_explore_multi_runs/<timestamp-pid>/`; có `run.json`, CSV, TensorBoard và
  checkpoint riêng. `--run-dir` phải là thư mục chưa tồn tại, tránh ghi đè.
  Elite mới được đánh giá riêng theo từng run; dùng `--load` để warm-start trọng
  số, không trộn lại bảng elite lịch sử.
- Sửa resume: truyền đúng số bước bổ sung vào `learn`, giữ counter khi `--load`.
  `.zip` không chứa replay buffer; dành thêm warmup lấy dữ liệu mới trước khi
  cập nhật. Tần suất checkpoint được quy đổi theo số robot, khớp đơn vị flag.
- `explore_env.py`: quá hạn sensor thì stop và báo lỗi; cập nhật coverage trong
  info sau bước để callback/log không chậm một bước so với observation.
- `concurrent_vec_env.py`: stop toàn đội sau thu observation, trước update mạng,
  callback/reset; cũng stop toàn đội khi thu sensor bị lỗi.
- `analyze_explore_log.py`: phát hiện counter đảo chiều, không đưa kết luận
  “policy tiến bộ/thoái hóa” từ CSV trộn; stall không tự động quy cho reward.
- `--visit-memory`: thêm bitmap vị trí 5×5 từ odometry chuẩn hóa, observation
  150 → 175 chiều. Xóa theo episode, giữ độc lập mỗi robot. Không đọc nhãn zone
  từ registry hoặc tường chưa quan sát; không thêm BFS/DFS hoặc expert.
  Đây là bộ nhớ thô theo ô không gian, **không phải** 25 zone coverage gốc;
  các hành lang khác nhau có thể nằm chung ô nên chưa giải hết partial observability.
- `--gradient-steps`: cho phép thử 1, 4 hoặc −1; −1 cập nhật theo số transition
  thu được. Mặc định giữ 1 để tránh tự ý tăng tải tính toán.
- `EXPL_W_MAX` và `WHEEL_W_MAX` tăng 15 → 20 rad/s (vận tốc vành lý thuyết
  khoảng 0,36 → 0,48 m/s). Throttle theo clearance giảm lệnh còn tối thiểu 35%
  khi một tia LiDAR chạm ngưỡng, giữ tốc độ cao ở vùng trống nhưng hạn chế
  lao ngang vào tường.
- `EXPL_STALL_LIMIT` đổi 4 → 60 để tránh kết thúc episode sau khoảng 2,4 giây
  khi policy đang xoay tìm lối; grace mới khoảng 8 giây theo control loop.

Không thay luật thưởng, điều kiện đủ 25 vùng, kích thước robot hay vận tốc tối đa.
Checkpoint 150 chiều không tương thích với `--visit-memory`; cần policy mới hoặc
checkpoint 175 chiều. Các sửa được giữ trong working tree, không commit/reset
những thay đổi đã có của người dùng.

## 4. Kế hoạch tăng tốc có đo lường

### A. Kiểm tra điều khiển trước khi mở run dài

Chỉ một trainer; trước hết dùng một robot, sau đó 13 robot. Đo scan timestamp,
simulation delta, khoảng cách mỗi lệnh, stale count và reset pose thực tế.
So sánh 1 với 13 robot: thời lượng lệnh và quãng đường mỗi bước phải ổn định.
Nếu sensor vẫn timeout thì sửa nguồn tải/bridge/nhịp mô phỏng; không tăng timeout
mù quáng rồi lại coi dữ liệu stale là valid.

Bước kỹ thuật kế tiếp nên chuyển sang stepping theo thời gian mô phỏng và
snapshot đồng bộ scan/pose; đo speed bằng delta sim-time thực. Bản sửa hiện tại
stop quanh update nhưng chưa bảo đảm cả 13 robot có cùng timestamp snapshot,
và chưa loại bỏ chuyển động do quán tính trong quá trình stop/reset.

### B. So sánh thuật toán tuần tự, cùng điều kiện

Chạy baseline mới sau sửa hạ tầng, rồi biến thể memory, rồi memory tăng số
gradient update. Giữ cùng mazes, seeds, mức thưởng, stall-limit và budget.
Không so trực tiếp với CSV trộn để tuyên bố mức cải thiện.

Ví dụ, sau khi đã khởi động đúng world và spawn robot/bridge:

```bash
source /opt/ros/jazzy/setup.bash
source install/setup.bash

# A: baseline; hoàn tất rồi mới chạy B.
rl_venv/bin/python -m rl_training.train_explore_multi \
  --n-robots 13 --steps 50000 --seed 0 --stall-limit 60 \
  --roam-bonus 0.3 --gradient-steps 1 \
  --run-dir sac_explore_multi_runs/baseline_seed0

# B: chỉ đổi bộ nhớ, dùng policy mới.
rl_venv/bin/python -m rl_training.train_explore_multi \
  --n-robots 13 --steps 50000 --seed 0 --stall-limit 60 \
  --roam-bonus 0.3 --gradient-steps 1 --visit-memory \
  --run-dir sac_explore_multi_runs/memory_seed0

rl_venv/bin/python -m rl_training.analyze_explore_log \
  --csv sac_explore_multi_runs/memory_seed0/logs/explore_multi.csv
```

Sau đó thử memory với `--gradient-steps 4`, rồi −1 nếu throughput đủ tốt; mỗi
thử nghiệm dùng thư mục mới. 50k bước là pilot để phát hiện lỗi/so sánh tín hiệu,
không phải cam kết hội tụ. Lặp nhiều seed trước khi chọn phương án.

Đo đồng thời: success theo từng maze, coverage median/max, collision/OOB/stale,
transition/s, update/s, wall-time đến các mốc coverage 20/40/60/100%, và thời gian
mô phỏng giải ở các episode thành công. Không tăng vận tốc chỉ vì FPS thấp:
FPS huấn luyện, RTF và tốc độ robot là ba đại lượng khác nhau.

### C. Nếu bộ nhớ thô chưa đủ

Thử observation có local occupancy + visitation map và encoder CNN, hoặc policy
có trạng thái recurrent với replay theo chuỗi. Vẫn giữ LiDAR và policy neural
ra hành động. Chỉ thêm curriculum 4→8→16→25 vùng trong một thí nghiệm riêng,
có target coverage trong observation; báo rõ success curriculum khác success
nhiệm vụ gốc. Đánh giá cuối luôn yêu cầu đủ 25 vùng rồi ra cửa. Không giảm chuẩn
đánh giá để tạo tỷ lệ thành công đẹp.

## 5. Kiểm chứng và giới hạn

**48 tests pass**, gồm bộ test explore/multi-maze/reward và 8 regression mới:
memory phân biệt lịch sử, reset/terminal observation, khóa liên tiến trình,
sensor stale, dừng toàn đội, analyzer CSV trộn, warm-start SAC đúng 8 bước bổ
sung, và SAC thực hiện gradient update với observation 175 chiều.
CLI `--help` chạy được. Pytest được chạy với `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1`
vì plugin ROS launch_testing tự nạp cần `lark` chưa có; không thay môi trường.

Đây là kiểm chứng offline với env nhỏ và test hình học, không phải robot đã
giải mê cung trong Gazebo. Chưa đo success hay thời gian giải sau sửa; chưa
khởi động thêm phiên huấn luyện hàng chục giờ trong audit này.

Đã dọn các artifact sinh ra từ các phiên cũ: `archive/`, toàn bộ nhóm
`dqn_*`, `sac_*` checkpoint/log/TensorBoard cũ, `sac_explore_*` cũ và
`explore_maps/`. Các file nguồn, world/maze input, `rl_venv/`, build/install
và báo cáo/evidence được giữ lại. Phiên mới phải ghi vào
`sac_explore_multi_runs/<run>/`, không tái sử dụng thư mục cũ.

Tham chiếu API: đối chiếu trực tiếp SB3 2.8.0 cài tại workspace về
`_setup_learn`, replay warm-start và `gradient_steps`.
[Tài liệu mã nguồn SB3](https://stable-baselines3.readthedocs.io/en/master/_modules/stable_baselines3/common/off_policy_algorithm.html)
giải thích −1 dùng số transition vừa thu; tăng số update không bảo đảm giảm
wall-time nếu CPU/GPU trở thành nút thắt.
