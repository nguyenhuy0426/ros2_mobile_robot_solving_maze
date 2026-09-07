# Deliverables — LiDAR-only Wheel-Control RL: Diagnosis & Improvements

Companion to [`last_train_analysis.md`](last_train_analysis.md) (full evidence).
This file is the operator-facing summary: what the last run did, what was
wrong, what changed, and the exact commands to run everything.

---

## 1. Summary of the previous training run (from the actual logs)

SAC, LiDAR-only observation (36 rays × 4-frame stack = 144-d), direct 4-wheel
velocity action, fixed maze `nhom8_maze.sdf`.

| Metric | Value |
|---|---|
| Episodes / env steps | 818 / 187,841 |
| Wall-clock | 9.57 h |
| **Success rate** | **0.0%** (0/818) |
| Collision / timeout | 79.3% / 20.7% |
| Best reward (ep) | −12.3 (never positive; `+200` goal never collected) |
| Best `geo_dist_final` | 2.31 m (start ≈ 4.13 m, goal = 0) |

Source: `sac_wheel_logs/train_monitor.csv`, `sac_wheel_tensorboard/SAC_1`,
`sac_wheel_checkpoints/`. Reproduce with the `analyze_logs` command in §5.

## 2. Diagnosed failure modes (evidence in `last_train_analysis.md` §3)

- **A — SAC exploration collapse (primary).** `ent_coef` decayed 0.40 → 0.004;
  the policy went near-deterministic before ever sampling the `+200` terminal.
  Probe shows tiny, unsaturated actions (0–4% saturation).
- **B — Reward geometry rewards stalling (primary).** Timeout cost
  (600×−0.05 = −30) is *cheaper* than collision (−50). Over training,
  collision fell 94%→70% while timeout rose 2%→30% and episode length grew
  117→284: the agent learned survival-by-stalling.
- **C — No anti-stuck / no exploration incentive.** Nothing penalizes zero
  displacement or rewards visiting new maze cells.
- **D — Collision penalty fires too early** (94% first-decile collision rate)
  — "movement is dangerous" is learned before "movement reaches goals".
- **E — Mecanum strafing is dead.** Wheels are plain cylinders
  (`mu=0.8/mu2=0.3`), not a roller model; strafe commands produce ≤ 0.004 m.
  The robot is effectively differential-drive (4 outputs → 2 useful DOF).
- **F — LiDAR-only partial observability (contributing, not binding yet).**
  Address only after exploration/reward are fixed.

## 3. Chosen improvement strategy & why

Fix **exploration + reward geometry** and make success **reachable** before
touching the observation — the agent never even attempts traversal, so
observation capacity is not the binding constraint yet.

1. **Reward v2** (attacks A–D): softened collision (−15), anti-stall penalty,
   count-based **coverage/exploration bonus**, action-smoothing term.
2. **Start-distance curriculum** (attacks reachability): start near the goal so
   `+200` is actually collected, widen the start pool as recent success rises.
3. **TD3 baseline** on the *same* env: fixed Gaussian noise can't auto-collapse
   like SAC entropy — a direct test of failure mode A.
4. **Upgraded evaluation**: stuck rate, action saturation, coverage, final dist.

All shaping/curriculum is **default-OFF behind flags**, so the v1 baseline
(`sac_wheel_checkpoints`) stays byte-for-byte reproducible.

## 4. Files created / modified

**Created**
- `rl_training/analysis/analyze_logs.py` — reusable Monitor-CSV + TB analyzer.
- `rl_training/reward_shaping.py` — `StuckTracker`, `CoverageTracker`,
  `action_smoothness_penalty` (pure, Gazebo-free).
- `rl_training/curriculum.py` — `StartCurriculum` (geodesic-ranked starts).
- `rl_training/callbacks.py` — shared `EpisodeMetricsCallback` (TB metrics).
- `rl_training/train_sac_wheels_v2.py` — SAC + shaping + curriculum.
- `rl_training/train_td3_wheels.py` — TD3 baseline, same env.
- `rl_training/eval_wheels.py` — SAC/TD3 evaluator with the new metrics.
- `rl_training/tests/test_reward_shaping.py` — 8 offline unit tests.
- `rl_training/reports/last_train_analysis.md`, this file.

**Modified**
- `rl_training/config.py` — appended v2 reward, curriculum, and TD3/v2 output
  dir constants (no existing values changed).
- `rl_training/wheel_env.py` — `shaping`/`curriculum` kwargs + `_shaping_terms`,
  `_teleport_to`, saturation telemetry. Defaults preserve v1 behavior.

**Untouched (frozen baseline):** `train_sac_wheels.py`, `eval_sac_wheels.py`,
`smoke_test_wheels.py`, `maze_field.py`, `spawn_robot_wheel.sh`, world SDFs.

## 5. Exact commands (run from `/home/huynn/ros2_gazebo/ros2_ws`)

**Prereqs — 3 terminals** (unchanged from baseline):
```bash
# T1: simulator
gz sim -r worlds/nhom8_maze.sdf
# T2: robot spawn + ros_gz bridges
source /opt/ros/jazzy/setup.bash && ./spawn_robot_wheel.sh 1
# T3: source ROS before any python below
source /opt/ros/jazzy/setup.bash
```

**Offline unit tests (no Gazebo needed):**
```bash
./rl_venv/bin/python -m pytest rl_training/tests/ -q
```

**Re-analyze the previous run:**
```bash
./rl_venv/bin/python -m rl_training.analysis.analyze_logs \
    --monitor sac_wheel_logs/train_monitor.csv --tb sac_wheel_tensorboard/SAC_1
```

**Train — SAC v2 (shaping + curriculum):**
```bash
./rl_venv/bin/python -m rl_training.train_sac_wheels_v2 --steps 300000
# ablations: --no-curriculum   (shaping only)
```

**Train — TD3 baseline (same env):**
```bash
./rl_venv/bin/python -m rl_training.train_td3_wheels --steps 300000
# ablations: --no-shaping  --no-curriculum
```

**Evaluate either algorithm (real START→GOAL, curriculum off):**
```bash
./rl_venv/bin/python -m rl_training.eval_wheels \
    --algo sac --model sac_wheel_v2_checkpoints/sac_wheels_v2_final --episodes 20
./rl_venv/bin/python -m rl_training.eval_wheels \
    --algo td3 --model td3_wheel_checkpoints/td3_wheels_final --episodes 20
```

**TensorBoard (compare v1 / v2 / TD3):**
```bash
./rl_venv/bin/tensorboard --logdir_spec \
    v1:sac_wheel_tensorboard,v2:sac_wheel_v2_tensorboard,td3:td3_wheel_tensorboard
```

## 6. Dependencies & install commands

Verified present: `stable_baselines3` 2.8.0 (SAC, TD3), `gymnasium` 1.2.3,
`torch` 2.12.0+cpu, `numpy` 1.26.4, `tensorboard` 2.20.0 — all in `rl_venv`.

**Not installed** (needed only for the optional roadmap items; not faked):
```bash
# Recurrent RL (RecurrentPPO) — into the SAME venv:
./rl_venv/bin/pip install sb3-contrib==2.8.0

# SLAM + Nav2 (system ROS, apt):
sudo apt install ros-jazzy-slam-toolbox ros-jazzy-navigation2 \
                 ros-jazzy-nav2-bringup
```

## 7. Known limitations

- **Mecanum strafing does not work** in this world (plain-cylinder collision).
  The action space keeps 4 wheels for interface compatibility, but only ~2 DOF
  are usable. A true mecanum roller model would require editing the robot SDF
  collision geometry — out of scope and not requested.
- **Curriculum uses the geodesic field for start selection only**; the policy
  observation remains strictly LiDAR-only (no pose/goal leakage).
- **No SLAM/Nav2 hybrid implemented** — dependencies absent; §6 gives the path.
- **CPU-only PyTorch** ⇒ training is real-time-bound (~5–8 FPS), so 300k steps
  is many hours. Results in this report are from the v1 run; the v2/TD3 runs
  are **not yet executed** — commands above are provided to run them.

## 8. Next recommended experiment

Run `train_sac_wheels_v2.py`. **Success criterion:** collect ≥ 1 `+200`
terminal in the first curriculum stage (start ~1 cell from goal) within ~20k
steps. If entropy still collapses → switch to `train_td3_wheels.py`. If short
starts succeed but long ones don't → escalate the observation (longer frame
stack, or a recurrent policy after installing `sb3_contrib`).
