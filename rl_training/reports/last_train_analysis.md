# Last Training Run — Analysis & Diagnosis

**Run:** SAC, LiDAR-only 4-wheel velocity control, fixed maze `nhom8_maze.sdf`
**Artifacts:** `sac_wheel_logs/train_monitor.csv`, `sac_wheel_tensorboard/SAC_1`,
`sac_wheel_checkpoints/` (10k→final).
**Reproduce this report:**
`./rl_venv/bin/python -m rl_training.analysis.analyze_logs --monitor sac_wheel_logs/train_monitor.csv --tb sac_wheel_tensorboard/SAC_1`

---

## 1. Headline result — the policy did not learn to solve the maze

| Metric | Value |
|---|---|
| Episodes | 818 |
| Env steps | 187,841 |
| Wall-clock | 9.57 h (≈ 5–8 FPS, real-time bound) |
| **Success rate** | **0.0%** (0 / 818, entire run) |
| Collision rate | 79.3% |
| Timeout rate | 20.7% |
| Reward mean / min / max | −50.6 / −85.3 / **−12.3** (never positive) |
| Best `geo_dist_final` ever | **2.31 m** (start geo ≈ 4.13 m, goal = 0 m) |
| Episodes ending `geo < 1.0 m` | **0%** |

The maximum episode reward across the whole run is **−12.3**. The success
bonus (`WR_GOAL = +200`) was **never once collected** — the robot never
reached the goal in 818 attempts.

## 2. What the policy actually learned (checkpoint probe)

Ran `sac_wheels_final` deterministically (3 episodes, live Gazebo):

```
ep0: COLLISION  steps=39  geo 4.13->3.97  min_geo=3.83  path=0.66m  sat=4%  act_mean=[ 0.24 -0.05  0.15 -0.10]
ep1: TIMEOUT    steps=300 geo 4.13->3.90  min_geo=3.79  path=1.35m  sat=0%  act_mean=[ 0.12  0.26 -0.10 -0.25]
ep2: COLLISION  steps=121 geo 4.13->3.81  min_geo=3.67  path=2.00m  sat=2%  act_mean=[ 0.31  0.08 -0.22 -0.01]
```

- **`geo_dist` barely moves** (4.13 → ~3.8 m). The robot never travels more
  than ~2 m of path and never approaches the goal.
- **Actions are tiny** (mean magnitude ~0.1–0.3 of full scale) and **almost
  never saturated** (0–4%). This is a *timid, near-stalled* policy, not an
  aggressive one.
- Action means are **left/right asymmetric** (e.g. `[0.24,−0.05,0.15,−0.10]`),
  i.e. a weak in-place turn bias — it spins/creeps rather than driving down a
  corridor.

## 3. Diagnosed failure modes (ranked by evidence)

### A. Exploration collapse → degenerate "stall" optimum *(primary cause)*
`train/ent_coef` decays **0.40 → 0.004** and `actor_loss` crosses to positive
(+3.1) while `critic_loss` settles at ~2.4 — the SAC entropy term collapsed
early, so the policy became near-deterministic **before** it ever discovered a
path to the goal. With exploration gone and the +200 terminal never sampled,
there is nothing to bootstrap goal-seeking from. The probe (tiny,
unsaturated actions) is the fingerprint of this collapse.

### B. Reward geometry rewards stalling over progress *(primary cause)*
The only outcomes are collision (−50, terminal) and timeout
(600 × −0.05 = **−30**). Timeout is *strictly cheaper* than collision, and the
dense progress term (`5·Δgeo`) is near-zero when creeping. So the value
function correctly concludes: **"don't move much — a slow timeout beats a
collision."** The data confirms it: as training proceeds, **collision rate
falls 94% → 70% while timeout rate rises 2% → 30%** and mean episode length
grows 117 → 284 steps. The agent is optimizing survival-by-stalling, exactly
as the reward is (mis)shaped.

### C. No anti-stuck / no exploration incentive
Nothing penalizes zero displacement and nothing rewards visiting new maze
cells. In a maze whose goal is a long corridor away, a purely *local* geodesic
gradient plus an *unreachable* sparse terminal gives no usable learning signal
once the agent decides stalling is safe (see B).

### D. Collision penalty fires too early, too often
31% of all collisions end within 50 steps (5 s); the first-decile collision
rate is 94%. Early training is dominated by the −50 cliff near spawn, teaching
"movement is dangerous" long before "movement reaches goals."

### E. Over-parameterized action space (mecanum strafing is dead)
Empirical test (commanded 2.5 s each, measured pose displacement):

```
forward   [ 1, 1, 1, 1] -> |d| = 0.605 m   ✓ drives
strafe_L  [-1, 1, 1,-1] -> |d| = 0.003 m   ✗ no lateral motion
strafe_R  [ 1,-1,-1, 1] -> |d| = 0.004 m   ✗ no lateral motion
turn      [-1, 1,-1, 1] -> rotates in place
```

The wheels are **plain cylinders** with anisotropic friction
(`mu=0.8` roll / `mu2=0.3` lateral, `one_robot.sdf:239-241`) — a mecanum
*mesh* but **not** a roller collision model. There is no lateral propulsion.
The robot is effectively **differential-drive**: the 4 wheel outputs collapse
to 2 useful DOF (left side `fl≈rl`, right side `fr≈rr`). The policy must waste
capacity discovering this coupling.

### F. LiDAR-only partial observability *(contributing)*
36 rays × 4-frame stack (144-d). Perceptually aliased corridors; prior
DAgger/DRQN attempts on this maze also failed to close the loop (project
memory). Contributes to difficulty but is **not** the reason for 0% here —
the agent never even attempts to traverse, so observation capacity is not yet
the binding constraint. Address it *after* exploration/reward are fixed.

## 4. What is NOT the problem

- **Sim/actuation is healthy:** forward command → +0.605 m; scan streams at
  ~20 Hz; teleport reset returns to start exactly (smoke test).
- **Not a divergence/NaN issue:** losses are stable and bounded.
- **Not simply "needs more steps":** 187k steps with a completely flat
  `geo_dist_final` (no downward trend at all) indicates a mis-specified
  objective, not undertraining.

## 5. Improvement strategy (implemented — see `IMPROVEMENTS` below)

Priority follows the evidence: fix **exploration + reward geometry** and make
success **reachable via curriculum** before touching the observation.

1. **Reward v2** (attacks A–D): anti-stuck penalty, a count-based **coverage /
   exploration bonus** over maze cells, **softened collision** penalty, and a
   mild action-smoothing term — all gated behind config flags, default OFF so
   the existing baseline stays reproducible.
2. **Start-distance curriculum** (attacks A/reachability): begin near the goal
   so the agent *actually collects* `+200`, then push the start farther as the
   recent success rate rises — bootstrapping the value function.
3. **TD3 baseline** reusing the same env: TD3's fixed exploration noise does
   not auto-collapse like SAC's entropy, a direct test of failure mode A.
4. **Upgraded evaluation** reporting stuck rate, action saturation, coverage,
   and final distance so future runs are diagnosable at a glance.
5. **SLAM/Nav2 & recurrent RL:** dependencies are **not installed**
   (`slam_toolbox`, `nav2`, `sb3_contrib` all absent). Roadmap + exact install
   commands provided; **not faked**.

## 6. Next recommended experiment
Run the curriculum + reward-v2 SAC (`train_sac_wheels_v2.py`). Success
criterion: the agent collects at least one `+200` terminal within the first
curriculum stage (start 1 cell from goal) inside ~20k steps. If entropy still
collapses, switch to the TD3 baseline; if the agent reaches the goal on short
starts but not long ones, escalate the observation (longer stack or a
recurrent policy once `sb3_contrib` is installed).
