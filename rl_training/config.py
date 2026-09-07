#!/usr/bin/env python3
"""
config.py — Centralized configuration for DQN maze solver.

Single source of truth for all constants, hyperparameters, and paths.
7-action discrete design: STOP, FORWARD, BACKWARD, STRAFE_L, STRAFE_R, TURN_L, TURN_R.
"""

from pathlib import Path

# ══════════════════════════════════════════════════════════════════════════
# Robot geometry (matches Gazebo nhom8_mecanum)
# ══════════════════════════════════════════════════════════════════════════
ROBOT_RADIUS = 0.10       # effective collision radius (chassis 0.26×0.155m)
CHASSIS_L    = 0.26       # one_robot.sdf box size, along +x
CHASSIS_W    = 0.155      # one_robot.sdf box size, along +y
WHEEL_RADIUS = 0.024
WHEEL_SEP    = 0.18       # track width: wheel links at y = ±0.09 in one_robot.sdf

# ══════════════════════════════════════════════════════════════════════════
# Lidar
# ══════════════════════════════════════════════════════════════════════════
N_RAYS       = 36         # 360° / 10° = 36 rays
LIDAR_X_OFF  = 0.08       # lidar mount, +x from the chassis center
LIDAR_MAX    = 3.0        # max range (m)
LIDAR_MIN    = 0.05       # min range (m)

# ══════════════════════════════════════════════════════════════════════════
# Velocity limits (mecanum) — used by discrete action table
# ══════════════════════════════════════════════════════════════════════════
VX_MAX       =  0.35      # forward max (m/s)
VX_MIN       = -0.15      # reverse max (m/s)
VY_MAX       =  0.25      # lateral max (m/s)
WZ_MAX       =  2.5       # angular velocity max (rad/s) — fast 90° turns

# ══════════════════════════════════════════════════════════════════════════
# Discrete Action Space — 7 cardinal actions (body frame)
# ══════════════════════════════════════════════════════════════════════════
#   Action ID →  (vx,     vy,      wz)
DISCRETE_ACTIONS = {
    0: (0.0,     0.0,     0.0),       # STOP
    1: (VX_MAX,  0.0,     0.0),       # FORWARD
    2: (VX_MIN,  0.0,     0.0),       # BACKWARD
    3: (0.0,     VY_MAX,  0.0),       # STRAFE LEFT
    4: (0.0,    -VY_MAX,  0.0),       # STRAFE RIGHT
    5: (0.0,     0.0,     WZ_MAX),    # TURN LEFT (in-place)
    6: (0.0,     0.0,    -WZ_MAX),    # TURN RIGHT (in-place)
}
N_ACTIONS    = len(DISCRETE_ACTIONS)   # 7
ACTION_NAMES = ["STOP", "FWD", "BWD", "STR_L", "STR_R", "TRN_L", "TRN_R"]

# ══════════════════════════════════════════════════════════════════════════
# Episode
# ══════════════════════════════════════════════════════════════════════════
DT           = 0.10       # simulation timestep (s)
MAX_STEPS    = 800        # max steps per episode (80 seconds — maze needs more time)
GOAL_RADIUS  = 0.15       # goal-reach threshold (m) — slightly larger for 3-action
MAX_COLLISIONS_PER_EP = 10  # allow N collisions before terminating (bounce-back)
COLLISION_MARGIN     = 0.05  # margin added to ray-to-chassis distance for collision

# ══════════════════════════════════════════════════════════════════════════
# Observation / Action dimensions
# ══════════════════════════════════════════════════════════════════════════
# One frame: 36 lidar + dist + cos + sin. The network input is FRAME_STACK frames
# (oldest→newest) plus the last-action one-hot — temporal memory so the policy can
# disambiguate perceptually-aliased states (junctions/dead-ends that look identical
# in a single lidar frame but need different actions depending on arrival direction).
SINGLE_OBS_DIM = 39                   # one frame: 36 lidar + 3 goal info
FRAME_STACK    = 4                    # stacked frames (temporal memory)
STATE_DIM    = FRAME_STACK * SINGLE_OBS_DIM + N_ACTIONS   # 4*39 + 7 = 163 (network input)
ACTION_DIM   = N_ACTIONS  # 7 discrete actions (for Q-network output)

# ══════════════════════════════════════════════════════════════════════════
# Maze grid (for exploration tracking)
# ══════════════════════════════════════════════════════════════════════════
MAZE_CELLS   = 25         # 5×5 maze
CELL_SIZE    = 0.5        # meters per cell

# ══════════════════════════════════════════════════════════════════════════
# Reward — Maze navigation (collision-recovery design)
# ══════════════════════════════════════════════════════════════════════════
R_GOAL           = 200.0      # goal reached — make it worth surviving for
R_COLLISION      = -5.0       # collision penalty — REDUCED (non-terminal bounce-back)
R_DIST_SCALE     = 2.0        # distance-progress shaping — REDUCED to prevent tunnel vision
R_NEW_CELL_BASE  = 15.0       # exploration bonus — INCREASED to reward deeper exploration
R_REVISIT        = -0.02      # very light revisit penalty
R_TIME_PENALTY   = -0.05      # reduced time penalty to allow more exploration
R_PROX_THRESH    = 0.20       # DRASTICALLY reduced — don't penalize normal corridor traversal
R_PROX_SCALE     = -0.3       # lighter proximity penalty
R_COVERAGE_BONUS = 5.0        # BIG coverage bonus to drive exploration
R_NONMOTION_PENALTY = -0.1    # per-step penalty when the action doesn't translate the robot
                              # (STOP / TURN_L / TURN_R) — breaks the spin-in-place / freeze
                              # local optimum so translating toward the goal is favored.

# ══════════════════════════════════════════════════════════════════════════
# Stuck detection
# ══════════════════════════════════════════════════════════════════════════
STUCK_WINDOW          = 20    # look-back window (steps) for stuck detection
STUCK_COLLISION_THRESH = 3    # collisions within window → stuck in corner

# ══════════════════════════════════════════════════════════════════════════
# DQN Hyperparameters
# ══════════════════════════════════════════════════════════════════════════
DQN_LR           = 1e-4       # learning rate
DQN_DISCOUNT     = 0.99       # gamma
DQN_TAU          = 0.005      # soft-update Polyak coefficient
DQN_EPS_START    = 1.0        # initial exploration epsilon
DQN_EPS_END      = 0.05       # final exploration epsilon
DQN_EPS_DECAY    = 0.998      # epsilon decay per episode (reaches ~0.05 floor by ~ep1500)

# ══════════════════════════════════════════════════════════════════════════
# Network architecture
# ══════════════════════════════════════════════════════════════════════════
HIDDEN_1     = 512        # first hidden layer
HIDDEN_2     = 256        # second hidden layer

# ══════════════════════════════════════════════════════════════════════════
# Training loop
# ══════════════════════════════════════════════════════════════════════════
BUFFER_SIZE       = 100_000     # replay buffer capacity
ELITE_BUFFER_SIZE = 50_000      # elite replay buffer (best exploration episodes)
ELITE_SAMPLE_RATIO = 0.2        # fraction of batch from elite buffer
BATCH_SIZE        = 128         # mini-batch size
TRAIN_EVERY_N_EP  = 2           # train after every N episodes
TRAIN_ITERATIONS  = 200         # gradient steps per training cycle
WARMUP_STEPS      = 2000        # random actions before training (reduced for Gazebo)
EVAL_EPISODES     = 20          # episodes per evaluation
EVAL_EVERY_N_EP   = 50          # evaluate every N episodes
SAVE_EVERY        = 200         # save model every N training cycles
ELITE_AVG_WINDOW  = 10          # episodes for moving average of cells_visited

# Expert-guided exploration: inject whole BFS-expert episodes at a decaying rate so
# the replay/elite buffers always contain optimal goal-reaching trajectories. The
# fraction decays linearly to 0 by EXPERT_DECAY_EPISODES, after which the model
# learns purely from its own (epsilon-greedy) experience.
EXPERT_FRAC_START     = 0.5     # initial fraction of post-warmup episodes driven by expert
EXPERT_DECAY_EPISODES = 1500    # expert-injection fraction reaches 0 at this episode

# Behavior-cloning warm-start: pretrain the Q-net on expert demonstrations
# (supervised obs→action) before RL, so the policy starts as a navigator instead of
# first discovering the passive spin-in-place optimum. After BC, epsilon starts at
# BC_EPS_START (not 1.0) so the agent actually exploits the warm-started policy.
BC_WARMSTART_EPISODES = 400     # expert episodes collected for BC pretraining
BC_WARMSTART_EPOCHS   = 25      # supervised epochs over the BC dataset
BC_EPS_START          = 0.5     # epsilon after BC warm-start (overrides DQN_EPS_START)

# ══════════════════════════════════════════════════════════════════════════
# Paths
# ══════════════════════════════════════════════════════════════════════════
CKPT_DIR  = Path("dqn_checkpoints")
LOG_DIR   = Path("dqn_logs")
TB_DIR    = Path("dqn_tensorboard")

# ══════════════════════════════════════════════════════════════════════════
# Wheel-control SAC (continuous, LiDAR-only policy) — Gazebo training
# ══════════════════════════════════════════════════════════════════════════
# Policy input:  WHEEL_FRAME_STACK stacked LiDAR frames (nothing else).
# Policy output: 4 wheel angular velocities [fl, fr, rl, rr] in rad/s.
# Pose/goal are used ONLY by the env for reward and termination.

WS_ROOT   = Path(__file__).resolve().parents[1]     # ros2_ws/
WORLD_SDF = WS_ROOT / "worlds" / "nhom8_maze.sdf"
WORLD_NAME = "nhom8_mecanum"

MAZE_ORIGIN_XY = (2.0, -4.5)      # SW corner of the 5x5 maze in world frame
START_XY       = (2.25, -3.25)    # cell (row 2, col 0) center
GOAL_XY        = (4.25, -3.25)    # cell (row 2, col 4) center
START_YAW      = 0.0

WHEEL_W_MAX      = 20.0           # rad/s per wheel (rim ≈ 0.48 m/s at r=0.024)
WHEEL_ACTION_DIM = 4              # [fl, fr, rl, rr]
WHEEL_FRAME_STACK = 4             # LiDAR temporal memory (POMDP workaround)
WHEEL_STATE_DIM  = WHEEL_FRAME_STACK * N_RAYS   # 4*36 = 144

WHEEL_DT         = 0.10           # control period (s), matches 10 Hz lidar
WHEEL_MAX_STEPS  = 600            # 60 s per episode
WHEEL_SETTLE_SEC = 0.6            # wait after teleport before first obs

# Reward (training-only signals; geodesic field from maze_field.py)
WR_PROGRESS   = 5.0               # per meter of geodesic progress to goal
WR_GOAL       = 200.0             # reaching GOAL_RADIUS of the goal
WR_COLLISION  = -50.0             # lidar-detected collision (terminal)
WR_TIME       = -0.05             # per-step time penalty
WR_PROX_THRESH = 0.18             # unsafe wall proximity threshold (m)
WR_PROX_SCALE  = -0.5             # penalty scale inside the threshold

# SB3 SAC hyperparameters
SACW_LR            = 3e-4
SACW_GAMMA         = 0.99
SACW_TAU           = 0.005
SACW_BUFFER_SIZE   = 200_000
SACW_BATCH_SIZE    = 256
SACW_LEARN_STARTS  = 2_000        # random-action warmup steps
SACW_TRAIN_FREQ    = 1            # gradient step every env step
SACW_NET_ARCH      = [256, 256]
SACW_TOTAL_STEPS   = 300_000      # default training budget
SACW_CKPT_EVERY    = 10_000       # env steps between checkpoints

SACW_CKPT_DIR = Path("sac_wheel_checkpoints")
SACW_LOG_DIR  = Path("sac_wheel_logs")
SACW_TB_DIR   = Path("sac_wheel_tensorboard")

# ══════════════════════════════════════════════════════════════════════════
# Reward-shaping v2 + curriculum (opt-in; v1 baseline unchanged when disabled)
# ══════════════════════════════════════════════════════════════════════════
# GazeboWheelEnv(shaping=True) turns these on; make_wheel_env defaults keep the
# original v1 behavior so sac_wheel_checkpoints stays reproducible.
# See rl_training/reports/last_train_analysis.md for the evidence.

# Softened collision (v1 -50 cliff dominated learning; timeout at -30 was
# "cheaper" than colliding, so the agent learned to stall).
WR_COLLISION_V2  = -15.0

# Anti-stall: penalize < WR_STUCK_MIN_DISP metres of net travel over a window.
WR_STUCK_WINDOW   = 20            # steps (~2 s at WHEEL_DT)
WR_STUCK_MIN_DISP = 0.05          # metres
WR_STUCK_PENALTY  = -0.20         # per step while stuck

# Count-based exploration bonus: one-time reward per newly entered maze cell.
WR_COVERAGE_BONUS = 1.0

# Action smoothing (jerk) penalty — kept small so it does not re-induce timidity.
WR_SMOOTH_SCALE   = -0.02

# Start-distance curriculum (env-side; obs stays LiDAR-only).
CURR_GRID_CELLS       = 5         # 5x5 maze
CURR_SUCCESS_WINDOW   = 20        # episodes evaluated before advancing
CURR_ADVANCE_THRESH   = 0.6       # >=60% success in window → next level

# v2 / TD3 output directories (kept separate from the v1 baseline artifacts).
SACW_V2_CKPT_DIR = Path("sac_wheel_v2_checkpoints")
SACW_V2_LOG_DIR  = Path("sac_wheel_v2_logs")
SACW_V2_TB_DIR   = Path("sac_wheel_v2_tensorboard")

TD3W_CKPT_DIR = Path("td3_wheel_checkpoints")
TD3W_LOG_DIR  = Path("td3_wheel_logs")
TD3W_TB_DIR   = Path("td3_wheel_tensorboard")
# TD3 exploration noise (std, in normalized action units); does not auto-collapse
# like SAC entropy — a direct test of the v1 exploration-collapse failure mode.
TD3W_ACTION_NOISE = 0.2

# ══════════════════════════════════════════════════════════════════════════
# Gazebo RL v3 — fixed-maze training, pure RL (no BFS/DFS expert anywhere)
# ══════════════════════════════════════════════════════════════════════════
# Evidence-driven fixes over v1/v2 (see rl_training/reports/last_train_analysis.md):
#   * 2-DOF differential-drive action: the wheels are plain cylinders (mu 0.8/0.3)
#     with NO lateral propulsion, so 4-dim mecanum actions waste one dead DOF
#     (failure mode E). [left, right] wheel speeds cover the real dynamics.
#   * Odometry in the observation: the maze is FIXED, so (x, y, cos yaw, sin yaw)
#     makes the task nearly Markov — the policy can memorize the corridor value
#     landscape instead of fighting perceptual aliasing with frame stacks alone
#     (failure mode F). Odometry is a standard sensor; no solver is involved.
#   * Balanced terminal penalties: v1 collision (-50) made timeout cheaper than
#     crashing → stalling; v2 collision (-15) made crashing cheaper than anything
#     → instant suicides. v3 keeps collision above the worst-case timeout stream
#     while the start-distance curriculum keeps success reachable.

WHEEL_DIFF_DRIVE  = False      # legacy default (v1/v2 reproducible); v3 script opts in
WHEEL_USE_ODOM    = False      # legacy default (LiDAR-only 144-dim); v3 script opts in
WHEEL_ODOM_DIM    = 4          # [x_norm, y_norm, cos(yaw), sin(yaw)] within maze bounds

WHEEL_GOAL_RADIUS = 0.20       # v3 goal-reach threshold (m) — inside the goal
                               # cell, forgiving of ±10 cm centering error.
                               # Legacy envs keep C.GOAL_RADIUS (0.15).

WR_COLLISION_V3    = -25.0     # > worst-case timeout stream (600 × -0.05 = -30 is
                               # bootstrapped by SB3, so stalling never "wins")
WR_STUCK_PENALTY_V3 = -0.10    # softer than v2 (-0.2): time penalty already bites

SAC_V3_TOTAL_STEPS = 200_000   # ~6 h at real-time 10 Hz — with odom obs + curriculum
                               # success is typically discovered in the first levels
SAC_V3_CKPT_DIR = Path("sac_wheel_v3_checkpoints")
SAC_V3_LOG_DIR  = Path("sac_wheel_v3_logs")
SAC_V3_TB_DIR   = Path("sac_wheel_v3_tensorboard")

TD3_V3_CKPT_DIR = Path("td3_wheel_v3_checkpoints")
TD3_V3_LOG_DIR  = Path("td3_wheel_v3_logs")
TD3_V3_TB_DIR   = Path("td3_wheel_v3_tensorboard")

# ══════════════════════════════════════════════════════════════════════════
# Explore-then-exit (v4) — 0.75 m cells, coverage + 2D map + exit search
# ══════════════════════════════════════════════════════════════════════════
# New task (like the two reference repos' free-roaming robots):
#   Phase EXPLORE — roam the whole maze, scan obstacles (the walls) with the
#     lidar, build a 2D occupancy map, and physically visit ALL 25 cells
#     without ever touching a wall.
#   Phase EXIT — once every cell has been visited (map complete), find the
#     way out: the single gap in the east border at row 2, and leave.
# Speed rule: approaching a wall ahead is only acceptable while slowing down —
#   enforced by a clearance-scaled speed penalty (same incentive as the
#   DRL-Robot-Navigation-ROS2 clearance reward, but tied to velocity).

EXPL_CELL_SIZE  = 0.75
EXPL_WALL_T     = 0.03
EXPL_PITCH      = EXPL_CELL_SIZE + EXPL_WALL_T      # 0.78 m
EXPL_GRID       = 5
EXPL_OUTER      = EXPL_GRID * EXPL_PITCH + EXPL_WALL_T   # 3.93 m
EXPL_ORIGIN_XY  = (2.0, -4.5)
EXPL_START_XY   = (2.405, -2.535)   # cell (col 0, row 2) center
EXPL_START_YAW  = 0.0
EXPL_EXIT_ROW   = 2
EXPL_EAST_X     = EXPL_ORIGIN_XY[0] + EXPL_GRID * EXPL_PITCH    # 5.90
EXPL_EXIT_XY    = (EXPL_EAST_X, -2.535)   # gap-center point on the east border

EXPL_WORLD_SDF  = WS_ROOT / "worlds" / "nhom8_maze75.sdf"
EXPL_WORLD_NAME = "nhom8_maze75"

# Robot control (same mecanum robot as v3: 2-DOF differential drive)
EXPL_W_MAX      = 20.0        # rad/s per wheel (rim ≈ 0.48 m/s; +33%)
EXPL_DT         = 0.10        # control period (s), 10 Hz lidar
EXPL_SENSOR_TIMEOUT = 5.0     # tolerate low-RTF 13-robot Gazebo sensor bursts
EXPL_SETTLE_SEC = 0.6

# ── Teleport verification ────────────────────────────────────────────────
# Reset teleports the robot to its maze start via the Gazebo `set_pose`
# service. When that service stops answering (a hung server, not a dead one —
# the watchdog's pgrep still sees the process) the old code logged a warning
# and carried on, so `reset` returned the robot's STALE pose as a legal start
# state and the next step scored an immediate -30 out-of-bounds terminal.
# Measured across two campaigns: 112/993 (v8) and 1059/1503 (v9) episodes,
# 100% of them ep_len == 1 with the terminal pose 7-15 m outside EVERY maze.
# Those -30 terminals went straight into the replay buffer and collapsed the
# policy (mean coverage 0.35 -> 0.00, mean ep_len 673 -> 1).
#
# Slots are 8 m apart, so the tolerance has to be far below that or a robot
# that landed in the maze next door would be scored against a maze it is not
# standing in. Settling jitter after a 0.6 s stationary settle is millimetres,
# so 0.25 m is generous by two orders of magnitude and still 30x clear of the
# nearest wrong answer.
EXPL_TELEPORT_TOL   = 0.25    # m; |observed - requested| accepted as arrival
EXPL_TELEPORT_TRIES = 3       # attempts before declaring the simulator dead
EXPL_MAX_STEPS  = 1500        # 150 s: full coverage (~35 m) + exit leg
# ...except that the budget turned out to be for coverage ALONE. v9 produced
# the first 25/25 episode (ortho_3, R=+127.99) and it TIMED OUT: exploration
# used all 1500 steps, so PHASE_EXIT started with nothing left and the robot
# was truncated on the step it earned the right to leave.
#
# The fix is an extension granted only when coverage completes, not a bigger
# EXPL_MAX_STEPS: raising the base would also stretch every failed
# exploration episode, and with 4 robots sharing one simulator that is pure
# throughput lost. At ~0.027 m of travel per step, 300 steps buys ~8 m of
# corridor — comfortably more than the longest geodesic from anywhere in a
# 5x5x0.75 m maze to its exit gap.
EXPL_EXIT_BUDGET = 300        # extra steps granted on entering PHASE_EXIT

# Observation: 4×36 LiDAR stack + odom [x, y, cos yaw, sin yaw]
#              + coverage fraction + exit-phase flag  → 150 dims
EXPL_STATE_DIM  = 4 * N_RAYS + 4 + 2
EXPL_MEMORY_GRID = 5          # optional odometry visitation bitmap (25 features)

# Reward — phase EXPLORE
EXPL_R_CELL      = 8.0        # one-time bonus per newly visited cell (25 → +200)
EXPL_R_MAP_DONE  = 20.0       # one-time bonus when the 25th cell is visited
EXPL_R_TIME      = -0.02      # per-step time penalty (standing still loses)
EXPL_STUCK_WINDOW   = 20
EXPL_STUCK_MIN_DISP = 0.08    # <8 cm net travel per 2 s window = not moving
EXPL_STUCK_PENALTY  = -0.10
EXPL_STALL_LIMIT    = 60      # 60 consecutive samples after window (~8 s) → episode
                              # ends with the collision penalty: standing
                              # still must never beat acting under discounting

# Reward — "slow down near a wall ahead" (clearance-scaled speed penalty)
EXPL_SAFE_CLEAR      = 0.35   # front-clearance threshold (m) — v4 legacy gate
EXPL_SAFE_SPEED_SCALE = -2.0  # r += scale · v · (1 − clear/thresh) when clear<thresh
EXPL_ROAM_BONUS      = 0.3    # bold-roaming bonus: r += bonus · v when the
                              # front is clear — brisk motion in open space
                              # earns, creeping earns nothing (v = m/s)

# v6: the safety gate reads MIN PER-RAY SLACK (scan − per-ray collision
# threshold, all 36 rays) instead of the ±45° front clearance. Measured on
# 916918 reachable configurations across the 13 mazes: 87.4% of deaths are
# triggered by side/rear rays, and front_clear separates "about to die" from
# "safe" with AUC 0.390 (worse than a coin flip — inverted), while min-slack
# scores 0.056 (i.e. 0.944 as a danger score). A 0.08 m margin warns before
# 97.1% of deaths at a 28.1% false-alarm rate (0.10 m → 99.7% / 38.7%).
EXPL_SAFE_MARGIN     = 0.08   # slack (m) below which the speed penalty bites
EXPL_ESCAPE_SLACK    = 0.06   # reactive turn threshold before the collision gate
EXPL_ESCAPE_TURN     = 0.65   # normalized differential-drive turn command

# ── v7: forward-biased action space ──────────────────────────────────────
# Measured on the 2026-09-06 run (240 episodes, 9971 steps, 0 successes):
# 79.6% of terminal contacts were REAR rays, 14.9% side, only 5.5% front.
# The cause is geometric, not behavioural. The lidar sits 0.08 m ahead of a
# 0.26 m chassis, so compute_collision_thresholds() gives 0.10 m at ray 0
# (front) but 0.26 m at ray 18 (rear). With the legacy symmetric [left,
# right] wheel action space a high-entropy SAC policy commands full reverse
# roughly a quarter of the time, and from a cell centre there are only
# 0.195 m / 4.1 control steps of margin at EXPL_W_MAX = 20 rad/s.
#
# "twist" maps the 2-D action to (forward, turn) with the forward channel
# squashed onto [-EXPL_REVERSE_FRAC, 1] instead of [-1, 1]: full forward
# authority (the user's speed requirement is unchanged), just enough reverse
# to back out of a dead end, and a zero action now drifts FORWARD rather
# than sitting still. "wheels" restores the legacy mapping bit-for-bit.
EXPL_ACTION_MODE   = "twist"  # "twist" (v7) | "wheels" (v4-v6 legacy)
EXPL_REVERSE_FRAC  = 0.25     # max reverse as a fraction of forward authority
EXPL_TURN_MAX      = 0.45     # differential term at |a_turn| = 1

# ── v7: front-cone safety shield ─────────────────────────────────────────
# EXPL_ESCAPE_SLACK = 0.06 m fires 1.2 control steps before the terminal
# contact at 0.48 m/s — far too late to change the outcome, which is why the
# shield never showed up in the outcome mix. A 15-line scripted reactive
# controller using the thresholds below (probe, 2026-09-06) survived 500
# steps on delta_1 with 6/25 zones and R = +44.8, against SAC's 39 steps,
# 1.5/25 zones and R = -27. The shield watches a FRONT CONE rather than the
# global min slack so that merely passing a side wall in a 0.75 m corridor
# (side slack 0.247 m when centred) does not trip it.
EXPL_SHIELD_CONE    = 35.0    # half-angle (deg) of the watched front cone
EXPL_SHIELD_SLOW    = 0.30    # front slack (m) below which forward is throttled
EXPL_SHIELD_TURN    = 0.18    # front slack (m) below which the shield turns
EXPL_SHIELD_FLOOR   = 0.45    # residual forward authority at zero front slack
EXPL_SHIELD_STREAK  = 150     # consecutive shield steps that count as a stall
EXPL_SHIELD_RELEASE = 0.26    # front slack (m) the pivot must recover to before
                              # the shield re-arms (hysteresis; see v9 below)

# ── v10: rotation lookahead ──────────────────────────────────────────────
# The bands above are sized for the TRANSLATION failure: at 0.48 m/s a step
# covers 0.048 m, so 0.18 m of slack is ~3.7 control steps of warning. The
# ROTATION failure is 2.5x faster and the bands never saw it. The collision
# threshold is a function of bearing (0.100 m at the nose, 0.127 m at the
# flank, 0.260 m astern — the lidar sits LIDAR_X_OFF forward of the chassis
# centre), and it steps by 0.0685 m between the 150° and 160° rays, where a
# ray stops striking the side of the chassis rectangle and starts striking
# its rear face. So a STATIONARY obstacle loses 0.0685 m of slack per 10° of
# yaw. At EXPL_TURN_MAX the chassis yaws 13.8°/step, i.e. 0.094 m of slack
# per step — 1.9 steps from the 0.18 m trigger to contact, too late to stop.
#
# Measured consequence, over the 355 collisions of the v9 run that survived
# the teleport bug: the breaching ray was FRONT on 1% (REAR 34%, LEFT 34%,
# RIGHT 30%), so the cone the shield watches is the one sector the robot
# almost never dies in. Raising the trigger cannot fix that — a threshold
# with room for a spin would fire continuously in a 0.75 m corridor, whose
# flank slack is 0.248 m. Predicting instead is both cheap and direction-
# aware: rotate the threshold curve by the yaw the command is ASKING for and
# re-subtract.
#
# The prediction is a VETO with its own small margin, NOT a second slack
# threshold. The first cut compared the global minimum of the prediction
# against EXPL_SHIELD_TURN, and measured over 1560 collision-free poses
# sampled across all 13 mazes that fires on 84% of them with a STRAIGHT
# command: a global minimum is not comparable to a cone minimum, because the
# rear threshold is 0.260 m and a wall 0.30 m astern — ordinary in a 0.75 m
# maze — already reads 0.04 m of slack while the robot is perfectly safe
# driving away from it. A guard that fires everywhere buys survival by
# refusing to explore, which is the failure that already cost this project a
# campaign. The margin below asks the narrow question instead: will the yaw
# this command is requesting leave anything at all?
#
# Measured on the 1094 of those poses that are not already hugging a wall:
# the cone rule alone fires on 30.1%, the veto adds 9.8 points, and 9.7 of
# those 9.8 are commands whose OPPOSITE turn is clear — so the shield mirrors
# the turn and the robot keeps driving. Only 0.1% are boxed in badly enough
# to reach the escape. At 0.06 m the escape share jumps to 6.3%, which is
# where the trap starts, so the margin stays under one ray-step of the
# threshold cliff.
EXPL_SHIELD_LOOKAHEAD = 2.0    # control steps of commanded yaw the veto looks
                               # ahead (0 disables the lookahead entirely)
EXPL_SHIELD_ROT_MARGIN = 0.04  # m of slack the predicted yaw must leave; below
                               # this the turn is mirrored, or escaped if both
                               # directions are doomed

# ── v9: motion smoothing ─────────────────────────────────────────────────
# Measured on the live v8 policy (checkpoint 310k, 2026-09-06). SAC draws a
# FRESH sample from its Gaussian every 100 ms control step, so on an
# IDENTICAL observation two consecutive turn commands differ by |da| = 0.33
# on average (p90 1.10) and flip SIGN on 17% of steps. At the v8
# EXPL_TURN_MAX = 0.90 the full-scale differential is 2*0.90*0.48/0.18 =
# 4.8 rad/s = 275 deg/s, so that sampling noise ALONE swung the heading by
# 9 deg per step on average and 30 deg at the p90 — the weave the operator
# reported watching in Gazebo on a straight corridor. Three changes:
#
#   1. EXPL_TURN_MAX 0.90 -> 0.45 (137 deg/s): a 90 deg junction turn still
#      takes only 0.66 s, but the noise amplitude halves.
#   2. A slew-rate cap on the (forward, turn) command. The policy keeps full
#      authority — it just has to hold an opinion for two consecutive steps
#      to spend it, which uncorrelated noise cannot do.
#   3. An action-rate penalty, so smoothness is TRAINED rather than merely
#      filtered: a policy that saturates the limiter every step still weaves
#      at the limit, and nothing in the v8 reward ever charged it for that.
#
# Rates are in normalized command units per control step.
EXPL_FWD_RATE      = 0.35     # max |d forward| per step (full stop->go in 3)
EXPL_TURN_RATE     = 0.22     # max |d turn| per step (full-scale turn in 5)
EXPL_R_SMOOTH      = -0.50    # r += scale * (d_fwd^2 + d_turn^2)

# v6: discount horizon. gamma 0.99 at dt=0.10 s = 100 steps = 10 s, against a
# 1500-step (150 s) episode: a zone 4 cells away was worth 8·0.99^300 = 0.39
# and EXPL_R_EXIT was worth 0.03 — less than one step of roam bonus, so
# farming the roam bonus was the arithmetically optimal policy. 0.997 gives a
# 333-step (33 s) horizon without the variance of 0.999.
EXPL_GAMMA           = 0.997

# v6: entropy floor. The v4 run relapsed into the creep trap at t≈362k as
# ent_coef auto-tuned down to 0.0089 (policy went near-deterministic). SAC's
# default target entropy is −dim(A) = −2.0; −1.0 keeps the policy exploring.
EXPL_TARGET_ENTROPY  = -1.0

# ── v8: EXPLORE-phase geodesic shaping ───────────────────────────────────
# The EXIT phase has had potential shaping since v4, but EXPLORE — where the
# agent actually spends its 1500 steps — had only the one-shot +8 per zone.
# With EXPL_GAMMA = 0.997 a zone four cells away is worth 8·0.997^160 = 4.9
# at best and nothing at all if the agent never stumbles into it, so the
# gradient the critic sees between "walking toward an unvisited zone" and
# "walking away from it" was zero. These constants add the same Ng et al.
# potential term the exit leg uses, computed on ONE geodesic field per zone
# (rl_training/zone_field.py) against the nearest UNVISITED zone.
EXPL_N_ZONES        = 25      # watershed zones per maze (maze_registry raster)
EXPL_ZONE_FIELD_RES = 0.025   # raster resolution (m) — matches the exit field
EXPL_ZONE_POT_SCALE = 2.0     # scale · Δ(geodesic dist to nearest unvisited)
EXPL_ZONE_DESCENT_R = 6       # descent-direction search radius, in cells
                              # (0.15 m: past the 0.10 m inflation band, so a
                              # robot hugging a wall still finds a free cell)

# Reward — phase EXIT
EXPL_EXIT_POT_SCALE = 2.0     # potential shaping: scale · Δ(geodesic dist to exit)
EXPL_R_EXIT        = 100.0    # terminal: left the maze with a complete map
EXPL_R_EXIT_EARLY  = -30.0    # terminal: crossed the exit before the map was done
EXPL_R_COLLISION   = -30.0    # terminal: touched a wall (hard rule: never touch)

# 2D occupancy map (built from lidar + pose; the maze walls are the obstacles)
EXPL_MAP_RES     = 0.05       # grid resolution (m)
EXPL_MAP_MARGIN  = 0.10       # grid extends this far beyond the maze walls
EXPL_MAP_EVERY   = 5          # publish /map every N env steps (2 Hz)
EXPL_MAP_DIR     = Path("explore_maps")

# Training artifacts
SAC_V4_TOTAL_STEPS = 400_000
SAC_V4_CKPT_DIR = Path("sac_explore_checkpoints")
SAC_V4_LOG_DIR  = Path("sac_explore_logs")
SAC_V4_TB_DIR   = Path("sac_explore_tensorboard")

# ══════════════════════════════════════════════════════════════════════════
# Multi-maze registry (v5) — 13 decoded mazes in worlds/maze_defs/*.json
# ══════════════════════════════════════════════════════════════════════════
# The 13 mazes decoded from mazes_png/*.png (ortho/sigma/delta families) are
# stored maze-local: origin = wall-bbox center, +y up, longest side 3.93 m,
# walls 0.03 m thick / 0.40 m tall (same conventions as the 0.75 m maze).
# The original EXPL_* section above is kept untouched for v4 reproducibility.

MAZE_DEFS_DIR = WS_ROOT / "worlds" / "maze_defs"

# Standalone per-maze worlds place the maze wall-bbox CENTER at the center of
# the old 0.75 m maze footprint, so lighting/floor/spawn conventions are
# unchanged (decision 2026-08-30). EXPL_ORIGIN_XY + EXPL_OUTER / 2 = (3.965, -2.535).
MAZE_CENTER_XY = (EXPL_ORIGIN_XY[0] + EXPL_OUTER / 2.0,
                  EXPL_ORIGIN_XY[1] + EXPL_OUTER / 2.0)

# Combined world: all 13 mazes side by side (4x4 grid, 8 m center spacing —
# > LIDAR_MAX 3 m, so neighbouring mazes cannot cross-talk through the lidar).
# One Gazebo instance hosts this world and the env switches maze per episode
# by teleporting to the selected maze's start pose (gz cannot hot-swap worlds).
MAZE_MULTI_WORLD_NAME = "nhom8_maze_multi"
MAZE_MULTI_SDF       = WS_ROOT / "worlds" / f"{MAZE_MULTI_WORLD_NAME}.sdf"
MAZE_MULTI_SPACING   = 8.0
MAZE_MULTI_COLS      = 4

# Per-episode maze selection inside the combined world.
EXPL_V5_SELECTION = "round_robin"      # "round_robin" | "random"

# v5 training artifacts (kept separate from v4 sac_explore_* outputs).
EXPL_V5_CKPT_DIR = Path("sac_explore_v5_checkpoints")
EXPL_V5_LOG_DIR  = Path("sac_explore_v5_logs")
EXPL_V5_TB_DIR   = Path("sac_explore_v5_tensorboard")
