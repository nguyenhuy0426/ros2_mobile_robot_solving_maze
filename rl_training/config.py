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
WHEEL_RADIUS = 0.024

# ══════════════════════════════════════════════════════════════════════════
# Lidar
# ══════════════════════════════════════════════════════════════════════════
N_RAYS       = 36         # 360° / 10° = 36 rays
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

WHEEL_W_MAX      = 15.0           # rad/s per wheel (rim ≈ 0.36 m/s at r=0.024)
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
EXPL_W_MAX      = 15.0        # rad/s per wheel (rim ≈ 0.36 m/s)
EXPL_DT         = 0.10        # control period (s), 10 Hz lidar
EXPL_SETTLE_SEC = 0.6
EXPL_MAX_STEPS  = 1500        # 150 s: full coverage (~35 m) + exit leg

# Observation: 4×36 LiDAR stack + odom [x, y, cos yaw, sin yaw]
#              + coverage fraction + exit-phase flag  → 150 dims
EXPL_STATE_DIM  = 4 * N_RAYS + 4 + 2

# Reward — phase EXPLORE
EXPL_R_CELL      = 8.0        # one-time bonus per newly visited cell (25 → +200)
EXPL_R_MAP_DONE  = 20.0       # one-time bonus when the 25th cell is visited
EXPL_R_TIME      = -0.02      # per-step time penalty (standing still loses)
EXPL_STUCK_WINDOW   = 20
EXPL_STUCK_MIN_DISP = 0.08    # <8 cm net travel per 2 s window = not moving
EXPL_STUCK_PENALTY  = -0.10
EXPL_STALL_LIMIT    = 4       # 4 consecutive stuck windows (~8 s) → episode
                              # ends with the collision penalty: standing
                              # still must never beat acting under discounting

# Reward — "slow down near a wall ahead" (clearance-scaled speed penalty)
EXPL_SAFE_CLEAR      = 0.35   # front-clearance threshold (m)
EXPL_SAFE_SPEED_SCALE = -2.0  # r += scale · v · (1 − clear/thresh) when clear<thresh
EXPL_ROAM_BONUS      = 0.3    # bold-roaming bonus: r += bonus · v when the
                              # front is clear — brisk motion in open space
                              # earns, creeping earns nothing (v = m/s)

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
