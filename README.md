# GA-NN Mecanum Maze Solver

Genetic Algorithm with Neural Network controller for training a mecanum-wheeled robot to solve a specific 5×5 maze in Gazebo simulation.

> **Note:** The current version is trained on a **specific maze layout** (`nhom8_maze.sdf`). Generalization to arbitrary mazes would require curriculum learning, domain randomization, or a different maze representation — this is left for future development.

## System Overview

```
┌─────────────────────────────────────────────────────────────┐
│                     GA Training Loop                        │
│                                                             │
│  Population of N genomes (neural network weights)           │
│       ↓                                                     │
│  For each genome:                                           │
│    1. Teleport robot to START                               │
│    2. Run episode: observe → NN forward → act → repeat      │
│    3. Compute fitness based on progress/goal/collision       │
│       ↓                                                     │
│  GA: selection → crossover → mutation → next generation     │
└─────────────────────────────────────────────────────────────┘
```

### Neural Network

| Layer   | Size | Activation |
|---------|------|------------|
| Input   | 16   | —          |
| Hidden1 | 16   | tanh       |
| Hidden2 | 8    | tanh       |
| Output  | 4    | clip[-1,1] |

**Total weights:** 444

### Observation Space (16 inputs)

| Index   | Description                              |
|---------|------------------------------------------|
| [0:12]  | 12 lidar rays (every 30°), normalized    |
| [12]    | Goal angle in robot frame / π            |
| [13]    | Normalized distance to goal              |
| [14:16] | Heading: sin(yaw), cos(yaw)              |

### Action Space

The 4 outputs represent mecanum wheel velocities → converted to (vx, vy, wz) via forward kinematics. Reverse motion is allowed for dead-end recovery.

### Fitness Function (minimize J)

```
J = Penalties − Rewards

Penalties:
  + 300 × collision
  + 300 × out_of_bounds
  + 100 × stationary
  +  50 × timeout
  + 0.5 × steps (time pressure)

Rewards:
  + 500 × progress (normalized 0-1)
  + 250 × best_progress (normalized 0-1)
  + 2.0 × unique_cells
  + 5000 (goal reached)
  + max(0, 3000 − steps×5) (speed bonus)
```

## File Structure

```
ros2_ws/
├── run.py              # Main GA training script
├── run_last.py         # Demo helper (runs last checkpoint)
├── spawn_robot.sh      # Spawn N robots + ROS2 bridges
├── gen_multi_maze.py   # Generate multi-maze SDF from single maze
├── one_robot.sdf       # Robot model template (mecanum 4WD + lidar)
├── nhom8_maze.sdf      # Single 5×5 maze world
├── nhom8_multi5.sdf    # 5 parallel mazes (generated)
├── CLAUDE.md           # AI coding guidelines
├── README.md           # This file
└── src/                # ROS2 workspace packages
    ├── my_pkg/         # Basic ROS2 package
    └── my_interfaces/  # Custom ROS2 messages
```

## Quick Start

### Prerequisites

- ROS2 Humble/Iron
- Gazebo Harmonic (gz-sim)
- `ros_gz_bridge`, `ros_gz_sim`
- Python: `numpy`, `rclpy`

### Step 1: Launch Gazebo

```bash
# Single maze (for demo/testing):
gz sim nhom8_maze.sdf

# 5 parallel mazes (for training):
gz sim nhom8_multi5.sdf
```

### Step 2: Spawn Robots

```bash
./spawn_robot.sh 5   # spawns robot_1..robot_5 with bridges
```

### Step 3: Train

```bash
# Start fresh training (50 genomes, 200 generations, 5 workers):
python3 run.py --pop 50 --gen 200 --workers 5 --fresh

# Resume from checkpoint:
python3 run.py --gen 200 --workers 5

# Resume from specific checkpoint:
python3 run.py --load ga_checkpoints_v2/best.json --workers 5
```

### Step 4: Demo

```bash
# Run best genome:
python3 run.py --run ga_checkpoints_v2/best.json --workers 1

# Quick demo from last training:
python3 run_last.py
```

### View Architecture

```bash
python3 run.py --info
```

## Training Parameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| POP       | 50    | Population size |
| ELITE     | 5     | Preserved top genomes |
| MUT_P     | 0.12  | Base mutation probability (adaptive) |
| MUT_STD   | 0.25  | Base mutation std dev (decays) |
| MAX_STEPS | 600   | Steps per episode |
| STEP_DT   | 0.20s | Time per step |
| GOAL_R    | 0.20m | Goal radius |
| STAG_LIM  | 20    | Generations before diversity injection |

### Expected Training Time

With 5 workers: ~15-20 min/generation (varies with genome quality).

- **50 gens** (~15 hours): Robot learns to navigate corridors
- **100 gens** (~30 hours): Consistent progress toward goal
- **200+ gens** (~48-72 hours): Goal-reaching behavior

## Maze Geometry

- **Grid:** 5×5 cells, each 0.5m × 0.5m
- **Corridor width:** ~0.44m (cell minus wall thickness)
- **START:** (2.25, -3.25) — bottom-left
- **GOAL:** (4.25, -3.25) — bottom-right
- **Path length:** ~5m through the maze

## Robot Specifications

| Property | Value |
|----------|-------|
| Chassis  | 0.26 × 0.155 × 0.08 m |
| Wheel radius | 0.024 m |
| Drive    | Mecanum 4WD (omnidirectional) |
| Lidar    | 36 rays, 360°, 0.05-12m range |
| Max vx   | 0.20 m/s (forward), -0.10 m/s (reverse) |
| Max vy   | 0.15 m/s |
| Max wz   | 2.0 rad/s |
