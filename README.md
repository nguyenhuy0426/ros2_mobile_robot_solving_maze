# Neural LiDAR Maze Navigation for ROS 2 and Gazebo

This repository contains the promoted maze-navigation stack for a four-wheel
robot in ROS 2 Jazzy and Gazebo. The deployed controller combines:

- a neural local policy that maps LiDAR and a local subgoal to wheel commands;
- a known-map global planner for route selection;
- LiDAR scan matching over command odometry for localization;
- a geometric safety filter that can reject unsafe commands.

The neural network produces every motion command. The planner supplies a local
target and the safety layer enforces the no-wall-contact requirement.

## Current release

Only the promoted weights and compact validation evidence are stored in
`hybrid_runs/release_latest/`:

| Model | Architecture | SHA-256 |
|---|---|---|
| `local_policy.pt` | 38 -> 64 -> 64 -> 2 | `987c703af37e394f2886f480eb958a6cd78654a604b946e9966370a871b7cd6a` |
| `tracking_policy.pt` | 42 -> 96 -> 96 -> 2 | `7a405bc7e6ca3a9f3746c94e55977d624370605a627e638aae21e5eb256750db` |

Validation completed through 2026-09-18:

- 13/13 fixed-maze Gazebo episodes passed, covering 12 distinct geometries;
- 90/90 generated-maze episodes passed across 45 distinct geometries, with
  every geometry rerun using a different episode seed;
- 15/15 additional Gazebo stress episodes passed with 1.5 cm Gaussian LiDAR
  noise, 5% ray dropout, +3% linear odometry scale error, -3% angular odometry
  scale error, and up to 0.5 rad initial-yaw variation;
- 20/20 further unseen generated geometries passed under the same sensor and
  odometry stress, spanning 5x5, 6x6, and 7x7 mazes with 0.60-0.80 m cells;
- zero wall contacts and zero geometry collisions were observed in these
  release gates.

The fixed and generated cohorts cover 77 distinct successful geometries in
total (12 fixed and 65 generated). The 45-geometry reload cohort was run twice;
the newest 20-geometry cohort used disjoint generation and episode seeds. Its
largest audited pose error was 0.077 m, and 59 inconsistent scan matches were
rejected instead of being applied to the controller pose.

These results show repeatability in Gazebo. They do not prove operation in all
possible mazes or on physical hardware.

## Scope

The current release requires a known maze map. In `scan_match` mode, Gazebo
ground truth is subscribed only to audit goal completion, collisions, and pose
error; it is not used by the controller. Online map construction and SLAM are
the next major milestone and are not implemented in this release.

## Setup

Requirements:

- Ubuntu with ROS 2 Jazzy;
- Gazebo Harmonic and `ros_gz_bridge`;
- Python 3.12.

Create a local environment instead of committing one:

```bash
python3 -m venv --system-site-packages .venv
.venv/bin/pip install -r rl_training/requirements-hybrid.txt
source /opt/ros/jazzy/setup.bash
```

## Run one visible Gazebo evaluation

```bash
source /opt/ros/jazzy/setup.bash
.venv/bin/python -m rl_training.eval_hybrid_gazebo \
  --maze delta_1 \
  --navigation auto \
  --localization scan_match \
  --gui \
  --keep-open 10 \
  --output hybrid_runs/evaluations/delta_1_gui
```

The command loads the promoted models by default. The generated `result.json`
records the weight hash, termination reason, contact audit, localization mode,
and pose error.

## Run generated holdouts with sensor stress

```bash
source /opt/ros/jazzy/setup.bash
.venv/bin/python -m rl_training.eval_holdouts_gazebo \
  --output hybrid_runs/evaluations/noisy_holdouts \
  --seeds 31001 31002 31003 31004 31005 \
  --workers 4 \
  --yaw-jitter 0.5 \
  --localization scan_match \
  --lidar-noise-std 0.015 \
  --lidar-dropout 0.05 \
  --odom-linear-scale 1.03 \
  --odom-angular-scale 0.97
```

Maze generation and episode randomness use separate seeds. Use `--repeats` and
`--episode-seed-offset` to rerun the same geometry independently.

## Train and promote a candidate

`train_hybrid_multi.py` runs up to four isolated Gazebo workers, collects DAgger
data, trains candidate local and tracking policies, and validates candidates
before selection:

```bash
source /opt/ros/jazzy/setup.bash
.venv/bin/python -m rl_training.train_hybrid_multi \
  --workers 4 \
  --rounds 1 \
  --include-sigma \
  --procedural-train-seeds 41001 41002 41003 41004 \
  --procedural-validation-seeds 42001 42002 42003 42004
```

Training output is intentionally ignored by Git. Copy only a candidate that
passes the fixed, generated, reload, localization, and contact gates into
`hybrid_runs/release_latest/`, then update its manifest and evidence.

## Tests

```bash
source /opt/ros/jazzy/setup.bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv/bin/pytest -q \
  rl_training/tests/test_{command_odometry,holdout_mazes,holdout_reliability,hybrid_multi,hybrid_navigation,oriented_navigation,sigma_se2}.py
```

The promoted-release suite contains 53 tests. The complete research suite also
contains tests for older SAC, TD3, and exploration experiments and therefore
requires their optional dependencies.

## Relevant files

```text
rl_training/
  eval_hybrid_gazebo.py       Gazebo controller, contact audit, noisy-sensor stress
  eval_holdouts_gazebo.py     parallel generated-maze evaluation
  holdout_mazes.py            deterministic held-out maze generation
  hybrid_navigation.py        disk-footprint planner, local policy, safety filter
  oriented_navigation.py      orientation-aware tracker for constrained mazes
  train_hybrid_multi.py       multi-worker DAgger and promotion pipeline
  tests/                      deterministic unit and integration tests
worlds/maze_defs/             fixed maze registry
hybrid_runs/release_latest/   promoted weights and compact evidence only
```
