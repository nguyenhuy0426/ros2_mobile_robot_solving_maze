# CLAUDE.md

Behavioral guidelines to reduce common LLM coding mistakes. Merge with project-specific instructions as needed.

**Tradeoff:** These guidelines bias toward caution over speed. For trivial tasks, use judgment.

## 1. Think Before Coding

**Don't assume. Don't hide confusion. Surface tradeoffs.**

Before implementing:
- State your assumptions explicitly. If uncertain, ask.
- If multiple interpretations exist, present them - don't pick silently.
- If a simpler approach exists, say so. Push back when warranted.
- If something is unclear, stop. Name what's confusing. Ask.

## 2. Simplicity First

**Minimum code that solves the problem. Nothing speculative.**

- No features beyond what was asked.
- No abstractions for single-use code.
- No "flexibility" or "configurability" that wasn't requested.
- No error handling for impossible scenarios.
- If you write 200 lines and it could be 50, rewrite it.

Ask yourself: "Would a senior engineer say this is overcomplicated?" If yes, simplify.

## 3. Surgical Changes

**Touch only what you must. Clean up only your own mess.**

When editing existing code:
- Don't "improve" adjacent code, comments, or formatting.
- Don't refactor things that aren't broken.
- Match existing style, even if you'd do it differently.
- If you notice unrelated dead code, mention it - don't delete it.

When your changes create orphans:
- Remove imports/variables/functions that YOUR changes made unused.
- Don't remove pre-existing dead code unless asked.

The test: Every changed line should trace directly to the user's request.

## 4. Goal-Driven Execution

**Define success criteria. Loop until verified.**

Transform tasks into verifiable goals:
- "Add validation" → "Write tests for invalid inputs, then make them pass"
- "Fix the bug" → "Write a test that reproduces it, then make it pass"
- "Refactor X" → "Ensure tests pass before and after"

For multi-step tasks, state a brief plan:
```
1. [Step] → verify: [check]
2. [Step] → verify: [check]
3. [Step] → verify: [check]
```

Strong success criteria let you loop independently. Weak criteria ("make it work") require constant clarification.

---

## Project-Specific Context

### System: GA-NN Mecanum Maze Solver

This project trains a mecanum robot to solve a specific 5×5 maze using a Genetic Algorithm with neural network controllers, evaluated in Gazebo simulation.

### Maze Geometry (invariants — do not change)
- **Grid:** 5×5 cells, 0.5m per cell
- **Corridor width:** ~0.44m (0.5m cell − 0.03m wall × 2)
- **START:** (2.25, −3.25), **GOAL:** (4.25, −3.25)
- **START_DIST:** 2.0m (straight-line)
- **Maze bounds:** x∈[1.85, 4.65], y∈[−4.65, −1.85]
- Multi-maze: Y-offset 3.5m per maze (DY_STEP in gen_multi_maze.py)

### Robot (invariants — do not change)
- **Chassis:** 0.26×0.155×0.08m, **Wheel radius:** 0.024m
- **Lidar:** 36 rays, 360°, range 0.05–12m, 10Hz
- **Kinematics:** Mecanum forward kinematics in `MLP.to_twist()`
- **Spawn height:** z = 0.024m (wheel radius)

### Training Architecture
- `run.py` — main training script (all logic in one file)
- Parallel evaluation: N mazes × continuous queue → no lidar cross-talk
- Fitness: minimize J = Penalties − Rewards (more negative = better)
- Checkpoints: `ga_checkpoints_v2/`, auto-resume from `last_run.json`

### Key Design Decisions
- **No wall avoidance override:** The NN must learn to avoid walls itself via collision penalty. External safety systems prevent learning in narrow corridors.
- **Allow reverse:** `VX_MIN = -0.10` — essential for dead-end recovery.
- **12 lidar rays at 30° spacing:** Full 360° coverage with no blind spots (critical for 0.44m corridors).
- **Goal angle in robot frame:** Rotation-invariant representation.

---

**These guidelines are working if:** fewer unnecessary changes in diffs, fewer rewrites due to overcomplication, and clarifying questions come before implementation rather than after mistakes.
