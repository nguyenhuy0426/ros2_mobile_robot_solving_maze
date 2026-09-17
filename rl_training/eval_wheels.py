#!/usr/bin/env python3
"""
eval_wheels.py — Deterministic evaluation for SAC/TD3 wheel policies.

Supersedes eval_sac_wheels.py: works for either algorithm (``--algo``) and
reports the diagnostics the v1 analysis showed were missing — stuck rate,
action saturation, exploration coverage, and final distance to goal — in
addition to success/collision/timeout/length/reward.

Evaluation always runs from the real START_XY to GOAL (curriculum OFF) so the
number reflects the actual maze-solving task. Shaping is ON only to populate
coverage/stuck telemetry; it does not change termination or the reported task
outcome.

Example:
  source /opt/ros/jazzy/setup.bash
  ./rl_venv/bin/python -m rl_training.eval_wheels \\
      --algo sac --model sac_wheel_v2_checkpoints/sac_wheels_v2_final --episodes 20
"""

import argparse
import sys
from pathlib import Path

try:
    import rclpy
except ImportError:
    sys.exit("rclpy not found — run: source /opt/ros/jazzy/setup.bash")

import numpy as np
from stable_baselines3 import SAC, TD3

from rl_training import config as C
from rl_training.wheel_env import make_wheel_env

_ALGOS = {"sac": SAC, "td3": TD3}


def main() -> None:
    ap = argparse.ArgumentParser(description="Evaluate a wheel policy")
    ap.add_argument("--algo", choices=sorted(_ALGOS), default="sac")
    ap.add_argument("--model", type=str, required=True,
                    help="path to SB3 model (.zip, extension optional)")
    ap.add_argument("--episodes", type=int, default=20)
    ap.add_argument("--robot", type=int, default=1)
    args = ap.parse_args()

    model_path = Path(args.model)
    if not model_path.exists() and not model_path.with_suffix(".zip").exists():
        sys.exit(f"model not found: {args.model}")

    model = _ALGOS[args.algo].load(str(model_path))

    # Build an env that matches the checkpoint's policy interface:
    # v1/v2 models are LiDAR-only (144-d) with 4 wheel outputs; v3 models
    # carry odometry in the observation and use 2-DOF diff-drive actions.
    obs_dim = model.observation_space.shape[0]
    act_dim = model.action_space.shape[0]
    use_odom = obs_dim > C.WHEEL_FRAME_STACK * C.N_RAYS
    diff_drive = act_dim == 2
    print(f"policy interface: obs={obs_dim} action={act_dim} "
          f"(odom={use_odom}, diff_drive={diff_drive})")

    # Shaping ON for telemetry; curriculum OFF → evaluate the real task.
    env = make_wheel_env(robot_id=args.robot, shaping=True, curriculum=False,
                         use_odom=use_odom, diff_drive=diff_drive)

    successes = collisions = 0
    lengths, rewards, final_geo = [], [], []
    saturations, stuck_rates, coverages, min_geos = [], [], [], []
    try:
        for ep in range(1, args.episodes + 1):
            obs, info = env.reset()
            done = False
            ep_reward = ep_len = 0
            sat_acc = stuck_acc = 0.0
            min_geo = info.get("geo_dist", float("nan"))
            max_cells = 0
            while not done:
                action, _ = model.predict(obs, deterministic=True)
                obs, r, terminated, truncated, info = env.step(action)
                ep_reward += r
                ep_len += 1
                sat_acc += info.get("saturation", 0.0)
                stuck_acc += float(info.get("stuck", False))
                min_geo = min(min_geo, info.get("geo_dist", min_geo))
                # coverage_cells is only set on non-terminal shaped steps, so
                # carry the running max rather than reading the final info.
                max_cells = max(max_cells, info.get("coverage_cells", max_cells))
                done = terminated or truncated

            outcome = ("SUCCESS" if info.get("success")
                       else "COLLISION" if info.get("collision")
                       else "TIMEOUT")
            successes += int(info.get("success", False))
            collisions += int(info.get("collision", False))
            lengths.append(ep_len)
            rewards.append(ep_reward)
            final_geo.append(info.get("geo_dist", float("nan")))
            saturations.append(sat_acc / max(ep_len, 1))
            stuck_rates.append(stuck_acc / max(ep_len, 1))
            coverages.append(max_cells)
            min_geos.append(min_geo)
            print(f"ep {ep:3d}/{args.episodes}: {outcome:9s} "
                  f"steps={ep_len:4d} reward={ep_reward:8.2f} "
                  f"geo_final={final_geo[-1]:.2f} geo_min={min_geo:.2f} "
                  f"sat={saturations[-1]:.0%} stuck={stuck_rates[-1]:.0%} "
                  f"cells={coverages[-1]}")
    finally:
        env.close()
        if rclpy.ok():
            rclpy.shutdown()

    n = len(lengths)
    if n == 0:
        sys.exit("no episodes completed")
    timeouts = n - successes - collisions
    print("\n══════════ Evaluation summary ══════════")
    print(f"algorithm       : {args.algo}")
    print(f"episodes        : {n}")
    print(f"success rate    : {successes / n:.1%}")
    print(f"collision rate  : {collisions / n:.1%}")
    print(f"timeout rate    : {timeouts / n:.1%}")
    print(f"avg episode len : {np.mean(lengths):.1f} steps")
    print(f"avg reward      : {np.mean(rewards):.2f}")
    print(f"avg final dist  : {np.nanmean(final_geo):.2f} m (goal = 0)")
    print(f"best geo reached : {np.nanmin(min_geos):.2f} m")
    print(f"avg saturation  : {np.mean(saturations):.1%} of wheel commands")
    print(f"avg stuck rate  : {np.mean(stuck_rates):.1%} of steps")
    print(f"avg coverage    : {np.mean(coverages):.1f} maze cells "
          f"(of {C.CURR_GRID_CELLS**2})")


if __name__ == "__main__":
    main()
