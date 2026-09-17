"""Evaluate a frozen neural checkpoint on generated, training-free mazes."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import subprocess
import sys


def build_episode_jobs(seeds, repeats=1, episode_seed_offset=0):
    """Keep maze geometry fixed while varying the independent episode seed."""
    if repeats < 1:
        raise ValueError("repeats must be positive")
    return [(maze_index, repeat, maze_seed,
             maze_seed + episode_seed_offset + repeat * 1_000_003)
            for maze_index, maze_seed in enumerate(seeds)
            for repeat in range(repeats)]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default="hybrid_runs/release_latest/local_policy.pt")
    p.add_argument("--output", default="hybrid_runs/evaluations/generated_holdouts")
    p.add_argument("--seeds", type=int, nargs="+", default=list(range(31001, 31006)))
    p.add_argument("--seconds", type=float, default=180.)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--domain-base", type=int, default=140)
    p.add_argument("--size", type=int, default=5)
    p.add_argument("--cell", type=float, default=.75)
    p.add_argument("--yaw-jitter", type=float, default=0.)
    p.add_argument("--repeats", type=int, default=1,
                   help="episodes per fixed maze geometry")
    p.add_argument("--episode-seed-offset", type=int, default=0,
                   help="change spawn/control randomness without changing maze geometry")
    p.add_argument("--localization", choices=["truth", "command_odom", "scan_match"], default="truth")
    p.add_argument("--lidar-noise-std", type=float, default=0.)
    p.add_argument("--lidar-dropout", type=float, default=0.)
    p.add_argument("--odom-linear-scale", type=float, default=1.)
    p.add_argument("--odom-angular-scale", type=float, default=1.)
    args = p.parse_args()
    if not 1 <= args.workers <= 4 or args.repeats < 1:
        p.error("workers must be in 1..4 and repeats must be positive")
    root = Path(args.output).resolve()
    root.mkdir(parents=True, exist_ok=False)
    jobs = build_episode_jobs(args.seeds, args.repeats, args.episode_seed_offset)

    def worker(slot):
        rows = []
        for job_index in range(slot, len(jobs), args.workers):
            index, repeat, seed, episode_seed = jobs[job_index]
            suffix = f"_r{repeat + 1:02d}" if args.repeats > 1 else ""
            out = root / f"{index:02d}_holdout_{seed}{suffix}"
            out.mkdir()
            cmd = [sys.executable, "-m", "rl_training.eval_hybrid_gazebo",
                   "--holdout-seed", str(seed), "--model", args.model,
                   "--holdout-size", str(args.size), "--holdout-cell", str(args.cell),
                   "--seed", str(episode_seed), "--yaw-jitter", str(args.yaw_jitter),
                   "--localization", args.localization,
                   "--lidar-noise-std", str(args.lidar_noise_std),
                   "--lidar-dropout", str(args.lidar_dropout),
                   "--odom-linear-scale", str(args.odom_linear_scale),
                   "--odom-angular-scale", str(args.odom_angular_scale),
                   "--output", str(out), "--seconds", str(args.seconds),
                   "--domain-id", str(args.domain_base + slot)]
            with (out / "worker.log").open("w") as log:
                subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT,
                               check=False, timeout=args.seconds + 100)
            result_file = out / "result.json"
            result = json.loads(result_file.read_text()) if result_file.exists() else {
                "maze": f"holdout_{args.size}x{args.size}_{seed}", "success": False,
                "reason": "worker_no_result"}
            result["artifact_dir"] = str(out)
            result["maze_seed"] = seed
            result["repeat"] = repeat + 1
            result["episode_seed"] = episode_seed
            rows.append(result)
            print(seed, f"repeat={repeat + 1}", result.get("success"),
                  result.get("reason"), flush=True)
        return rows

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        rows = [row for future in [pool.submit(worker, slot) for slot in range(args.workers)]
                for row in future.result()]
    rows.sort(key=lambda row: (row["maze_seed"], row["repeat"]))
    per_maze = {str(seed): {"successes": sum(bool(row.get("success")) for row in rows
                                                    if row["maze_seed"] == seed),
                            "episodes": sum(row["maze_seed"] == seed for row in rows)}
                for seed in args.seeds}
    scope = ("generated holdout mazes, known map + Gazebo pose; not SLAM"
             if args.localization == "truth" else
             f"generated holdout mazes, known map + {args.localization}; Gazebo pose only audits results; not SLAM")
    payload = {"scope": scope,
               "seeds": args.seeds, "episodes": len(rows), "repeats": args.repeats,
               "episode_seed_offset": args.episode_seed_offset,
               "localization": args.localization,
               "lidar_noise_std_m": args.lidar_noise_std,
               "lidar_dropout": args.lidar_dropout,
               "odom_linear_scale": args.odom_linear_scale,
               "odom_angular_scale": args.odom_angular_scale,
               "size": args.size, "cell_m": args.cell, "yaw_jitter_rad": args.yaw_jitter,
               "successes": sum(bool(row.get("success")) for row in rows),
               "wall_contacts": sum(row.get("wall_contacts", 0) for row in rows),
               "per_maze": per_maze,
               "results": rows}
    (root / "summary.json").write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps({k: v for k, v in payload.items() if k != "results"}), flush=True)
    return 0 if payload["successes"] == len(rows) and payload["wall_contacts"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
