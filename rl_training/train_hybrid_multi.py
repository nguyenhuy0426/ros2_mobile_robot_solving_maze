"""Synchronous multi-robot DAgger with isolated Gazebo workers.

Each robot owns its simulator, transport partition and ROS domain. Workers
collect labels from a local teacher; one learner updates the shared neural
policy between rounds. Validation has no teacher actions. These experiments
use known maps and Gazebo pose, not SLAM. No old SAC run is resumed.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from datetime import datetime
import hashlib
import json
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time
import threading

import numpy as np
import torch

from rl_training.eval_hybrid import maze_setup
from rl_training.holdout_mazes import generate_holdout
from rl_training.hybrid_navigation import H, LocalPolicy
from rl_training.maze_registry import load_registry
from rl_training.train_hybrid import samples, teacher


TRAIN_MAZES = ["delta_1", "delta_2", "ortho_1", "ortho_2", "ortho_3", "ortho_4"]
VALIDATION_MAZES = ["ortho_5", "ortho_6"]
STOP_REQUESTED = threading.Event()


@dataclass(frozen=True)
class ProceduralMaze:
    """A reproducible generated maze that can be scheduled like a registry maze."""

    seed: int
    size: int = 5
    cell: float = .75

    @property
    def name(self):
        return generate_holdout(self.seed, size=self.size, cell=self.cell).name

    def evaluator_args(self):
        return ["--holdout-seed", str(self.seed),
                "--holdout-size", str(self.size), "--holdout-cell", str(self.cell)]


def maze_name(maze):
    return maze if isinstance(maze, str) else maze.name


def evaluator_maze_args(maze):
    return ["--maze", maze] if isinstance(maze, str) else maze.evaluator_args()


def manifest_maze(maze):
    return maze if isinstance(maze, str) else {"name": maze.name, **asdict(maze)}


def write_json(path, data):
    path = Path(path)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=2) + "\n")
    tmp.replace(path)


def score(rows):
    """Safety first, then solve rate; failures stay in the denominator."""
    collisions = sum(bool(r.get("geometry_collision")) or r.get("wall_contacts", 0) > 0 for r in rows)
    successes = sum(bool(r.get("success")) and r.get("contact_monitor_observed", False)
                    and not r.get("geometry_collision") and not r.get("wall_contacts", 0) for r in rows)
    return (-collisions, successes)


def require_valid_evaluation(rows):
    """Never compare neural policies using broken simulator/transport runs."""
    if STOP_REQUESTED.is_set() or not rows:
        raise RuntimeError("Evaluation interrupted or empty; no checkpoint promotion")
    invalid = [r["maze"] for r in rows if not r.get("contact_monitor_observed")
               or r.get("reason", "").startswith(
                   ("error:", "stale_", "worker_", "interrupted", "no_motion_"))]
    if invalid:
        raise RuntimeError(f"Invalid evaluation workers {invalid}; repair simulation before training/promotion")


def schedule_jobs(mazes, workers):
    """Balance long sigma episodes without changing each maze's seed index."""
    names = [maze_name(maze) for maze in mazes]
    costs = [100 if name.startswith("sigma") else 30 if name.startswith("delta") else 60 for name in names]
    jobs, loads = [[] for _ in range(workers)], [0] * workers
    for index in sorted(range(len(mazes)), key=lambda i: -costs[i]):
        slot = min(range(workers), key=lambda s: loads[s])
        jobs[slot].append(index)
        loads[slot] += costs[index]
    return jobs


def collect_batch(model_path, mazes, directory, workers, seed, seconds,
                  domain_base, collect=False, expert_prob=0., yaw_jitter=0.35, navigation="disk",
                  tracking_model="hybrid_runs/release_latest/tracking_policy.pt"):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    jobs = schedule_jobs(mazes, workers)

    def worker(slot):
        results = []
        # Domain belongs to a worker for the whole batch, not to task index.
        for i in jobs[slot]:
            if STOP_REQUESTED.is_set():
                break
            maze = mazes[i]
            name = maze_name(maze)
            folder = directory / f"{i:02d}_{name}"
            folder.mkdir(parents=True, exist_ok=True)
            cmd = [sys.executable, "-m", "rl_training.eval_hybrid_gazebo",
                   *evaluator_maze_args(maze), "--model", str(model_path), "--output", str(folder),
                   "--domain-id", str(domain_base + slot), "--seconds", str(seconds),
                   "--seed", str(seed + i), "--yaw-jitter", str(yaw_jitter), "--navigation", navigation,
                   "--tracking-model", str(tracking_model)]
            if collect:
                cmd += ["--collect-data", "--expert-prob", str(expert_prob)]
            timed_out = False
            with (folder / "worker.log").open("w") as log:
                proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT)
                deadline = time.monotonic() + seconds + 100
                while proc.poll() is None:
                    if STOP_REQUESTED.is_set() or time.monotonic() >= deadline:
                        timed_out = not STOP_REQUESTED.is_set()
                        proc.send_signal(signal.SIGINT)
                        proc.wait(timeout=30)
                        break
                    try:
                        proc.wait(timeout=1)
                    except subprocess.TimeoutExpired:
                        pass
            path = folder / "result.json"
            if path.exists():
                result = json.loads(path.read_text())
            else:
                result = {"maze": name, "success": False, "reason": "worker_no_result"}
            if timed_out:
                result.update(success=False, reason="worker_timeout")
            result["artifact_dir"] = str(folder)
            results.append(result)
            print(f"worker={slot + 1} maze={name} success={result['success']} "
                  f"contacts={result.get('wall_contacts', '?')} reason={result['reason']}", flush=True)
        return results

    pool = ThreadPoolExecutor(max_workers=workers)
    try:
        futures = [pool.submit(worker, slot) for slot in range(workers)]
        rows = [row for future in futures for row in future.result()]
    except BaseException:
        STOP_REQUESTED.set()
        raise
    finally:
        pool.shutdown(wait=True)
    rows.sort(key=lambda r: r["maze"])
    write_json(directory / "summary.json", {"episodes": len(rows), "score": score(rows), "results": rows})
    return rows


def load_dataset(rows, input_dim=38):
    observations, targets = [], []
    for row in rows:
        if row.get("observation_dim", 38) != input_dim:
            continue
        path = Path(row["artifact_dir"]) / "demonstrations.npz"
        if not path.exists():
            continue
        # Transport failures can produce observations with uncertain timing.
        if row.get("reason", "").startswith(("error:", "stale_", "worker_", "interrupted", "no_motion_")):
            continue
        with np.load(path, allow_pickle=False) as data:
            x, y = data["observations"], data["labels"]
        if x.shape != (len(x), input_dim) or y.shape != (len(x), 2):
            raise ValueError(f"invalid demonstrations: {path}")
        if len(x) and np.isfinite(x).all() and np.isfinite(y).all():
            observations.append(x)
            targets.append(y)
    if not observations:
        raise RuntimeError("No usable Gazebo data; refusing to train on empty or stale rollouts")
    return np.concatenate(observations), np.concatenate(targets)


def fit(model_path, x, y, output, seed, epochs=80):
    torch.manual_seed(seed)
    if x.shape[1] == 42:
        from rl_training.oriented_navigation import TrackingPolicy, tracking_teacher
        from rl_training.train_tracking import training_samples
        model = TrackingPolicy()
        replay = training_samples(seed, max(4000, len(x)))
        replay_labels = tracking_teacher(replay)
    else:
        model = LocalPolicy()
        replay = samples(np.random.default_rng(seed), max(2000, len(x)))
        replay_labels = teacher(replay)
    model.load_state_dict(torch.load(model_path, map_location="cpu", weights_only=True))
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0003)
    # Keep old local skills while learning the new corridor observations.
    tx = torch.from_numpy(np.concatenate([x, replay]))
    ty = torch.from_numpy(np.concatenate([y, replay_labels]))
    with torch.no_grad():
        before = ((model(torch.from_numpy(x)) - torch.from_numpy(y)) ** 2).mean().item()
    model.train()
    for _ in range(epochs):
        for ids in torch.randperm(len(tx)).split(256):
            loss = ((model(tx[ids]) - ty[ids]) ** 2).mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    model.eval()
    with torch.no_grad():
        after = ((model(torch.from_numpy(x)) - torch.from_numpy(y)) ** 2).mean().item()
    torch.save(model.state_dict(), output)
    # Optimizer + aggregated data are saved with every round for auditing.
    torch.save(optimizer.state_dict(), output.with_suffix(".optimizer.pt"))
    return {"gazebo_samples": len(x), "synthetic_replay_samples": len(replay),
            "training_mse_before": before, "training_mse_after": after}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--rounds", type=int, default=2)
    p.add_argument("--seconds", type=float, default=150.)
    p.add_argument("--seed", type=int, default=41)
    p.add_argument("--domain-base", type=int, default=90)
    p.add_argument("--model", default="hybrid_runs/release_latest/local_policy.pt")
    p.add_argument("--tracking-model", default="hybrid_runs/release_latest/tracking_policy.pt")
    p.add_argument("--output", default=None)
    p.add_argument("--include-sigma", action="store_true",
                   help="add sigma 1,2,5 training and sigma 3 validation; sigma 4 is a duplicate-geometry final check")
    p.add_argument("--procedural-train-seeds", type=int, nargs="*", default=[],
                   help="generated maze seeds used for DAgger collection")
    p.add_argument("--procedural-validation-seeds", type=int, nargs="*", default=[],
                   help="disjoint generated maze seeds reserved for validation")
    p.add_argument("--procedural-size", type=int, default=5)
    p.add_argument("--procedural-cell", type=float, default=.75)
    args = p.parse_args()
    STOP_REQUESTED.clear()
    signal.signal(signal.SIGTERM, lambda *_: STOP_REQUESTED.set())
    signal.signal(signal.SIGINT, lambda *_: STOP_REQUESTED.set())
    if not 1 <= args.workers <= 4 or args.rounds < 1 or not 0 <= args.domain_base <= 197:
        p.error("workers must be 1..4, rounds positive, domain-base 0..197")
    if bool(args.procedural_train_seeds) != bool(args.procedural_validation_seeds):
        p.error("procedural train and validation seeds must both be provided")
    if set(args.procedural_train_seeds) & set(args.procedural_validation_seeds):
        p.error("procedural train and validation seeds must be disjoint")
    if args.procedural_size < 2 or args.procedural_cell <= .45:
        p.error("procedural-size must be >=2 and procedural-cell must be >0.45 m")
    torch.set_num_threads(2)
    out = Path(args.output or f"hybrid_runs/multi_{datetime.now():%Y%m%d_%H%M%S}").resolve()
    out.mkdir(parents=True, exist_ok=False)
    incumbent = out / "selected.pt"
    shutil.copyfile(args.model, incumbent)
    tracker = out / "selected_tracking.pt"
    if args.include_sigma:
        shutil.copyfile(args.tracking_model, tracker)
    registry = load_registry()
    train_mazes = TRAIN_MAZES + (["sigma_1", "sigma_2", "sigma_5"] if args.include_sigma else [])
    validation_mazes = VALIDATION_MAZES + (["sigma_3"] if args.include_sigma else [])
    train_mazes += [ProceduralMaze(seed, args.procedural_size, args.procedural_cell)
                    for seed in args.procedural_train_seeds]
    validation_mazes += [ProceduralMaze(seed, args.procedural_size, args.procedural_cell)
                         for seed in args.procedural_validation_seeds]
    evaluation_mazes = train_mazes + validation_mazes
    final_mazes = evaluation_mazes + (["sigma_4"] if args.include_sigma else [])
    navigation = "auto" if args.include_sigma else "disk"
    planning = {}
    for name, spec in sorted(registry.items()):
        planner, _, goal = maze_setup(spec)
        planning[name] = bool(planner.plan(spec.start_xy_local, goal))
        if args.include_sigma and not planning[name]:
            from rl_training.oriented_navigation import OrientedRoute
            OrientedRoute(spec)
            planning[name] = True
    for maze in [m for m in evaluation_mazes if not isinstance(m, str)]:
        spec = generate_holdout(maze.seed, size=maze.size, cell=maze.cell)
        planner, _, goal = maze_setup(spec)
        planning[maze.name] = bool(planner.plan(spec.start_xy_local, goal))
    if not all(planning[maze_name(n)] for n in evaluation_mazes):
        raise RuntimeError("Unexpected planning regression in train/validation maze set")
    manifest = {"scope": "multi-robot Gazebo DAgger; known maps and ground-truth pose; not SLAM",
                "args": vars(args), "train_mazes": [manifest_maze(m) for m in train_mazes],
                "validation_mazes": [manifest_maze(m) for m in validation_mazes],
                "final_mazes": [manifest_maze(m) for m in final_mazes], "navigation": navigation,
                "duplicate_geometry": [["sigma_3", "sigma_4"]],
                "planning_gate_all_13": planning, "controller_config": asdict(H),
                "initial_model_sha256": hashlib.sha256(incumbent.read_bytes()).hexdigest(),
                "initial_tracking_sha256": hashlib.sha256(tracker.read_bytes()).hexdigest() if args.include_sigma else None,
                "source_sha256": {str(path): hashlib.sha256(path.read_bytes()).hexdigest()
                                  for path in [Path(__file__), Path(__file__).with_name("eval_hybrid_gazebo.py"),
                                               Path(__file__).with_name("hybrid_navigation.py"),
                                               Path(__file__).with_name("oriented_navigation.py"),
                                               Path(__file__).with_name("audit_sigma_se2.py"),
                                               Path(__file__).with_name("train_tracking.py")]}}
    write_json(out / "manifest.json", manifest)
    print(f"RUN_DIR={out}", flush=True)
    # Validation is reserved for selection. Final test below uses fresh seeds.
    reference = collect_batch(incumbent, evaluation_mazes, out / "baseline",
                              args.workers, 7000, args.seconds, args.domain_base, navigation=navigation,
                              tracking_model=tracker)
    require_valid_evaluation(reference)
    selected_score = score(reference)
    history, xs, ys = [], [], []
    tracking_xs, tracking_ys = [], []
    for iteration in range(1, args.rounds + 1):
        round_dir = out / f"round_{iteration:02d}"
        rows = collect_batch(incumbent, train_mazes, round_dir / "collect", args.workers,
                             args.seed + iteration * 100, args.seconds, args.domain_base,
                             collect=True, expert_prob=0.4 / iteration, yaw_jitter=0.5, navigation=navigation,
                             tracking_model=tracker)
        x, y = load_dataset(rows)
        xs.append(x)
        ys.append(y)
        x, y = np.concatenate(xs), np.concatenate(ys)
        np.savez_compressed(round_dir / "dataset.npz", observations=x, labels=y)
        candidate = round_dir / "candidate.pt"
        metrics = fit(incumbent, x, y, candidate, args.seed + iteration)
        tracking_candidate = tracker
        if args.include_sigma:
            tx, ty = load_dataset(rows, input_dim=42)
            tracking_xs.append(tx)
            tracking_ys.append(ty)
            tx, ty = np.concatenate(tracking_xs), np.concatenate(tracking_ys)
            np.savez_compressed(round_dir / "tracking_dataset.npz", observations=tx, labels=ty)
            tracking_candidate = round_dir / "candidate_tracking.pt"
            metrics["tracking"] = fit(tracker, tx, ty, tracking_candidate, args.seed + iteration)
        validations = collect_batch(candidate, evaluation_mazes, round_dir / "validation",
                                    args.workers, 7000, args.seconds, args.domain_base, navigation=navigation,
                                    tracking_model=tracking_candidate)
        require_valid_evaluation(validations)
        candidate_score = score(validations)
        # Never promote a network on loss alone or if a wall collision occurred.
        promoted = candidate_score[0] == 0 and candidate_score >= selected_score
        if promoted:
            shutil.copyfile(candidate, incumbent)
            if args.include_sigma:
                shutil.copyfile(tracking_candidate, tracker)
            selected_score = candidate_score
        row = {"round": iteration, **metrics, "validation_score": candidate_score, "promoted": promoted}
        history.append(row)
        write_json(out / "training.json", history)
        print(json.dumps(row), flush=True)
    final = collect_batch(incumbent, final_mazes, out / "final_test",
                          args.workers, 19000, args.seconds, args.domain_base, yaw_jitter=0.5, navigation=navigation,
                          tracking_model=tracker)
    require_valid_evaluation(final)
    write_json(out / "result.json", {"planning_gate_all_13": planning, "history": history,
                                     "final_score": score(final), "final_episodes": len(final),
                                     "selected_model": str(incumbent),
                                     "selected_tracking_model": str(tracker) if args.include_sigma else None})
    print(f"FINISHED: {out}", flush=True)


if __name__ == "__main__":
    main()
