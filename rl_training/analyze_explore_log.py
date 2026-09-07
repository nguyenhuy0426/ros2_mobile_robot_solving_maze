#!/usr/bin/env python3
"""Automated analysis of one isolated multi-robot explore training log.

Reads a run's ``logs/explore_multi.csv`` (20-column v7 schema) and
prints a fixed report: overall outcome mix, a quartile trend, a per-maze
breakdown and a short verdict. The verdict is the part that drives the
"read the log, then adjust the algorithm" loop -- it names the failure mode
the numbers currently support, so the next tuning step does not have to be
re-derived by eye every time.

Usage:
    python -m rl_training.analyze_explore_log [--csv PATH] [--tail N]
"""
from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List

RUN_ROOT = Path("sac_explore_multi_runs")


def default_csv() -> Path:
    """Select the newest isolated run; never silently read a legacy log."""
    candidates = sorted(RUN_ROOT.glob("*/logs/explore_multi.csv"),
                        key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0] if candidates else RUN_ROOT / "<run>/logs/explore_multi.csv"

FLAGS = ["success", "collision", "early_exit", "timeout", "stall",
         "out_of_bounds"]


def load(path: Path) -> List[Dict[str, Any]]:
    with path.open(newline="") as fh:
        rows = list(csv.DictReader(fh))
    out: List[Dict[str, Any]] = []
    for r in rows:
        try:
            rec: Dict[str, Any] = {
                "timesteps": int(float(r["timesteps"])),
                "robot": int(float(r["robot"])),
                "maze": r["maze"],
                "ret": float(r["episode_return"]),
                "cov": float(r["coverage_frac"]),
            }
        except (KeyError, TypeError, ValueError):
            continue
        for f in FLAGS:
            rec[f] = int(float(r.get(f) or 0))
        for f, cast in (("ep_len", int), ("min_slack", float),
                        ("danger_ray", int), ("map_pct", float),
                        ("pose_x", float), ("pose_y", float),
                        ("pose_yaw", float), ("path_length", float),
                        ("action_scale", float)):
            try:
                rec[f] = cast(float(r[f]))
            except (KeyError, TypeError, ValueError):
                rec[f] = None
        out.append(rec)
    return out


def _mix(rows: List[Dict[str, Any]]) -> Dict[str, float]:
    n = max(len(rows), 1)
    return {f: 100.0 * sum(r[f] for r in rows) / n for f in FLAGS}


def _mean(rows: List[Dict[str, Any]], key: str) -> float:
    vals = [r[key] for r in rows if r.get(key) is not None]
    return sum(vals) / len(vals) if vals else float("nan")


def report(rows: List[Dict[str, Any]]) -> List[str]:
    L: List[str] = []
    if not rows:
        return ["(log is empty -- training has not closed an episode yet)"]

    n = len(rows)
    reversals = sum(b["timesteps"] < a["timesteps"]
                    for a, b in zip(rows, rows[1:]))
    if reversals:
        L.append(f"DATA INTEGRITY: {reversals} timestep decreases; possible "
                 "mixed/restarted trainers. Totals describe logged episodes, "
                 "not one policy. Quartiles below are file-order groups only.")
    steps = rows[-1]["timesteps"]
    mix = _mix(rows)
    L.append(f"episodes {n}   timesteps {steps}   "
             f"mean cov {_mean(rows, 'cov'):.3f} "
             f"({_mean(rows, 'cov') * 25:.1f}/25 zones)   "
             f"mean R {_mean(rows, 'ret'):+.2f}   "
             f"mean ep_len {_mean(rows, 'ep_len'):.0f}")
    path_values = [r["path_length"] for r in rows
                   if r.get("path_length") is not None]
    if path_values:
        L.append(f"terminal path length: mean {sum(path_values)/len(path_values):.2f} m "
                 f"max {max(path_values):.2f} m; this is measured travel, "
                 "not coverage-zone count")
    L.append("outcome mix: " + "  ".join(f"{f}={mix[f]:.1f}%" for f in FLAGS))

    # Quartile trend -- is the policy actually improving, and at what?
    L.append("")
    L.append("quartile      eps    cov    R       ep_len  coll%   succ%  stall%")
    q = max(n // 4, 1)
    for i in range(4):
        chunk = rows[i * q:(i + 1) * q] if i < 3 else rows[3 * q:]
        if not chunk:
            continue
        m = _mix(chunk)
        L.append(f"  Q{i + 1}      {len(chunk):6d} "
                 f"{_mean(chunk, 'cov'):6.3f} {_mean(chunk, 'ret'):+8.2f} "
                 f"{_mean(chunk, 'ep_len'):7.0f} "
                 f"{m['collision']:6.1f} {m['success']:6.1f} "
                 f"{m['stall']:6.1f}")

    # Per-maze: which mazes the shared policy is failing on.
    per: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for r in rows:
        per[r["maze"]].append(r)
    L.append("")
    L.append("maze         eps    cov   best_cov     R    succ  coll%")
    for maze in sorted(per):
        rs = per[maze]
        m = _mix(rs)
        L.append(f"  {maze:<10} {len(rs):5d} {_mean(rs, 'cov'):6.3f} "
                 f"{max(r['cov'] for r in rs):9.2f} "
                 f"{_mean(rs, 'ret'):+8.2f} "
                 f"{sum(r['success'] for r in rs):5d} {m['collision']:6.1f}")

    # Which lidar ray was the killer -- front (fore of the lidar) vs side/rear.
    deaths = [r for r in rows if r["collision"] and r["danger_ray"] is not None]
    if deaths:
        def sector(ray: int) -> str:
            deg = (ray * 10) % 360
            if deg <= 45 or deg >= 315:
                return "front"
            if 135 < deg < 225:
                return "rear"
            return "side"
        sec: Dict[str, int] = defaultdict(int)
        for r in deaths:
            sec[sector(r["danger_ray"])] += 1
        tot = len(deaths)
        L.append("")
        L.append("collision sector: " + "  ".join(
            f"{k}={100.0 * v / tot:.1f}%" for k, v in sorted(sec.items())))

    L.append("")
    L.extend(verdict(rows))
    return L


def verdict(rows: List[Dict[str, Any]]) -> List[str]:
    """Name the failure mode the numbers currently support."""
    if any(b["timesteps"] < a["timesteps"]
           for a, b in zip(rows, rows[1:])):
        return ["VERDICT:",
                "  * Mixed/restarted step counters: isolate runs and check "
                "trainer ownership before interpreting a learning trend."]
    n = len(rows)
    mix = _mix(rows)
    cov = _mean(rows, "cov")
    half = max(n // 2, 1)
    first, second = rows[:half], rows[half:]
    d_cov = _mean(second, "cov") - _mean(first, "cov")
    d_len = _mean(second, "ep_len") - _mean(first, "ep_len")
    d_coll = _mix(second)["collision"] - _mix(first)["collision"]

    out = ["VERDICT:"]
    if mix["success"] > 0:
        out.append(f"  * {mix['success']:.1f}% of episodes SOLVE the maze -- "
                   "the objective is being reached; keep training and watch "
                   "whether the rate is still climbing.")
    if cov < 0.12 and mix["collision"] > 70:
        out.append("  * Dominant mode is DIE-EARLY: the policy is not "
                   "surviving long enough to explore. Safety signal / action "
                   "magnitude is the lever, not the exploration reward.")
    if d_len > 60 and d_cov < 0.02:
        out.append("  * PASSIVE-SURVIVAL TRAP: episodes are getting longer "
                   "without covering more. The agent is farming per-step "
                   "reward instead of exploring -- lengthening episodes is "
                   "NOT progress here.")
    if d_cov > 0.02:
        out.append(f"  * Coverage is trending UP (+{d_cov:.3f} over the run "
                   "half-to-half) -- the current settings are working; let "
                   "it run.")
    elif d_cov < -0.02:
        out.append(f"  * Coverage is trending DOWN ({d_cov:.3f}) -- "
                   "regression; check whether entropy collapsed.")
    if d_coll > 3:
        out.append(f"  * Collision rate is rising (+{d_coll:.1f} pts) -- the "
                   "policy is getting bolder faster than it is getting safer.")
    if mix["stall"] > 20:
        out.append(f"  * {mix['stall']:.1f}% stalls -- check sensor freshness, "
                   "control timing and movement before attributing this to "
                   "the reward or policy.")
    if len(out) == 1:
        out.append("  * These aggregate metrics do not identify a dominant "
                   "cause; inspect control traces before extending training.")
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--csv", type=Path, default=None,
                    help="run CSV; default is newest isolated run")
    ap.add_argument("--tail", type=int, default=0,
                    help="analyse only the last N episodes (0 = all)")
    args = ap.parse_args()
    path = args.csv or default_csv()
    if not path.exists():
        print(f"[analyze] no isolated run log at {path}")
        return
    rows = load(path)
    if args.tail:
        rows = rows[-args.tail:]
    print("\n".join(report(rows)))


if __name__ == "__main__":
    main()
