#!/usr/bin/env python3
"""
analyze_logs.py — Parse SB3 training artifacts and summarize learning progress.

Reads a Stable-Baselines3 ``Monitor`` CSV (episode reward/length/time) and,
optionally, the TensorBoard event file written by the training scripts, then
prints a compact diagnostic table and (optionally) writes a JSON summary.

It is deliberately dependency-light: the Monitor CSV path always works; the
TensorBoard scalars are read only if the ``tensorboard`` package is importable.

Outcome classification (matches wheel_env.py reward constants):
  * success   : terminal reward jump (> WR_GOAL/2) → reached goal
  * timeout   : episode length >= WHEEL_MAX_STEPS
  * collision : everything else (terminated early, not at goal)

Example:
  ./rl_venv/bin/python -m rl_training.analysis.analyze_logs \\
      --monitor sac_wheel_logs/train_monitor.csv \\
      --tb sac_wheel_tensorboard/SAC_1 \\
      --json rl_training/analysis_outputs/last_train_metrics.json
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from rl_training import config as C


@dataclass
class EpisodeStats:
    """Aggregate outcome statistics over a set of episodes."""

    episodes: int
    total_steps: int
    wall_hours: float
    success_rate: float
    collision_rate: float
    timeout_rate: float
    reward_mean: float
    reward_min: float
    reward_max: float
    eplen_mean: float
    eplen_median: float
    short_collision_frac: float  # collisions ending within 50 steps
    last100_success: float
    last100_collision: float
    last100_timeout: float
    last100_reward: float


def _read_monitor(path: Path) -> np.ndarray:
    """Return an (N, 3) array of [reward, length, wall_time] from a Monitor CSV."""
    rows: List[List[float]] = []
    with path.open() as fh:
        fh.readline()  # JSON header line (starts with '#')
        fh.readline()  # column header: r,l,t
        for line in fh:
            line = line.strip()
            if not line:
                continue
            r, length, t = line.split(",")[:3]
            rows.append([float(r), float(length), float(t)])
    if not rows:
        raise ValueError(f"no episodes found in {path}")
    return np.asarray(rows, dtype=np.float64)


def summarize_monitor(path: Path,
                      goal_reward_threshold: Optional[float] = None,
                      max_steps: Optional[int] = None) -> EpisodeStats:
    """Compute :class:`EpisodeStats` from a Monitor CSV."""
    goal_thr = goal_reward_threshold if goal_reward_threshold is not None \
        else C.WR_GOAL / 2.0
    max_st = max_steps if max_steps is not None else C.WHEEL_MAX_STEPS

    data = _read_monitor(path)
    reward, length, wtime = data[:, 0], data[:, 1], data[:, 2]
    n = len(reward)

    success = reward > goal_thr
    timeout = (length >= max_st) & (~success)
    collision = (~success) & (~timeout)
    short_coll = collision & (length <= 50)

    k = min(100, n)
    return EpisodeStats(
        episodes=n,
        total_steps=int(length.sum()),
        wall_hours=float(wtime[-1] / 3600.0),
        success_rate=float(success.mean()),
        collision_rate=float(collision.mean()),
        timeout_rate=float(timeout.mean()),
        reward_mean=float(reward.mean()),
        reward_min=float(reward.min()),
        reward_max=float(reward.max()),
        eplen_mean=float(length.mean()),
        eplen_median=float(np.median(length)),
        short_collision_frac=float(short_coll.sum() / max(collision.sum(), 1)),
        last100_success=float(success[-k:].mean()),
        last100_collision=float(collision[-k:].mean()),
        last100_timeout=float(timeout[-k:].mean()),
        last100_reward=float(reward[-k:].mean()),
    )


def decile_trend(path: Path) -> List[Dict[str, float]]:
    """Reward / length / outcome rates in 10 equal-episode buckets over time."""
    data = _read_monitor(path)
    reward, length = data[:, 0], data[:, 1]
    success = reward > C.WR_GOAL / 2.0
    timeout = (length >= C.WHEEL_MAX_STEPS) & (~success)
    collision = (~success) & (~timeout)
    n = len(reward)
    edges = np.linspace(0, n, 11, dtype=int)
    out = []
    for i in range(10):
        s, e = edges[i], edges[i + 1]
        if e <= s:
            continue
        out.append({
            "bucket": i + 1,
            "reward_mean": float(reward[s:e].mean()),
            "eplen_mean": float(length[s:e].mean()),
            "success_rate": float(success[s:e].mean()),
            "collision_rate": float(collision[s:e].mean()),
            "timeout_rate": float(timeout[s:e].mean()),
        })
    return out


def read_tb_scalars(tb_dir: Path) -> Dict[str, Dict[str, float]]:
    """Best-effort TensorBoard scalar summary: {tag: {first, last, min, max}}."""
    try:
        from tensorboard.backend.event_processing.event_accumulator import (
            EventAccumulator,
        )
    except ImportError:
        return {}
    ea = EventAccumulator(str(tb_dir))
    ea.Reload()
    result: Dict[str, Dict[str, float]] = {}
    for tag in ea.Tags().get("scalars", []):
        vals = np.array([e.value for e in ea.Scalars(tag)], dtype=np.float64)
        if vals.size == 0:
            continue
        k = max(1, vals.size // 10)
        result[tag] = {
            "first_decile_mean": float(vals[:k].mean()),
            "last_decile_mean": float(vals[-k:].mean()),
            "min": float(vals.min()),
            "max": float(vals.max()),
        }
    return result


def _fmt_stats(s: EpisodeStats) -> str:
    return (
        f"episodes            : {s.episodes}\n"
        f"total env steps     : {s.total_steps}\n"
        f"wall-clock hours    : {s.wall_hours:.2f}\n"
        f"success rate        : {s.success_rate:.1%}\n"
        f"collision rate      : {s.collision_rate:.1%}\n"
        f"timeout rate        : {s.timeout_rate:.1%}\n"
        f"reward mean/min/max : {s.reward_mean:.1f} / "
        f"{s.reward_min:.1f} / {s.reward_max:.1f}\n"
        f"episode len mean/med: {s.eplen_mean:.1f} / {s.eplen_median:.0f}\n"
        f"short collisions(<50 steps) share of all collisions: "
        f"{s.short_collision_frac:.1%}\n"
        f"last-100 success/coll/timeout: "
        f"{s.last100_success:.1%} / {s.last100_collision:.1%} / "
        f"{s.last100_timeout:.1%}  (reward {s.last100_reward:.1f})"
    )


def main() -> None:
    ap = argparse.ArgumentParser(description="Analyze SB3 training logs")
    ap.add_argument("--monitor", type=Path,
                    default=C.SACW_LOG_DIR / "train_monitor.csv")
    ap.add_argument("--tb", type=Path, default=None,
                    help="TensorBoard run dir (e.g. sac_wheel_tensorboard/SAC_1)")
    ap.add_argument("--json", type=Path, default=None,
                    help="optional path to write a JSON metrics summary")
    args = ap.parse_args()

    if not args.monitor.exists():
        raise SystemExit(f"monitor CSV not found: {args.monitor}")

    stats = summarize_monitor(args.monitor)
    trend = decile_trend(args.monitor)
    tb = read_tb_scalars(args.tb) if args.tb else {}

    print("═══════════ Training log summary ═══════════")
    print(_fmt_stats(stats))
    print("\n── Outcome trend (10 equal-episode buckets, oldest→newest) ──")
    print("bucket  reward   eplen  succ%  coll%   to%")
    for b in trend:
        print(f"{b['bucket']:>5}  {b['reward_mean']:7.1f} "
              f"{b['eplen_mean']:6.0f}  {b['success_rate']:5.1%} "
              f"{b['collision_rate']:5.1%} {b['timeout_rate']:5.1%}")
    if tb:
        print("\n── TensorBoard scalars (first→last decile mean) ──")
        for tag, v in tb.items():
            print(f"{tag:28s} {v['first_decile_mean']:9.3f} → "
                  f"{v['last_decile_mean']:9.3f}  "
                  f"[min {v['min']:.3f}, max {v['max']:.3f}]")

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        payload = {"stats": asdict(stats), "trend": trend, "tensorboard": tb}
        args.json.write_text(json.dumps(payload, indent=2))
        print(f"\nwrote {args.json}")


if __name__ == "__main__":
    main()
