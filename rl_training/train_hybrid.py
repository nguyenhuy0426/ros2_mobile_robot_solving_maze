"""Cheap imitation baseline; no maze geometry or test-maze trajectories used.

The teacher is a local proportional tracker slowed by frontal LiDAR returns.
This is a control sanity check, not a claim of a novel RL algorithm.
"""
import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch

from rl_training import config as C
from rl_training.hybrid_navigation import H, LocalPolicy


def teacher(obs):
    angle = obs[:, -1] * np.pi
    distance = obs[:, -2]
    frontal = obs[:, [0, 1, 35]].min(axis=1) * C.LIDAR_MAX
    speed = np.minimum(distance * 1.5 / H.max_v, 1.) * np.maximum(np.cos(angle), 0.) ** 4
    speed *= np.clip((frontal - 0.15) / 0.35, 0., 1.)
    turn = np.clip(2. * angle / H.max_w, -1., 1.)
    return np.column_stack((speed, turn)).astype(np.float32)


def samples(rng, n):
    angles = np.arange(C.N_RAYS) * 2 * np.pi / C.N_RAYS
    # Random rectangular local rooms/corridors, relative sensor position.
    sides = rng.uniform(0.18, 3., (n, 4))
    rays = np.stack((sides[:, 0, None] / np.maximum(np.cos(angles), 1e-6),
                     sides[:, 1, None] / np.maximum(-np.cos(angles), 1e-6),
                     sides[:, 2, None] / np.maximum(np.sin(angles), 1e-6),
                     sides[:, 3, None] / np.maximum(-np.sin(angles), 1e-6)))
    scan = np.clip(rays.min(axis=0), C.LIDAR_MIN, C.LIDAR_MAX)
    bearing = rng.uniform(-1, 1, n)
    bearing[:n // 2] *= 0.15  # Fine straight-line steering needs more samples.
    return np.column_stack((scan / C.LIDAR_MAX, rng.uniform(0.03, 0.5, n), bearing)).astype(np.float32)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="hybrid_runs/training_single")
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--epochs", type=int, default=60)
    args = parser.parse_args()
    torch.set_num_threads(2)
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    x = samples(rng, 24000)
    vx = samples(np.random.default_rng(args.seed + 1000), 4000)
    tx, ty = torch.from_numpy(x), torch.from_numpy(teacher(x))
    valid_x, valid_y = torch.from_numpy(vx), torch.from_numpy(teacher(vx))
    model = LocalPolicy()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.002)
    started = time.monotonic()
    for epoch in range(args.epochs):
        for ids in torch.randperm(len(tx)).split(512):
            loss = ((model(tx[ids]) - ty[ids]) ** 2).mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        if (epoch + 1) % 10 == 0:
            with torch.no_grad():
                mse = ((model(valid_x) - valid_y) ** 2).mean().item()
            print(f"epoch={epoch + 1} validation_mse={mse:.6f}", flush=True)
    model.eval()
    with torch.no_grad():
        mae = (model(valid_x) - valid_y).abs().mean(0).tolist()
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), out / "local_policy.pt")
    meta = {"algorithm": "local behavior cloning", "architecture": [38, 64, 64, 2],
            "seed": args.seed, "epochs": args.epochs, "train_samples": len(x),
            "validation_samples": len(vx), "validation_mae_normalized": mae,
            "seconds": time.monotonic() - started,
            "training_maps": "none; synthetic local rectangles only",
            "scope": "local control baseline, not end-to-end maze RL"}
    (out / "training.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
