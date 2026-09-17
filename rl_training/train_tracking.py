"""Train a LiDAR/reference-pose neural tracker from synthetic local states."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from rl_training.train_hybrid import samples
from rl_training.oriented_navigation import TrackingPolicy, tracking_teacher


def training_samples(seed, n):
    rng = np.random.default_rng(seed)
    scan = samples(rng, n)[:, :36]
    error = rng.uniform(-.6, .6, (n, 2))
    error[:n // 2] *= .2
    angle = rng.uniform(-np.pi, np.pi, n)
    angle[:n // 2] *= .15
    rv = rng.choice([0., .3, .6, .8], n)
    rw = rng.uniform(-.6, .6, n)
    return np.column_stack((scan, error, np.sin(angle), np.cos(angle), rv, rw)).astype(np.float32)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", default="hybrid_runs/tracking_v1")
    p.add_argument("--epochs", type=int, default=100)
    args = p.parse_args()
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(2)
    torch.manual_seed(73)
    x = torch.from_numpy(training_samples(73, 40000))
    y = torch.from_numpy(tracking_teacher(x.numpy()))
    vx = torch.from_numpy(training_samples(1073, 5000))
    vy = torch.from_numpy(tracking_teacher(vx.numpy()))
    model = TrackingPolicy()
    opt = torch.optim.Adam(model.parameters(), lr=.001)
    for epoch in range(args.epochs):
        for ids in torch.randperm(len(x)).split(512):
            loss = ((model(x[ids]) - y[ids]) ** 2).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
        if (epoch + 1) % 20 == 0:
            with torch.no_grad():
                print(epoch + 1, ((model(vx) - vy) ** 2).mean().item(), flush=True)
    torch.save(model.state_dict(), out / "tracking_policy.pt")
    torch.save(opt.state_dict(), out / "optimizer.pt")
    with torch.no_grad():
        mae = (model(vx) - vy).abs().mean(0).tolist()
    (out / "training.json").write_text(json.dumps({"seed": 73, "architecture": [42, 96, 96, 2],
        "train_samples": len(x), "validation_samples": len(vx), "epochs": args.epochs,
        "validation_mae_normalized": mae, "training_maps": "none; synthetic local states"}, indent=2))


if __name__ == "__main__":
    main()
