"""Aggregate repeated Gazebo holdouts without hiding maze-level failures."""
import argparse
import json
import math
from pathlib import Path
import re


def wilson_interval(successes, episodes, z=1.959963984540054):
    """Two-sided Wilson score interval for a binary success probability."""
    if episodes <= 0 or not 0 <= successes <= episodes:
        raise ValueError("successes/episodes must describe a non-empty binomial sample")
    p = successes / episodes
    denominator = 1 + z * z / episodes
    center = (p + z * z / (2 * episodes)) / denominator
    radius = z * math.sqrt(p * (1 - p) / episodes + z * z / (4 * episodes * episodes)) / denominator
    return center - radius, center + radius


def aggregate(summary_paths):
    episodes = []
    for path in map(Path, summary_paths):
        summary = json.loads(path.read_text())
        # The first 5x5 holdout summaries predate explicit dimension metadata.
        size = summary.get("size", 5)
        cell = summary.get("cell_m", .75)
        for row in summary["results"]:
            suffix = re.search(r"(\d+)$", row.get("maze", ""))
            maze_seed = row.get("maze_seed", int(suffix.group(1)) if suffix else row.get("seed"))
            key = f"{size}x{size}:{cell:.3f}:{maze_seed}"
            episodes.append({"geometry": key, "success": bool(row.get("success")),
                             "wall_contacts": int(row.get("wall_contacts", 0)),
                             "geometry_collision": bool(row.get("geometry_collision")),
                             "model_sha256": row.get("model_sha256"), "source": str(path)})
    successes = sum(row["success"] for row in episodes)
    low, high = wilson_interval(successes, len(episodes))
    by_geometry = {}
    for row in episodes:
        counts = by_geometry.setdefault(row["geometry"], {"successes": 0, "episodes": 0,
                                                          "wall_contacts": 0,
                                                          "geometry_collisions": 0})
        counts["successes"] += row["success"]
        counts["episodes"] += 1
        counts["wall_contacts"] += row["wall_contacts"]
        counts["geometry_collisions"] += row["geometry_collision"]
    stable_geometries = sum(v["successes"] == v["episodes"] for v in by_geometry.values())
    geometry_low, geometry_high = wilson_interval(stable_geometries, len(by_geometry))
    return {"episodes": len(episodes), "successes": successes,
            "success_rate": successes / len(episodes),
            "wilson_95": [low, high], "unique_geometries": len(by_geometry),
            "geometries_all_successful": stable_geometries,
            "geometry_all_success_wilson_95": [geometry_low, geometry_high],
            "wall_contacts": sum(row["wall_contacts"] for row in episodes),
            "geometry_collisions": sum(row["geometry_collision"] for row in episodes),
            "model_hashes": sorted({row["model_sha256"] for row in episodes}),
            "per_geometry": by_geometry}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("summaries", nargs="+")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    result = aggregate(args.summaries)
    Path(args.output).write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k != "per_geometry"}, indent=2))


if __name__ == "__main__":
    main()
