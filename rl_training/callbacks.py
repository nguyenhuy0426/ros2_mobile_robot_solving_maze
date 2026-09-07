#!/usr/bin/env python3
"""
callbacks.py — Shared SB3 callback for wheel-env training (SAC v2, TD3).

Logs episode-outcome rates and shaping diagnostics to TensorBoard so any
algorithm reusing GazeboWheelEnv gets identical, comparable curves without
duplicating logging logic. (The frozen v1 train_sac_wheels.py keeps its own
inline callback; new scripts share this richer one.)
"""

from __future__ import annotations

from stable_baselines3.common.callbacks import BaseCallback


class EpisodeMetricsCallback(BaseCallback):
    """Record success/collision/timeout rates plus curriculum & coverage."""

    def __init__(self) -> None:
        super().__init__()
        self.episodes = 0
        self.successes = 0
        self.collisions = 0
        self.timeouts = 0

    def _on_step(self) -> bool:
        for done, info in zip(self.locals["dones"], self.locals["infos"]):
            if not done:
                continue
            self.episodes += 1
            self.successes += int(info.get("success", False))
            self.collisions += int(info.get("collision", False))
            self.timeouts += int(info.get("timeout", False))
            self.logger.record("episode/success_rate",
                               self.successes / self.episodes)
            self.logger.record("episode/collision_rate",
                               self.collisions / self.episodes)
            self.logger.record("episode/timeout_rate",
                               self.timeouts / self.episodes)
            self.logger.record("episode/geo_dist_final",
                               info.get("geo_dist", float("nan")))
            if "curriculum_level" in info:
                self.logger.record("episode/curriculum_level",
                                   info["curriculum_level"])
            if "coverage_cells" in info:
                self.logger.record("episode/coverage_cells",
                                   info["coverage_cells"])
        return True
