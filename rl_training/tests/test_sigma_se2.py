"""Dense replay catches clearance violations between planner samples."""
import math

import numpy as np
import pytest

from rl_training.audit_sigma_se2 import search, safe_pose
from rl_training.eval_hybrid import maze_setup
from rl_training.hybrid_navigation import integrate
from rl_training.maze_registry import load_registry


@pytest.mark.parametrize("maze", [f"sigma_{i}" for i in range(1, 6)])
def test_sigma_path_preserves_margin_during_dense_replay(maze):
    spec = load_registry()[maze]
    result = search(spec, seconds=10)
    assert result["path_found"]
    _, geometry, goal = maze_setup(spec)
    assert math.dist(result["path"][-1][:2], goal) < .06
    assert len(result["commands"]) == len(result["path"]) - 1
    for start, end, (v, w, dt) in zip(result["path"], result["path"][1:], result["commands"]):
        start = np.asarray(start)
        np.testing.assert_allclose(integrate(start, v, w, dt), end, atol=1e-10)
        for t in np.linspace(0., dt, math.ceil(dt / .005) + 1):
            assert safe_pose(geometry, integrate(start, v, w, t), margin=.01)
