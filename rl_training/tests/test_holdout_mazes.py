import hashlib
import json
import math

import numpy as np

from rl_training.eval_hybrid import maze_setup
from rl_training.eval_holdouts_gazebo import build_episode_jobs
from rl_training.holdout_mazes import generate_hex_holdout, generate_holdout


def fingerprint(spec):
    values = sorted((round(w.cx, 5), round(w.cy, 5), round(w.half_length, 5),
                     round(w.half_thickness, 5), round(w.yaw, 5)) for w in spec.walls_local)
    return hashlib.sha256(json.dumps(values).encode()).hexdigest()


def test_holdout_generation_is_reproducible_but_seed_diverse():
    assert fingerprint(generate_holdout(31001)) == fingerprint(generate_holdout(31001))
    assert len({fingerprint(generate_holdout(seed)) for seed in range(31001, 31006)}) == 5


def test_holdouts_are_not_training_registry_entries_and_have_safe_disk_paths():
    from rl_training.maze_registry import load_registry
    registry = load_registry()
    for seed in range(31001, 31006):
        spec = generate_holdout(seed)
        assert spec.name not in registry
        planner, geometry, goal = maze_setup(spec)
        path = planner.plan(spec.start_xy_local, goal)
        assert path
        assert not geometry.collides([*spec.start_xy_local, spec.start_yaw])


def test_larger_and_narrower_holdouts_have_distinct_names_and_safe_routes():
    cases = [(32001, 6, .70), (33001, 5, .60), (34001, 7, .80)]
    names = set()
    for seed, size, cell in cases:
        spec = generate_holdout(seed, size=size, cell=cell)
        names.add(spec.name)
        planner, _, goal = maze_setup(spec)
        assert planner.plan(spec.start_xy_local, goal)
    assert len(names) == len(cases)


def test_hexagonal_holdout_is_deterministic_and_plannable():
    a = generate_hex_holdout(57001, diameter=7, side=.65)
    b = generate_hex_holdout(57001, diameter=7, side=.65)
    assert fingerprint(a) == fingerprint(b)
    assert a.start_xy_local == b.start_xy_local
    assert a.exit_opening == b.exit_opening
    assert a.family == "hex_holdout"
    assert len(a.walls_local) > 0
    assert any(abs(w.yaw) % (math.pi / 2) > .1 for w in a.walls_local)
    planner, geometry, goal = maze_setup(a)
    assert planner.plan(a.start_xy_local, goal)
    assert not geometry.collides(np.array([*a.start_xy_local, a.start_yaw]))


def test_repeat_jobs_keep_geometry_seed_and_change_episode_seed():
    jobs = build_episode_jobs([7, 11], repeats=2, episode_seed_offset=500)
    assert jobs == [
        (0, 0, 7, 507), (0, 1, 7, 1_000_510),
        (1, 0, 11, 511), (1, 1, 11, 1_000_514)]
