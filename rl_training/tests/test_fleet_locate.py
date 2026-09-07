#!/usr/bin/env python3
"""
test_fleet_locate.py — map a live robot pose back to the maze it is standing
in, so the fleet can be identified by number while it runs.

MEASURED MOTIVE. The world holds 13 mazes but the watchdog spawns MAZE_FLEET
(default 4) robots, so 9 maze slots are permanently empty on screen. The 4
that exist do train on all 13 mazes -- attempt_20260907_120924 logged 10-15
episodes per maze across 162 episodes -- but they get there by TELEPORTING
between the mazes of their own group each episode, not by having a model per
maze. So the answer to "which robot is in which maze right now" changes every
episode, and nothing publishes it: the trainer prints "[robot 3 | ortho_5]"
only when the episode ENDS.

``maze_at_world`` closes that gap with a pure lookup: given a pose, which
maze footprint contains it. That is what lets a status view name the occupant
of every slot without waiting for a terminal event.
"""
import pytest

import rl_training.maze_registry as MR


@pytest.fixture(scope="module")
def placed():
    """Every registry maze, moved to its slot in the combined world."""
    registry = MR.load_registry()
    centers = MR.multi_placement()
    return {name: registry[name].place_at(centers[name]) for name in centers}


def test_each_maze_start_locates_to_its_own_maze(placed):
    """The one pose every robot is guaranteed to occupy is its start cell."""
    for name, spec in placed.items():
        assert MR.maze_at_world(spec.start_xy_world) == name


def test_a_pose_between_the_slots_belongs_to_no_maze(placed):
    """Slots are spaced wider than a footprint, so the gaps must read None."""
    a = placed["delta_1"].wall_bbox_world
    b = placed["delta_2"].wall_bbox_world
    # Midpoint of the horizontal gap between two adjacent slots.
    gap_x = (a[2] + b[0]) / 2.0
    gap_y = (a[1] + a[3]) / 2.0
    assert MR.maze_at_world((gap_x, gap_y)) is None


def test_every_maze_footprint_is_disjoint(placed):
    """A pose may never be claimed by two mazes, or the label is ambiguous."""
    for name, spec in placed.items():
        x0, y0, x1, y1 = spec.wall_bbox_world
        for probe in ((x0, y0), (x1, y1), ((x0 + x1) / 2, (y0 + y1) / 2)):
            hit = MR.maze_at_world(probe)
            assert hit in (name, None), f"{probe} in {name} resolved to {hit}"


def test_lookup_restricted_to_a_subset_ignores_other_mazes(placed):
    """A caller watching one robot's group must not get its neighbour's name."""
    spec = placed["ortho_1"]
    assert MR.maze_at_world(spec.start_xy_world, names=["sigma_3"]) is None
    assert MR.maze_at_world(spec.start_xy_world,
                            names=["ortho_1", "sigma_3"]) == "ortho_1"


def test_a_robot_just_outside_the_wall_bbox_still_reports_the_maze(placed):
    """Out-of-bounds deaths happen just past the wall; the view must still
    name where it happened rather than going blank at the moment it matters."""
    spec = placed["sigma_2"]
    x0, y0, x1, y1 = spec.wall_bbox_world
    just_out = (x1 + 0.05, (y0 + y1) / 2.0)
    assert MR.maze_at_world(just_out) is None
    assert MR.maze_at_world(just_out, margin=0.10) == "sigma_2"
