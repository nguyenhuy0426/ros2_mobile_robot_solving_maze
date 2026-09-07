#!/usr/bin/env python3
"""
test_exit_budget.py — Full coverage must not be a wasted episode.

v9 produced the campaign's first 25/25 coverage (ortho_3, +127.99). It then
TIMED OUT: exploration had consumed all 1500 steps, so PHASE_EXIT began with
zero budget left and the robot was truncated on the same step it earned the
right to leave. Four more episodes reached 22/25.

At ~0.027 m of travel per step the exit leg is worth a few hundred steps, so
the fix is a bounded EXTENSION granted when the phase flips — not a larger
EXPL_MAX_STEPS, which would also stretch every failed exploration episode and
slow the campaign for no gain.
"""

from rl_training import config as C
from rl_training.explore_env import episode_step_limit


def test_an_episode_that_never_completes_coverage_keeps_the_normal_limit():
    """The extension must cost nothing to the episodes that do not earn it."""
    assert episode_step_limit(None) == C.EXPL_MAX_STEPS


def test_completing_coverage_at_the_buzzer_still_buys_an_exit_leg():
    """The regression: ortho_3 hit 25/25 on the last step and was truncated."""
    assert episode_step_limit(C.EXPL_MAX_STEPS) == \
        C.EXPL_MAX_STEPS + C.EXPL_EXIT_BUDGET


def test_completing_coverage_early_does_not_shorten_the_episode():
    """max(), not a reset: an early finisher already has budget in hand.

    Returning exit_at + budget unconditionally would TRUNCATE an episode that
    mapped the maze in 200 steps, which is the opposite of the intent.
    """
    assert episode_step_limit(200) == C.EXPL_MAX_STEPS


def test_the_extension_is_bounded_and_long_enough_to_walk_out():
    """Travel is ~0.027 m/step; the exit leg is a few metres of corridor.

    Too small and the extension changes nothing; unbounded and a policy that
    never exits would hold a robot forever and starve the other three.
    """
    assert 150 <= C.EXPL_EXIT_BUDGET <= 600
    assert C.EXPL_EXIT_BUDGET * 0.027 > 4.0      # metres of reachable corridor
