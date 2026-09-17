#!/usr/bin/env python3
"""
test_exit_curriculum.py — PHASE_EXIT must be reachable before it is learnable.

Across ~5000 logged episodes the explore-then-exit policy entered PHASE_EXIT
once (coverage 1.00, ortho_5). The +100 exit terminal is therefore almost
never sampled, so the exit leg is untrained. ``--exit-min-zones`` lowers the
zone count that flips the phase so early training can experience the exit and
reinforce it; the default keeps the legacy 25-zone contract.
"""

from rl_training import config as C
from rl_training.explore_env import explore_complete


def test_legacy_default_requires_every_zone():
    assert C.EXPL_EXIT_MIN_ZONES == C.EXPL_N_ZONES
    assert not explore_complete(C.EXPL_N_ZONES - 1, C.EXPL_EXIT_MIN_ZONES)
    assert explore_complete(C.EXPL_N_ZONES, C.EXPL_EXIT_MIN_ZONES)


def test_a_lowered_threshold_triggers_the_exit_phase_early():
    assert explore_complete(12, 12)
    assert not explore_complete(11, 12)


def test_the_threshold_is_inclusive_at_its_boundary():
    # n_cells >= min is the trigger; an off-by-one here either makes the
    # curriculum unreachable or flips the phase one zone too soon.
    for k in range(0, C.EXPL_N_ZONES + 2):
        assert explore_complete(k, k) is True
        assert explore_complete(k, k + 1) is False


def test_no_cells_never_completes_a_positive_threshold():
    assert not explore_complete(0, 1)
