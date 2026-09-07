#!/usr/bin/env python3
"""
test_shield_lock_monitor.py — a robot the shield has taken over must be
detected even when the takeover flickers.

MEASURED MOTIVE. ``step()`` used a CONSECUTIVE counter (``_shield_streak``):
it incremented while ``safety_override`` was true and reset to zero on any
single step the shield released. EXPL_SHIELD_STREAK = 150 then meant "150
override steps in an unbroken row is a stall". In a staircase-diagonal corridor
the shield's cone slack oscillates across EXPL_SHIELD_RELEASE, so the override
drops for a step every few steps and the counter never reaches 150 -- the
sigma_3 episode at timesteps 419672 rode the full 1500-step timeout with the
shield engaged 88% of the time, coverage 0.04, R = -169.7.

ShieldLockMonitor counts override steps over a SLIDING window instead. A brief
release no longer zeroes the progress, so a robot the shield is driving 85% of
the time terminates as a stall in ~1 window instead of never.
"""

import pytest

from rl_training.reward_shaping import ShieldLockMonitor


def test_a_full_window_of_overrides_is_a_lock():
    m = ShieldLockMonitor(window=10, frac=0.8)
    for _ in range(9):
        assert not m.update(True)          # window not full yet
    assert m.update(True)                   # 10/10 -> lock


def test_a_half_shielded_window_is_not_a_lock():
    m = ShieldLockMonitor(window=10, frac=0.8)
    locked = False
    for k in range(40):
        locked = m.update(k % 2 == 0)       # 50% override, sustained
    assert not locked


def test_a_single_release_does_not_reset_the_progress():
    """The exact flicker the consecutive counter was blind to."""
    m = ShieldLockMonitor(window=10, frac=0.8)
    for _ in range(9):
        m.update(True)
    assert m.update(False)                  # 9 of the last 10 were overrides


def test_the_window_must_fill_before_it_can_lock():
    m = ShieldLockMonitor(window=150, frac=0.85)
    for _ in range(149):
        assert not m.update(True)
    assert m.update(True)


def test_reset_clears_the_window():
    m = ShieldLockMonitor(window=10, frac=0.8)
    for _ in range(10):
        m.update(True)
    m.reset()
    assert not m.update(True)               # one step, window no longer full


def test_recovery_below_the_fraction_clears_the_lock():
    m = ShieldLockMonitor(window=10, frac=0.8)
    for _ in range(10):
        m.update(True)
    # The policy takes back over: eight clean steps drop the count to 2/10.
    last = True
    for _ in range(8):
        last = m.update(False)
    assert last is False
