"""Idle animations — modulation maths, no rendering."""

from __future__ import annotations

import math

import pytest

from semanticvibe.render import idle_animations


def test_none_is_identity():
    mod = idle_animations.evaluate("none", t_since_start=1.5)
    assert mod.alpha_mul == 1.0
    assert mod.scale_mul == 1.0
    assert mod.dx == 0
    assert mod.dy == 0
    assert mod.rotation_deg == 0


def test_unknown_name_is_identity():
    """Unknown idle names degrade gracefully — render shouldn't crash."""
    mod = idle_animations.evaluate("not_a_real_idle", t_since_start=1.0)
    assert mod.alpha_mul == 1.0


def test_pulse_is_periodic():
    """Pulse scale should oscillate around 1.0 within ±amplitude."""
    samples = [
        idle_animations.evaluate("pulse", t_since_start=t / 10).scale_mul
        for t in range(0, 100)
    ]
    # Default amplitude=0.05 → range [0.95, 1.05]
    assert 0.94 <= min(samples) <= 1.0
    assert 1.0 <= max(samples) <= 1.06


def test_pulse_period_at_1_5s():
    """Default period is 1.5s — at t=0 and t=1.5s we should land at the same phase."""
    a = idle_animations.evaluate("pulse", t_since_start=0).scale_mul
    b = idle_animations.evaluate("pulse", t_since_start=1.5).scale_mul
    assert a == pytest.approx(b, abs=1e-9)


def test_idle_wiggle_seed_phase_decoupled():
    """Different seeds should produce different phase offsets, so a flock
    of wiggles doesn't move in lockstep."""
    a = idle_animations.evaluate("wiggle", t_since_start=0.5, seed=1)
    b = idle_animations.evaluate("wiggle", t_since_start=0.5, seed=2)
    assert (a.dx, a.dy) != (b.dx, b.dy)


def test_drift_traces_figure_eight():
    """Over one full period, dx should swing both signs."""
    samples = [
        idle_animations.evaluate("drift", t_since_start=t / 10, seed=0).dx
        for t in range(0, 30)
    ]
    assert max(samples) > 0
    assert min(samples) < 0


def test_rotate_slow_is_monotone():
    """Constant angular speed → strictly increasing rotation."""
    a = idle_animations.evaluate("rotate_slow", t_since_start=0).rotation_deg
    b = idle_animations.evaluate("rotate_slow", t_since_start=1).rotation_deg
    c = idle_animations.evaluate("rotate_slow", t_since_start=2).rotation_deg
    assert a < b < c


def test_shimmer_alpha_in_range():
    """Shimmer should keep alpha_mul inside its declared (0.7, 1.0) range."""
    samples = [
        idle_animations.evaluate("shimmer", t_since_start=t / 10).alpha_mul
        for t in range(0, 50)
    ]
    assert min(samples) >= 0.69  # tiny float slack
    assert max(samples) <= 1.01
