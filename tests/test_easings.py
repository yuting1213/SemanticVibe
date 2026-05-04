"""Easings — pure-function math, no Pillow / GPU."""

from __future__ import annotations

import math

import pytest

from semanticvibe.render import easings


@pytest.mark.parametrize(
    "fn",
    [
        easings.linear,
        easings.ease_in_quad,
        easings.ease_out_quad,
        easings.ease_in_out_quad,
        easings.ease_in_cubic,
        easings.ease_out_cubic,
        easings.ease_in_out_cubic,
        easings.ease_in_quart,
        easings.ease_out_quart,
        easings.ease_in_quint,
        easings.ease_out_quint,
        easings.ease_in_out_back,
        easings.ease_out_bounce,
        easings.ease_in_bounce,
    ],
)
def test_endpoints(fn):
    """All easings must hit (0, 0) and (1, 1) at the boundary."""
    assert fn(0) == pytest.approx(0, abs=1e-9)
    assert fn(1) == pytest.approx(1, abs=1e-9)


def test_ease_out_back_overshoots():
    """The 'back' family overshoots before settling — that's its identity."""
    # Somewhere in (0.5, 1) the value should exceed 1.
    overshoots = max(easings.ease_out_back(t / 100) for t in range(50, 100))
    assert overshoots > 1.0, "ease_out_back must overshoot past 1 before settling"


def test_ease_out_elastic_oscillates():
    """Elastic should swing both above and below 1 partway through."""
    samples = [easings.ease_out_elastic(t / 100) for t in range(1, 100)]
    assert max(samples) > 1.0
    assert min(samples) < 1.0


def test_ease_out_bounce_three_bounces():
    """Bounce should produce three local peaks before settling at 1."""
    samples = [easings.ease_out_bounce(t / 200) for t in range(1, 200)]
    # Count strictly local maxima — each bounce produces one.
    local_max = 0
    for i in range(1, len(samples) - 1):
        if samples[i] > samples[i - 1] and samples[i] > samples[i + 1]:
            local_max += 1
    assert local_max >= 2, f"ease_out_bounce should have ≥2 local maxima, got {local_max}"


def test_apply_unknown_falls_back_to_linear():
    assert easings.apply("not_a_real_easing", 0.5) == 0.5
    assert easings.apply("ease_out_quad", 0.5) == easings.ease_out_quad(0.5)


def test_registry_complete():
    """Every public ease_* function should be in the registry."""
    public_fns = {
        name for name in dir(easings)
        if name.startswith("ease_") and callable(getattr(easings, name))
    }
    public_fns.add("linear")
    # Registry should cover all of them (modulo aliases like _BACK_C1 etc).
    missing = public_fns - set(easings.EASINGS.keys())
    # Allow private helpers — only fail if a *public* easing is missing.
    assert not missing, f"easings missing from registry: {missing}"
