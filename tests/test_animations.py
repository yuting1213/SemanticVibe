"""Animation envelope tests — pure-function math, no Pillow dependency."""

from __future__ import annotations

import pytest

from semanticvibe.render.animations import evaluate


@pytest.mark.parametrize("name", ["fade", "bounce_in", "typewriter", "draw_in", "wiggle"])
def test_outside_window_is_invisible(name):
    state = evaluate(name, now=10.0, start=1.0, end=2.0)
    assert state.alpha == 0.0


@pytest.mark.parametrize("name", ["fade", "bounce_in", "typewriter", "draw_in", "wiggle"])
def test_steady_state_is_fully_visible(name):
    # Mid-window, well past the entry envelope and before the exit envelope.
    state = evaluate(name, now=2.0, start=0.0, end=10.0)
    assert state.alpha == pytest.approx(1.0)


def test_typewriter_reveal_advances_with_time():
    early = evaluate("typewriter", now=0.05, start=0.0, end=10.0)
    later = evaluate("typewriter", now=0.30, start=0.0, end=10.0)
    assert early.reveal_fraction < later.reveal_fraction


def test_bounce_in_overshoots_then_settles():
    # During the entry envelope the ease-out-back overshoot can briefly exceed 1.0.
    mid_entry = evaluate("bounce_in", now=0.25, start=0.0, end=5.0)
    settled = evaluate("bounce_in", now=2.0, start=0.0, end=5.0)
    assert mid_entry.scale > 0.5
    assert settled.scale == pytest.approx(1.0)


def test_wiggle_offsets_oscillate():
    # Two times far apart should give meaningfully different offsets.
    a = evaluate("wiggle", now=1.0, start=0.0, end=10.0)
    b = evaluate("wiggle", now=1.4, start=0.0, end=10.0)
    assert (a.dx, a.dy) != (b.dx, b.dy)


def test_unknown_animation_raises():
    with pytest.raises(ValueError):
        evaluate("nope", now=1.0, start=0.0, end=2.0)  # type: ignore[arg-type]
