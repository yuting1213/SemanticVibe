"""Animation primitives for text and decorations.

Each animation is a pure function of (element, current_time) → AnimationState,
keeping rendering itself stateless. The state describes per-frame transforms
that text_render and composite then apply.

Five animations per spec §5.5: bounce_in, typewriter, wiggle, draw_in, fade.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from semanticvibe.schemas.decision import AnimationName

# Default animation envelope durations (seconds). Tunable per element later.
ENTRY_DURATION = 0.35
EXIT_DURATION = 0.25


@dataclass(frozen=True)
class AnimationState:
    """Per-frame parameters consumed by the renderer."""

    alpha: float = 1.0  # 0..1
    scale: float = 1.0
    dx: float = 0.0  # pixel offset
    dy: float = 0.0
    rotation_deg: float = 0.0
    reveal_fraction: float = 1.0  # for typewriter / draw_in: fraction of glyphs to draw


def _ease_out_back(t: float) -> float:
    # Overshoot ease, classic bounce-in feel.
    c1 = 1.70158
    c3 = c1 + 1
    return 1 + c3 * pow(t - 1, 3) + c1 * pow(t - 1, 2)


def _local_time(now: float, start: float, end: float) -> tuple[float, float, float]:
    """Return (t_in, t_active, t_out) — only one is in (0,1] at a time.

    t_in:  progress through the entry envelope (0→1 then stays at 1)
    t_active: 1 if in the steady-state portion, else 0
    t_out: progress through the exit envelope (0→1)
    """
    duration = end - start
    if now < start or now > end:
        return 0.0, 0.0, 0.0

    entry_end = start + min(ENTRY_DURATION, duration / 2)
    exit_start = end - min(EXIT_DURATION, duration / 2)

    if now < entry_end:
        return (now - start) / (entry_end - start), 0.0, 0.0
    if now > exit_start:
        return 1.0, 0.0, (now - exit_start) / (end - exit_start)
    return 1.0, 1.0, 0.0


def evaluate(
    name: AnimationName,
    *,
    now: float,
    start: float,
    end: float,
) -> AnimationState:
    """Compute the per-frame state for `name` at time `now`."""
    t_in, t_active, t_out = _local_time(now, start, end)

    # If we're outside the visibility window altogether, fully transparent.
    if t_in == 0.0 and t_active == 0.0 and t_out == 0.0:
        return AnimationState(alpha=0.0)

    if name == "fade":
        if t_in < 1.0:
            return AnimationState(alpha=t_in)
        if t_out > 0.0:
            return AnimationState(alpha=1.0 - t_out)
        return AnimationState(alpha=1.0)

    if name == "bounce_in":
        if t_in < 1.0:
            return AnimationState(alpha=t_in, scale=_ease_out_back(t_in))
        if t_out > 0.0:
            return AnimationState(alpha=1.0 - t_out, scale=1.0)
        return AnimationState(alpha=1.0, scale=1.0)

    if name == "typewriter":
        if t_in < 1.0:
            return AnimationState(alpha=1.0, reveal_fraction=t_in)
        if t_out > 0.0:
            return AnimationState(alpha=1.0 - t_out, reveal_fraction=1.0)
        return AnimationState(alpha=1.0, reveal_fraction=1.0)

    if name == "draw_in":
        # Conceptually a stroke-by-stroke draw; approximated as combined
        # alpha sweep + reveal_fraction. text_render uses reveal_fraction to
        # build up the rendered string; decorations use alpha alone.
        if t_in < 1.0:
            eased = 1 - pow(1 - t_in, 2)  # ease-out quad
            return AnimationState(alpha=eased, reveal_fraction=eased)
        if t_out > 0.0:
            return AnimationState(alpha=1.0 - t_out, reveal_fraction=1.0)
        return AnimationState(alpha=1.0, reveal_fraction=1.0)

    if name == "wiggle":
        # Steady-state sinusoidal jitter, plus standard fade in/out envelope.
        amplitude_px = 4.0
        freq_hz = 2.5
        phase = (now - start) * freq_hz * 2 * math.pi
        dx = math.cos(phase) * amplitude_px
        dy = math.sin(phase * 1.3) * amplitude_px
        if t_in < 1.0:
            return AnimationState(alpha=t_in, dx=dx, dy=dy)
        if t_out > 0.0:
            return AnimationState(alpha=1.0 - t_out, dx=dx, dy=dy)
        return AnimationState(alpha=1.0, dx=dx, dy=dy)

    raise ValueError(f"unknown animation: {name}")
