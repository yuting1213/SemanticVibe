"""Entry animation primitives.

Each entry animation maps `(now, start, end) → AnimationState` describing
the per-frame transform to apply to a tile. State fields:

- alpha:           0..1 fade level
- scale:           multiplicative scale (1.0 = identity)
- dx, dy:          pixel offset
- rotation_deg:    rotation
- reveal_fraction: portion of glyphs visible (typewriter / draw_in)

The legacy 5 (bounce_in / typewriter / wiggle / draw_in / fade) are kept
for backwards-compat with committed example JSONs. The expanded set adds
IG-Reels-style entries:

- scale_pop:    0 → 1.2 → 1.0  (ease_out_back, 0.4 s snap)
- drop_in:      drops from y=-200 with bouncy settle (0.5 s)
- slide_in_*:   directional slide (left / right / top / bottom)
- stamp:        1.5 → 1.0 + ±5° rotation jitter — like an inked stamp
- wobble_in:    rotation oscillation -15° → 0°, decaying
- spin_in:      full 360° rotation while scaling 0 → 1

Idle animations (pulse / wiggle / drift / rotate_slow / shimmer) live in
idle_animations.py and modulate ON TOP of the entry envelope's steady
state — see render.composite for the composition.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from semanticvibe.render import easings

# Default animation envelope durations (seconds).
ENTRY_DURATION = 0.45
EXIT_DURATION = 0.30


@dataclass(frozen=True)
class AnimationState:
    """Per-frame parameters consumed by the renderer."""

    alpha: float = 1.0
    scale: float = 1.0
    dx: float = 0.0
    dy: float = 0.0
    rotation_deg: float = 0.0
    reveal_fraction: float = 1.0


def _phase(
    now: float, start: float, end: float, *,
    entry_dur: float = ENTRY_DURATION, exit_dur: float = EXIT_DURATION,
) -> tuple[float, float, float]:
    """Return (t_in, t_active, t_out). Each in [0, 1]; only one > 0 at a time."""
    duration = end - start
    if now < start or now > end:
        return 0.0, 0.0, 0.0

    entry_end = start + min(entry_dur, duration / 2)
    exit_start = end - min(exit_dur, duration / 2)

    if now < entry_end:
        return (now - start) / (entry_end - start), 0.0, 0.0
    if now > exit_start:
        return 1.0, 0.0, (now - exit_start) / (end - exit_start)
    return 1.0, 1.0, 0.0


def _invisible(now: float, start: float, end: float) -> bool:
    return now < start or now > end


# ---------------------------------------------------------------------------
# Legacy-compatible animations (used by committed example JSONs)
# ---------------------------------------------------------------------------


def fade(now: float, start: float, end: float) -> AnimationState:
    if _invisible(now, start, end):
        return AnimationState(alpha=0.0)
    t_in, _, t_out = _phase(now, start, end)
    if t_in < 1.0:
        return AnimationState(alpha=easings.ease_out_quad(t_in))
    if t_out > 0:
        return AnimationState(alpha=1.0 - easings.ease_in_quad(t_out))
    return AnimationState(alpha=1.0)


def bounce_in(now: float, start: float, end: float) -> AnimationState:
    if _invisible(now, start, end):
        return AnimationState(alpha=0.0)
    t_in, _, t_out = _phase(now, start, end)
    if t_in < 1.0:
        return AnimationState(alpha=t_in, scale=easings.ease_out_back(t_in))
    if t_out > 0:
        return AnimationState(alpha=1.0 - t_out, scale=1.0)
    return AnimationState(alpha=1.0, scale=1.0)


def typewriter(now: float, start: float, end: float) -> AnimationState:
    if _invisible(now, start, end):
        return AnimationState(alpha=0.0)
    t_in, _, t_out = _phase(now, start, end, entry_dur=0.6)
    if t_in < 1.0:
        return AnimationState(alpha=1.0, reveal_fraction=t_in)
    if t_out > 0:
        return AnimationState(alpha=1.0 - t_out, reveal_fraction=1.0)
    return AnimationState(alpha=1.0, reveal_fraction=1.0)


def draw_in(now: float, start: float, end: float) -> AnimationState:
    """Left-to-right reveal via reveal_fraction + alpha sweep — text_render
    consumes reveal_fraction; sticker tiles degenerate to a fade.
    """
    if _invisible(now, start, end):
        return AnimationState(alpha=0.0)
    t_in, _, t_out = _phase(now, start, end, entry_dur=0.6)
    if t_in < 1.0:
        eased = easings.ease_out_quad(t_in)
        return AnimationState(alpha=eased, reveal_fraction=eased)
    if t_out > 0:
        return AnimationState(alpha=1.0 - t_out, reveal_fraction=1.0)
    return AnimationState(alpha=1.0, reveal_fraction=1.0)


def wiggle(now: float, start: float, end: float) -> AnimationState:
    """Steady-state sinusoidal jitter + standard fade envelope.

    NOTE: This is the *legacy entry-time wiggle*. The new idle wiggle
    (idle_animations.idle_wiggle) is composed on top of any entry
    animation and is the preferred way to add ambient motion.
    """
    if _invisible(now, start, end):
        return AnimationState(alpha=0.0)
    t_in, _, t_out = _phase(now, start, end)
    amp = 4.0
    phase = (now - start) * 2.5 * 2 * math.pi
    dx = math.cos(phase) * amp
    dy = math.sin(phase * 1.3) * amp
    if t_in < 1.0:
        return AnimationState(alpha=t_in, dx=dx, dy=dy)
    if t_out > 0:
        return AnimationState(alpha=1.0 - t_out, dx=dx, dy=dy)
    return AnimationState(alpha=1.0, dx=dx, dy=dy)


# ---------------------------------------------------------------------------
# Expanded entry animations (v4 — IG-Reels style)
# ---------------------------------------------------------------------------


def scale_pop(now: float, start: float, end: float) -> AnimationState:
    """0 → 1.2 → 1.0 with ease_out_back. Quick punchy 0.4 s entry."""
    if _invisible(now, start, end):
        return AnimationState(alpha=0.0)
    t_in, _, t_out = _phase(now, start, end, entry_dur=0.4)
    if t_in < 1.0:
        return AnimationState(
            alpha=easings.ease_out_quad(t_in),
            scale=easings.ease_out_back(t_in),
        )
    if t_out > 0:
        return AnimationState(alpha=1.0 - t_out, scale=1.0)
    return AnimationState(alpha=1.0, scale=1.0)


def drop_in(now: float, start: float, end: float) -> AnimationState:
    """Drops from y = -200 to 0 with ease_out_bounce (3 decaying bounces)."""
    if _invisible(now, start, end):
        return AnimationState(alpha=0.0)
    t_in, _, t_out = _phase(now, start, end, entry_dur=0.5)
    if t_in < 1.0:
        bounced = easings.ease_out_bounce(t_in)
        dy = -200 * (1 - bounced)
        return AnimationState(alpha=easings.ease_out_quad(t_in), dy=dy)
    if t_out > 0:
        return AnimationState(alpha=1.0 - t_out)
    return AnimationState(alpha=1.0)


def _slide_in_factory(direction: str, distance: int = 200):
    """Build a sliding entry from `direction` over `distance` px, ease_out_cubic."""

    def _impl(now: float, start: float, end: float) -> AnimationState:
        if _invisible(now, start, end):
            return AnimationState(alpha=0.0)
        t_in, _, t_out = _phase(now, start, end, entry_dur=0.4)
        if t_in < 1.0:
            eased = easings.ease_out_cubic(t_in)
            offset = distance * (1 - eased)
            dx, dy = 0.0, 0.0
            if direction == "left":
                dx = -offset
            elif direction == "right":
                dx = offset
            elif direction == "top":
                dy = -offset
            elif direction == "bottom":
                dy = offset
            return AnimationState(alpha=easings.ease_out_quad(t_in), dx=dx, dy=dy)
        if t_out > 0:
            return AnimationState(alpha=1.0 - t_out)
        return AnimationState(alpha=1.0)

    return _impl


slide_in_left = _slide_in_factory("left")
slide_in_right = _slide_in_factory("right")
slide_in_top = _slide_in_factory("top")
slide_in_bottom = _slide_in_factory("bottom")


def stamp(now: float, start: float, end: float) -> AnimationState:
    """1.5 → 1.0 with ease_out_back + ±5° decaying rotation jitter."""
    if _invisible(now, start, end):
        return AnimationState(alpha=0.0)
    t_in, _, t_out = _phase(now, start, end, entry_dur=0.3)
    if t_in < 1.0:
        # 1.5 settles to 1.0 along ease_out_back.
        scale = 1.5 - 0.5 * easings.ease_out_back(t_in)
        decay = 1 - t_in
        rot = math.sin(t_in * math.pi * 4) * 5 * decay
        return AnimationState(
            alpha=easings.ease_out_quad(t_in), scale=scale, rotation_deg=rot,
        )
    if t_out > 0:
        return AnimationState(alpha=1.0 - t_out, scale=1.0)
    return AnimationState(alpha=1.0, scale=1.0)


def wobble_in(now: float, start: float, end: float) -> AnimationState:
    """Decaying rotation oscillation, max ±15°, fades to 0° as t_in → 1."""
    if _invisible(now, start, end):
        return AnimationState(alpha=0.0)
    t_in, _, t_out = _phase(now, start, end, entry_dur=0.5)
    if t_in < 1.0:
        decay = 1 - t_in
        rot = math.sin(t_in * math.pi * 5) * 15 * decay
        return AnimationState(alpha=easings.ease_out_quad(t_in), rotation_deg=rot)
    if t_out > 0:
        return AnimationState(alpha=1.0 - t_out)
    return AnimationState(alpha=1.0)


def spin_in(now: float, start: float, end: float) -> AnimationState:
    """Full 360° rotation + scale 0 → 1, both ease_out_cubic."""
    if _invisible(now, start, end):
        return AnimationState(alpha=0.0)
    t_in, _, t_out = _phase(now, start, end, entry_dur=0.6)
    if t_in < 1.0:
        eased = easings.ease_out_cubic(t_in)
        rot = -360 * (1 - eased)
        return AnimationState(
            alpha=easings.ease_out_quad(t_in), rotation_deg=rot, scale=eased,
        )
    if t_out > 0:
        return AnimationState(alpha=1.0 - t_out)
    return AnimationState(alpha=1.0)


# ---------------------------------------------------------------------------
# Registry — Decision schema's `animation` field is a string into this map.
# ---------------------------------------------------------------------------


REGISTRY = {
    # Legacy 5 (kept for backwards-compat with committed example JSONs)
    "fade": fade,
    "bounce_in": bounce_in,
    "typewriter": typewriter,
    "draw_in": draw_in,
    "wiggle": wiggle,
    # v4 expanded set
    "scale_pop": scale_pop,
    "drop_in": drop_in,
    "slide_in_left": slide_in_left,
    "slide_in_right": slide_in_right,
    "slide_in_top": slide_in_top,
    "slide_in_bottom": slide_in_bottom,
    "stamp": stamp,
    "wobble_in": wobble_in,
    "spin_in": spin_in,
}


def evaluate(name: str, *, now: float, start: float, end: float) -> AnimationState:
    """Look up `name` in the registry; raise on unknown to surface schema bugs."""
    fn = REGISTRY.get(name)
    if fn is None:
        raise ValueError(f"unknown animation: {name!r} (known: {sorted(REGISTRY)})")
    return fn(now, start, end)
