"""Idle animations — steady-state modulation layered ON TOP of entry envelopes.

Entry animations (animations.py) handle the 0 → 1 envelope: fade-in,
bounce, slide, etc. After they complete the element sits at its
"settled" state. Idle animations add subtle continuous motion to that
settled state so the frame doesn't go dead between transitions.

Each idle returns an `IdleModulation` describing additive offsets to
overlay on the entry's `AnimationState`. Composition (in composite.py):

    state = entry(now, start, end)
    if now > start + entry_dur and now < end - exit_dur:
        idle = idle_evaluate(idle_name, now - start)
        state = state + idle  # alpha *= , scale *= , dx +=, dy +=, rotation +=

Pure functions, no side effects, deterministic per (name, now).
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass


@dataclass(frozen=True)
class IdleModulation:
    """Multiplicative + additive modulations to apply on top of entry state.

    `alpha_mul` and `scale_mul` multiply the entry's values.
    `dx`, `dy`, `rotation_deg` are additive (entry value + idle offset).
    """

    alpha_mul: float = 1.0
    scale_mul: float = 1.0
    dx: float = 0.0
    dy: float = 0.0
    rotation_deg: float = 0.0


# ---------------------------------------------------------------------------
# Idle animation implementations
# ---------------------------------------------------------------------------


def none_(t: float, *, seed: int = 0) -> IdleModulation:
    return IdleModulation()


def pulse(t: float, *, seed: int = 0, period: float = 1.5, amplitude: float = 0.05) -> IdleModulation:
    """Heart-beat scale modulation: scale 1.0 ± amplitude on a sine wave."""
    phase = (t / period) * 2 * math.pi
    return IdleModulation(scale_mul=1.0 + amplitude * math.sin(phase))


def idle_wiggle(t: float, *, seed: int = 0, freq: float = 2.0, amplitude: float = 8.0) -> IdleModulation:
    """High-frequency small-amplitude position jitter — hand-drawn instability."""
    rng = random.Random(seed)
    phase_x = rng.uniform(0, 2 * math.pi)
    phase_y = rng.uniform(0, 2 * math.pi)
    dx = math.sin(t * 2 * math.pi * freq + phase_x) * amplitude
    dy = math.cos(t * 2 * math.pi * freq * 0.85 + phase_y) * amplitude
    return IdleModulation(dx=dx, dy=dy)


def drift(t: float, *, seed: int = 0, distance: float = 20.0, period: float = 3.0) -> IdleModulation:
    """Slow figure-eight drift over `period` seconds.

    Each element gets a different start phase via `seed` so a flock of
    drifting hearts doesn't move in lockstep.
    """
    rng = random.Random(seed)
    phase = rng.uniform(0, 2 * math.pi)
    angle = (t / period) * 2 * math.pi + phase
    dx = math.sin(angle) * distance
    dy = math.sin(angle * 2) * distance * 0.5  # figure-eight
    return IdleModulation(dx=dx, dy=dy)


def rotate_slow(t: float, *, seed: int = 0, speed_deg_per_sec: float = 10.0) -> IdleModulation:
    """Constant angular drift in degrees per second."""
    return IdleModulation(rotation_deg=t * speed_deg_per_sec)


def shimmer(t: float, *, seed: int = 0, period: float = 0.8, alpha_range: tuple[float, float] = (0.7, 1.0)) -> IdleModulation:
    """Twinkle-like alpha modulation between `alpha_range[0]` and `alpha_range[1]`."""
    lo, hi = alpha_range
    phase = (t / period) * 2 * math.pi
    # sine in [-1, 1] → map to [lo, hi]
    alpha = lo + (hi - lo) * (0.5 + 0.5 * math.sin(phase))
    return IdleModulation(alpha_mul=alpha)


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


REGISTRY = {
    "none": none_,
    "pulse": pulse,
    "wiggle": idle_wiggle,
    "drift": drift,
    "rotate_slow": rotate_slow,
    "shimmer": shimmer,
}


def evaluate(name: str, *, t_since_start: float, seed: int = 0) -> IdleModulation:
    """Look up an idle animation. Returns identity for unknown names — idle
    is decorative-only so unknown values shouldn't break a render.
    """
    fn = REGISTRY.get(name)
    if fn is None:
        return IdleModulation()
    return fn(t_since_start, seed=seed)
