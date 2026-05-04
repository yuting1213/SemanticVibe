"""Easing functions for animation interpolation.

Each function takes `t ∈ [0, 1]` and returns a multiplier in roughly
the same range (some, like ease_out_back, intentionally overshoot).

Reference behaviour:
- ease_in_*  : starts slow, accelerates toward 1
- ease_out_* : starts fast, decelerates into 1
- ease_in_out_*: slow → fast → slow

Used by animations.py + idle_animations.py to keep curve maths in one
place. Pure functions, no side effects, safe to call millions of times
per render.
"""

from __future__ import annotations

import math


def linear(t: float) -> float:
    return t


# ---------------------------------------------------------------------------
# Polynomial easings
# ---------------------------------------------------------------------------


def ease_in_quad(t: float) -> float:
    return t * t


def ease_out_quad(t: float) -> float:
    return 1 - (1 - t) ** 2


def ease_in_out_quad(t: float) -> float:
    if t < 0.5:
        return 2 * t * t
    return 1 - (-2 * t + 2) ** 2 / 2


def ease_in_cubic(t: float) -> float:
    return t ** 3


def ease_out_cubic(t: float) -> float:
    return 1 - (1 - t) ** 3


def ease_in_out_cubic(t: float) -> float:
    if t < 0.5:
        return 4 * t * t * t
    return 1 - (-2 * t + 2) ** 3 / 2


def ease_in_quart(t: float) -> float:
    return t ** 4


def ease_out_quart(t: float) -> float:
    return 1 - (1 - t) ** 4


def ease_in_quint(t: float) -> float:
    return t ** 5


def ease_out_quint(t: float) -> float:
    return 1 - (1 - t) ** 5


# ---------------------------------------------------------------------------
# Specialty easings — overshoot / spring / bounce
# ---------------------------------------------------------------------------


_BACK_C1 = 1.70158
_BACK_C3 = _BACK_C1 + 1


def ease_out_back(t: float) -> float:
    """Overshoots past 1 then settles. Classic 'pop' feel."""
    return 1 + _BACK_C3 * (t - 1) ** 3 + _BACK_C1 * (t - 1) ** 2


def ease_in_out_back(t: float) -> float:
    c2 = _BACK_C1 * 1.525
    if t < 0.5:
        return ((2 * t) ** 2 * ((c2 + 1) * 2 * t - c2)) / 2
    return ((2 * t - 2) ** 2 * ((c2 + 1) * (t * 2 - 2) + c2) + 2) / 2


def ease_out_elastic(t: float) -> float:
    """Spring oscillation, decaying. Several overshoots before settling."""
    if t == 0 or t == 1:
        return t
    c4 = (2 * math.pi) / 3
    return 2 ** (-10 * t) * math.sin((t * 10 - 0.75) * c4) + 1


def ease_out_bounce(t: float) -> float:
    """Three decaying bounces — useful for drop_in animations."""
    n1 = 7.5625
    d1 = 2.75
    if t < 1 / d1:
        return n1 * t * t
    if t < 2 / d1:
        t -= 1.5 / d1
        return n1 * t * t + 0.75
    if t < 2.5 / d1:
        t -= 2.25 / d1
        return n1 * t * t + 0.9375
    t -= 2.625 / d1
    return n1 * t * t + 0.984375


def ease_in_bounce(t: float) -> float:
    return 1 - ease_out_bounce(1 - t)


# ---------------------------------------------------------------------------
# Registry — string → callable. Animations refer to easings by name so
# they can be swapped via JSON config.
# ---------------------------------------------------------------------------


EASINGS = {
    "linear": linear,
    "ease_in_quad": ease_in_quad,
    "ease_out_quad": ease_out_quad,
    "ease_in_out_quad": ease_in_out_quad,
    "ease_in_cubic": ease_in_cubic,
    "ease_out_cubic": ease_out_cubic,
    "ease_in_out_cubic": ease_in_out_cubic,
    "ease_in_quart": ease_in_quart,
    "ease_out_quart": ease_out_quart,
    "ease_in_quint": ease_in_quint,
    "ease_out_quint": ease_out_quint,
    "ease_out_back": ease_out_back,
    "ease_in_out_back": ease_in_out_back,
    "ease_out_elastic": ease_out_elastic,
    "ease_out_bounce": ease_out_bounce,
    "ease_in_bounce": ease_in_bounce,
}


def apply(easing_name: str, t: float) -> float:
    """Look up easing by name. Falls back to linear if name is unknown."""
    return EASINGS.get(easing_name, linear)(t)
