"""find_placement_zone — morphological erosion + quadrant preference."""

from __future__ import annotations

import numpy as np
import pytest

from semanticvibe.layout.zones import find_placement_zone


def _empty_mask(h: int, w: int) -> np.ndarray:
    return np.zeros((h, w), dtype=bool)


def _person_centred(h: int, w: int) -> np.ndarray:
    """A mask with the central 1/3 vertical strip occupied (typical dancer)."""
    mask = np.zeros((h, w), dtype=bool)
    mask[:, w // 3 : 2 * w // 3] = True
    return mask


def test_empty_mask_returns_in_preferred_quadrant():
    """No occupants → result lands in the preferred quadrant."""
    mask = _empty_mask(720, 1280)
    pos = find_placement_zone(mask, target_size=(200, 80), prefer="left_upper")
    assert pos is not None
    x, y = pos
    # Left-upper means x < w/2 and y < h/2.
    assert x < 1280 / 2
    assert y < 720 / 2


def test_zone_avoids_central_subject():
    """A central-strip occupant should push placement out of that strip."""
    mask = _person_centred(720, 1280)
    pos = find_placement_zone(mask, target_size=(200, 80), prefer="left_upper")
    assert pos is not None
    x, y = pos
    # Element must end before the central strip starts (1280/3 ≈ 426).
    assert x + 200 <= 1280 / 3 + 5  # tiny slack for rounding


def test_returns_none_when_target_too_big():
    mask = _empty_mask(100, 100)
    assert find_placement_zone(mask, target_size=(200, 200)) is None


def test_returns_none_when_fully_occupied():
    mask = np.ones((720, 1280), dtype=bool)
    assert find_placement_zone(mask, target_size=(50, 50)) is None


@pytest.mark.parametrize(
    "prefer, asserts",
    [
        ("left_upper",  lambda x, y: x < 640 and y < 360),
        ("right_upper", lambda x, y: x >= 640 and y < 360),
        ("left_lower",  lambda x, y: x < 640 and y >= 360),
        ("right_lower", lambda x, y: x >= 640 and y >= 360),
    ],
)
def test_prefer_quadrant_routing(prefer, asserts):
    mask = _empty_mask(720, 1280)
    pos = find_placement_zone(mask, target_size=(200, 80), prefer=prefer)
    assert pos is not None
    assert asserts(*pos), f"prefer={prefer!r} returned {pos}"


def test_falls_back_outside_quadrant_when_quadrant_full():
    """If the preferred quadrant is fully occupied but elsewhere is free,
    we should still get a valid placement (better than None)."""
    mask = np.zeros((720, 1280), dtype=bool)
    mask[:360, :640] = True  # block the entire left_upper quadrant
    pos = find_placement_zone(mask, target_size=(200, 80), prefer="left_upper")
    assert pos is not None  # fallback should land outside the blocked quadrant
    x, y = pos
    # Some pixel of the (200×80) tile is outside the (0..640, 0..360) quadrant.
    assert (x + 200 > 640) or (y + 80 > 360)
