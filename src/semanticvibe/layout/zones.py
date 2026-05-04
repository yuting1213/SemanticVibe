"""Find a free placement zone in an occupancy mask via morphological erosion.

Counterpart to `pose_detector.detect_person_mask` — given a mask of
"occupied" (True) and "free" (False) pixels, plus a target tile size,
return a top-left corner that won't collide with any True pixel.

Algorithm:
1. Erode the *free* region by (target_w × target_h). What remains is
   exactly the set of pixels that are valid top-left corners for a
   target_size rectangle entirely inside the free region.
2. Filter candidates by `prefer` quadrant.
3. Pick deterministically (centroid of the largest matching connected
   component) so the same input always yields the same output —
   beat-aligned re-renders shouldn't shuffle text across frames.
"""

from __future__ import annotations

from typing import Literal

import cv2
import numpy as np

PreferZone = Literal[
    "left_upper", "right_upper", "left_lower", "right_lower",
    "center", "auto",
]


def find_placement_zone(
    mask: np.ndarray,
    target_size: tuple[int, int],
    *,
    prefer: PreferZone = "left_upper",
    edge_margin: int = 16,
) -> tuple[int, int] | None:
    """Find a top-left (x, y) where `target_size` fits in the free region.

    Args:
        mask: bool ndarray of shape (H, W). True = occupied; False = free.
        target_size: (width, height) of the tile we want to place.
        prefer: which quadrant to prefer. "auto" returns the largest
            free area regardless of position.
        edge_margin: keep this many pixels away from canvas borders.

    Returns:
        (x, y) top-left coordinate, or None if no valid spot exists.
    """
    h_canvas, w_canvas = mask.shape
    tw, th = target_size

    if tw <= 0 or th <= 0:
        return None
    if tw + 2 * edge_margin > w_canvas or th + 2 * edge_margin > h_canvas:
        return None

    # Step 1: build a "valid top-left" map.
    # A pixel (x, y) is a valid top-left iff mask[y:y+th, x:x+tw] is all False.
    # Equivalent to: erode the *free* region by an (tw, th) structuring element.
    free = (~mask).astype(np.uint8)
    # cv2 erosion treats white (1) as foreground. The kernel is the rectangle.
    # After eroding, a pixel = 1 iff every pixel within (tw, th) below-right
    # was originally free. That's exactly our top-left-validity map.
    kernel = np.ones((th, tw), dtype=np.uint8)
    valid_topleft = cv2.erode(free, kernel, anchor=(0, 0), iterations=1)
    # Mask out edge_margin from the canvas borders.
    valid_topleft[:edge_margin, :] = 0
    valid_topleft[h_canvas - th - edge_margin:, :] = 0
    valid_topleft[:, :edge_margin] = 0
    valid_topleft[:, w_canvas - tw - edge_margin:] = 0

    # Step 2: filter by quadrant preference.
    quadrant_mask = _quadrant_mask(prefer, w_canvas, h_canvas)
    candidates = valid_topleft & quadrant_mask
    if candidates.sum() == 0:
        # Fallback to anywhere valid — better off-quadrant than not at all.
        candidates = valid_topleft
        if candidates.sum() == 0:
            return None

    # Step 3: pick the centroid of the largest connected component.
    n_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        candidates, connectivity=8
    )
    if n_labels <= 1:
        return None

    # stats[0] is the background; pick the largest non-background component.
    areas = stats[1:, cv2.CC_STAT_AREA]
    largest_label = int(np.argmax(areas)) + 1
    cx, cy = centroids[largest_label]

    # Round + clamp.
    x = int(round(cx))
    y = int(round(cy))
    x = max(edge_margin, min(x, w_canvas - tw - edge_margin))
    y = max(edge_margin, min(y, h_canvas - th - edge_margin))
    return x, y


def _quadrant_mask(prefer: PreferZone, w: int, h: int) -> np.ndarray:
    """Return a boolean mask covering the preferred canvas region."""
    quad = np.zeros((h, w), dtype=np.uint8)
    if prefer == "auto":
        quad[:] = 1
        return quad

    mid_x = w // 2
    mid_y = h // 2

    if prefer == "left_upper":
        quad[:mid_y, :mid_x] = 1
    elif prefer == "right_upper":
        quad[:mid_y, mid_x:] = 1
    elif prefer == "left_lower":
        quad[mid_y:, :mid_x] = 1
    elif prefer == "right_lower":
        quad[mid_y:, mid_x:] = 1
    elif prefer == "center":
        # Central 50% of width and height.
        x1, x2 = w // 4, 3 * w // 4
        y1, y2 = h // 4, 3 * h // 4
        quad[y1:y2, x1:x2] = 1
    else:
        quad[:] = 1
    return quad
