"""Bin-packing helpers for Stage 4: place rectangles into the free regions of an occupancy map.

The algorithm is a coarse grid scan: divide the canvas into cells, score each
candidate top-left position by (free area, distance from frame edges, distance
from previously-placed rects), pick the best fit.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from semanticvibe.layout.occupancy import OccupancyMap


@dataclass(frozen=True)
class PlacedRect:
    x: int
    y: int
    w: int
    h: int


def _free_area(mask: np.ndarray, x: int, y: int, w: int, h: int) -> int:
    """Number of free pixels (mask==0) inside the candidate rect."""
    region = mask[y : y + h, x : x + w]
    return int((region == 0).sum())


def pack_rects(
    occupancy: OccupancyMap,
    rects_wh: list[tuple[int, int]],
    *,
    grid_step: int = 16,
    edge_margin: int = 24,
    min_free_fraction: float = 0.85,
    bias: str = "lower-band",
) -> list[PlacedRect | None]:
    """Greedy placement of rectangles into free regions of `occupancy`.

    Args:
        bias: "lower-band" prefers the bottom 30% of the frame (where text
            traditionally lives in motion graphics); "any" treats all free
            cells equally.

    Returns one PlacedRect per input (or None if nothing fits).
    """
    H, W = occupancy.mask.shape
    placed_mask = occupancy.mask.copy()
    out: list[PlacedRect | None] = []

    for w, h in rects_wh:
        if w <= 0 or h <= 0 or w > W - 2 * edge_margin or h > H - 2 * edge_margin:
            out.append(None)
            continue

        best: tuple[float, int, int] | None = None
        threshold = int(w * h * min_free_fraction)
        for y in range(edge_margin, H - h - edge_margin, grid_step):
            for x in range(edge_margin, W - w - edge_margin, grid_step):
                free = _free_area(placed_mask, x, y, w, h)
                if free < threshold:
                    continue
                score = float(free)
                if bias == "lower-band":
                    centre_y = y + h / 2
                    target_y = H * 0.78
                    score -= abs(centre_y - target_y) * 0.5
                if best is None or score > best[0]:
                    best = (score, x, y)

        if best is None:
            out.append(None)
            continue

        _score, bx, by = best
        out.append(PlacedRect(x=bx, y=by, w=w, h=h))
        # Mark this rect as occupied so subsequent rects don't collide with it.
        placed_mask[by : by + h, bx : bx + w] = 1

    return out
