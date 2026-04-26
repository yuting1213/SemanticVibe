"""Bin-packing helpers for Stage 4: place rectangles into the free regions of an occupancy map."""

from __future__ import annotations

from dataclasses import dataclass

from semanticvibe.layout.occupancy import OccupancyMap


@dataclass(frozen=True)
class PlacedRect:
    x: int
    y: int
    w: int
    h: int


def pack_rects(
    occupancy: OccupancyMap, rects_wh: list[tuple[int, int]]
) -> list[PlacedRect | None]:
    """Greedy placement of rectangles into free regions of `occupancy`.

    Returns one PlacedRect per input (or None if nothing fits).
    """
    raise NotImplementedError("Week 4: greedy or skyline algorithm over occupancy mask.")
