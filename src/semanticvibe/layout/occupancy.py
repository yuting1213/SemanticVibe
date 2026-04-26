"""Per-frame occupancy maps — pixels we should NOT cover with overlays.

Built from MediaPipe subject boxes; the union (with padding) of all subject
boxes whose `frame_time` falls inside the relevant time window is rasterised
into a binary mask.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from semanticvibe.preprocess.mediapipe_pose import SubjectBox


@dataclass
class OccupancyMap:
    """Single binary mask: 1 = occupied (avoid), 0 = free."""

    width: int
    height: int
    mask: np.ndarray  # uint8, shape (H, W)


def build_occupancy(
    width: int,
    height: int,
    subjects: list[SubjectBox],
    *,
    padding_px: int = 16,
) -> OccupancyMap:
    """Union of padded subject bounding boxes, rasterised into a binary mask."""
    mask = np.zeros((height, width), dtype=np.uint8)
    for box in subjects:
        x1 = max(0, box.x - padding_px)
        y1 = max(0, box.y - padding_px)
        x2 = min(width, box.x + box.w + padding_px)
        y2 = min(height, box.y + box.h + padding_px)
        if x2 > x1 and y2 > y1:
            mask[y1:y2, x1:x2] = 1
    return OccupancyMap(width=width, height=height, mask=mask)


def subjects_in_window(
    subjects: list[SubjectBox], start: float, end: float
) -> list[SubjectBox]:
    """Filter `subjects` to those whose frame_time intersects [start, end]."""
    return [s for s in subjects if start - 0.5 <= s.frame_time <= end + 0.5]
