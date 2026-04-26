"""Per-frame occupancy maps — pixels we should NOT cover with overlays.

Built from MediaPipe subject boxes (Stage 1) plus optional saliency.
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
    width: int, height: int, subjects: list[SubjectBox], padding_px: int = 16
) -> OccupancyMap:
    """Union of padded subject bounding boxes across the relevant time window."""
    raise NotImplementedError("Week 4: union subject boxes with padding.")
