"""MediaPipe pose / face detection — feeds the layout occupancy map.

NOTE: Hard-pins us to Python 3.10. MediaPipe has no 3.13 wheel.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class SubjectBox:
    """Pixel bounding box of a detected subject in a single frame."""

    frame_time: float
    x: int
    y: int
    w: int
    h: int


def detect_subjects(video_path: Path, *, sample_fps: float = 4.0) -> list[SubjectBox]:
    """Detect human subjects across the video, sampled at `sample_fps`.

    Used by `semanticvibe.layout.occupancy` to build a per-frame "do not
    overlap" mask so text/decoration placement avoids faces and bodies.
    """
    raise NotImplementedError("Stage 1: implement in Week 2 (mediapipe.solutions.pose).")
