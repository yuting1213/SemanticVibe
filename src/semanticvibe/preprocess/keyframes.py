"""Keyframe selection — pick a sparse set of frames that summarise the video."""

from __future__ import annotations

from pathlib import Path


def select_keyframes(
    video_path: Path,
    *,
    target_count: int = 12,
    output_dir: Path | None = None,
) -> list[Path]:
    """Pick ~`target_count` frames evenly spaced or at scene cuts. Save to `output_dir`."""
    raise NotImplementedError("Stage 1: implement in Week 2 (cv2 + scene-cut heuristic).")
