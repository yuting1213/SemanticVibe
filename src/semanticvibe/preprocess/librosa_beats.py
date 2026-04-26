"""Beat tracking + chorus segmentation via librosa."""

from __future__ import annotations

from pathlib import Path


def detect_beats(video_path: Path) -> list[float]:
    """Beat onsets in seconds from start. Empty list if no beats detected."""
    raise NotImplementedError("Stage 1: implement in Week 2 (librosa.beat.beat_track).")


def detect_chorus_segments(video_path: Path) -> list[tuple[float, float]]:
    """List of (start, end) seconds for sections classified as chorus.

    Heuristic: librosa structural segmentation + pick the most-repeated motif.
    """
    raise NotImplementedError(
        "Stage 1: implement in Week 2 (librosa.segment + repetition heuristic)."
    )
