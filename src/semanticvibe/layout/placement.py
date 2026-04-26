"""Resolve `anchor="auto"` elements into pixel coordinates.

Top-level entry point for Stage 4.
"""

from __future__ import annotations

from pathlib import Path

from semanticvibe.schemas.decision import Decision


def resolve_anchors(
    decision: Decision,
    *,
    video_path: Path,
    frame_size: tuple[int, int],
) -> Decision:
    """Return a copy of `decision` where every `auto` anchor is resolved to a pixel tuple."""
    raise NotImplementedError(
        "Week 4: build per-time occupancy, run bin_packing, write back resolved anchors."
    )
