"""BLIP-2 captioning on selected keyframes.

The captions are condensed (LLM-side or rule-based) into the
`FeatureSummary.video_description` paragraph.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class FrameCaption:
    frame_time: float
    caption: str


def caption_keyframes(keyframes: list[Path]) -> list[FrameCaption]:
    """Run BLIP-2 over selected keyframe images."""
    raise NotImplementedError("Stage 1: implement in Week 2 (transformers BLIP-2).")
