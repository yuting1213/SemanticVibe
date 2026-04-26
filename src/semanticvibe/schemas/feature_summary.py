"""Stage 1 → Stage 2 contract.

The LLM consumes `FeatureSummary` and *only* `FeatureSummary` — the raw video
is never sent. This is the central cost-optimisation lever (spec §4).
"""

from __future__ import annotations

from pydantic import BaseModel, Field, field_validator


class LyricSegment(BaseModel):
    time: float = Field(ge=0, description="Start of the segment, seconds from video start.")
    text: str = Field(min_length=1)


class FeatureSummary(BaseModel):
    lyrics: list[LyricSegment]
    video_description: str = Field(
        min_length=1,
        description="One-paragraph natural-language summary of the visual content "
        "(BLIP-2 captions on keyframes, then LLM-condensed).",
    )
    beat_times: list[float] = Field(
        description="Beat onsets in seconds (librosa). Empty list is allowed."
    )
    chorus_segments: list[tuple[float, float]] = Field(
        description="(start, end) tuples in seconds for sections classified as chorus."
    )
    video_duration: float = Field(gt=0, description="Total video length in seconds.")
    style_preset: str = Field(
        description="One of the keys in semanticvibe.config.STYLE_PRESETS — controls "
        "palette and overall vibe."
    )

    @field_validator("beat_times")
    @classmethod
    def _beats_monotonic(cls, v: list[float]) -> list[float]:
        for prev, curr in zip(v, v[1:]):
            if curr < prev:
                raise ValueError("beat_times must be monotonically non-decreasing")
        return v

    @field_validator("chorus_segments")
    @classmethod
    def _chorus_well_formed(cls, v: list[tuple[float, float]]) -> list[tuple[float, float]]:
        for start, end in v:
            if start < 0 or end <= start:
                raise ValueError(f"invalid chorus segment ({start}, {end})")
        return v
