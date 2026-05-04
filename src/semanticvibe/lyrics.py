"""Lyrics input/output schema, shared between every code path that takes
a list-of-(time, text) — Whisper output, hand-written JSON, or pulled
from the .cache by preview_lyrics.

The schema is a Pydantic model so callers get a clear ValidationError
when they feed a malformed JSON, instead of a KeyError 50 lines into
the render path. `duration` is optional; when omitted, downstream
modules fall back to "hold until the next line, capped at 5 s".

Format:
    [
        {"time": 2.5, "text": "もしもし", "duration": 1.2},
        {"time": 5.0, "text": "電波"}
    ]
"""

from __future__ import annotations

import json
from pathlib import Path

from pydantic import BaseModel, Field, RootModel, field_validator


class LyricLine(BaseModel):
    """One lyric line. `duration` is optional — leave it off and the
    renderer holds the line until the next one starts (capped at 5 s).
    """

    time: float = Field(ge=0, description="Start time in seconds from the audio's t=0.")
    text: str = Field(min_length=1, description="Display text. CJK / Latin / mixed all OK.")
    duration: float | None = Field(
        default=None,
        description="How long the line stays on screen. Defaults to "
        "min(5.0, gap-to-next-line - 0.3) when omitted.",
    )

    @field_validator("duration")
    @classmethod
    def _duration_positive(cls, v: float | None) -> float | None:
        if v is not None and v <= 0:
            raise ValueError("duration must be > 0")
        return v


class LyricsFile(RootModel[list[LyricLine]]):
    """Container so a .json file is just `LyricsFile.model_validate_json(text)`."""


def load_lyrics(path: Path | str) -> list[LyricLine]:
    """Read a lyrics .json and validate. Pydantic raises ValidationError on bad input."""
    raw = Path(path).read_text(encoding="utf-8")
    return LyricsFile.model_validate_json(raw).root


def save_lyrics(lyrics: list[LyricLine], path: Path | str) -> Path:
    """Write a lyrics list back to .json with sensible formatting."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = [L.model_dump(exclude_none=True) for L in lyrics]
    out.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return out


def to_dict_list(lyrics: list[LyricLine]) -> list[dict]:
    """Convenience for places that still expect plain dicts (e.g. tests, JSON dumps)."""
    return [L.model_dump(exclude_none=True) for L in lyrics]
