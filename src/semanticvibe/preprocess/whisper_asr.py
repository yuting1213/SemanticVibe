"""Whisper ASR via faster-whisper (4–5× the throughput of openai-whisper on Windows)."""

from __future__ import annotations

from pathlib import Path

from semanticvibe.schemas.feature_summary import LyricSegment


def transcribe(video_path: Path, *, model_size: str = "large-v3") -> list[LyricSegment]:
    """Extract lyric segments from the audio track of `video_path`.

    Implementation is wired in Week 2. Signature is stable.
    """
    raise NotImplementedError("Stage 1: implement in Week 2 (faster-whisper).")
