"""Stage 1 orchestrator: video → FeatureSummary.

Order of operations is chosen to share GPU residency:
1. Whisper (CUDA, ~3 GB) — frees on completion.
2. librosa (CPU only).
3. Keyframe selection (CPU).
4. BLIP-2 captions (CUDA, ~2 GB).
5. MediaPipe (CPU; runs anytime).

The caller may supply a pre-computed list of `SubjectBox` objects via
`extract_subjects=False` if Stage 4 has already done its own pose pass.
"""

from __future__ import annotations

import logging
from pathlib import Path

from semanticvibe.preprocess import (
    blip2_caption,
    keyframes,
    librosa_beats,
    mediapipe_pose,
    whisper_asr,
)
from semanticvibe.schemas.feature_summary import FeatureSummary

log = logging.getLogger(__name__)


def _video_duration(video_path: Path) -> float:
    import cv2

    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return n / fps if fps > 0 else 0.0


def extract_features(
    video_path: Path,
    *,
    style_preset: str,
    keyframe_count: int = 10,
    asr_language: str | None = "zh",
    device: str = "cuda",
) -> FeatureSummary:
    """Run all Stage 1 sub-steps and assemble a FeatureSummary."""

    duration = _video_duration(video_path)
    log.info("Stage 1 starting on %s (%.1fs)", video_path.name, duration)

    # Order matters on Windows: librosa pulls numba JIT, Whisper pulls cuDNN.
    # Loading them in the opposite order has been observed to heap-corrupt the
    # process. CPU-only stages first, then GPU stages, then GPU stages clean
    # up before the next one loads.
    log.info("  librosa beats + chorus…")
    beats = librosa_beats.detect_beats(video_path)
    chorus = librosa_beats.detect_chorus_segments(video_path)
    log.info("  → %d beats, %d chorus segments", len(beats), len(chorus))

    log.info("  Keyframe selection (target %d)…", keyframe_count)
    kfs = keyframes.select_keyframes(video_path, target_count=keyframe_count)
    log.info("  → %d keyframes", len(kfs))

    log.info("  Whisper ASR…")
    lyrics = whisper_asr.transcribe(video_path, language=asr_language, device=device)
    log.info("  → %d lyric segments", len(lyrics))

    log.info("  BLIP captioning…")
    caps = blip2_caption.caption_keyframes(kfs, device=device)
    description = blip2_caption.condense_captions(caps)
    log.info("  → %d captions, %d chars in description", len(caps), len(description))

    return FeatureSummary(
        lyrics=lyrics,
        video_description=description,
        beat_times=beats,
        chorus_segments=chorus,
        video_duration=duration,
        style_preset=style_preset,
    )


def detect_subjects(video_path: Path) -> list[mediapipe_pose.SubjectBox]:
    """Stage 4 needs subject boxes too; expose at the package boundary."""
    return mediapipe_pose.detect_subjects(video_path)
