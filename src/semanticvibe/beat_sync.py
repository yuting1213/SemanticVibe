"""Beat detection + snap-to-beat helpers.

v9 module. Lets the renderer align lyric overlays + decoration entries
to the underlying music's actual rhythm, so the visual punch lines
land on the music's punch lines.

Public surface:

    detect_beats(media_path)       → BeatInfo dict
    snap_to_beat(t, beats, max=)   → t snapped to nearest beat (or t)
    is_downbeat(t, downbeats)      → bool
    is_high_energy(t, segments)    → bool

`media_path` may be either an audio file or a video — for video we
reuse the existing `preprocess.librosa_beats.extract_wav` (loudnorm-on
by default — quiet phone recordings need amplification before librosa
can find the beats).
"""

from __future__ import annotations

import logging
from functools import lru_cache
from pathlib import Path
from typing import TypedDict

import librosa
import numpy as np

log = logging.getLogger(__name__)


class BeatInfo(TypedDict):
    tempo: float
    beat_times: list[float]
    downbeat_times: list[float]
    energy_envelope: list[tuple[float, float]]
    high_energy_segments: list[tuple[float, float]]


def _to_wav(media_path: Path) -> Path:
    """Resolve an arbitrary media file (mp4 / mp3 / wav / ...) to a wav we
    can hand to librosa. Reuses the loudnorm-cached wav from preprocess
    so we don't decode the same file twice."""
    from semanticvibe.preprocess.librosa_beats import extract_wav

    return extract_wav(Path(media_path), sr=22050, loudnorm=True)


@lru_cache(maxsize=8)
def detect_beats(media_path: str) -> BeatInfo:
    """Return BPM + beat grid + downbeats + per-second RMS energy +
    high-energy segments (chorus candidates).

    Cached on the resolved media-path string — callers can call this
    once per (video_path, audio_path) without performance worry.

    Notes:
    - Downbeats default to every 4th beat (4/4 assumption). Real
      downbeat tracking would need madmom or a CNN; for v9 the every-4
      heuristic is good enough — we only use them to bias animation
      strength, not to drive bar-accurate cuts.
    - "high_energy_segments" is a coarse RMS-threshold pass:
      mean × 1.2 with a 2-second min duration. Not a real chorus
      detector, but reliably catches the loud parts of a pop song.
    """
    wav = _to_wav(Path(media_path))
    y, sr = librosa.load(str(wav), sr=22050, mono=True)

    tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr)
    beat_times = librosa.frames_to_time(beat_frames, sr=sr).tolist()
    # Every 4th beat as downbeat — accurate enough for animation cues;
    # a real meter detector would need bar-tracking.
    downbeat_times = beat_times[::4]

    # RMS energy at ~1 Hz resolution (hop = 1 second of samples).
    rms = librosa.feature.rms(y=y, hop_length=sr)[0]
    energy_envelope = [(float(i), float(r)) for i, r in enumerate(rms)]

    # High-energy = RMS above mean × 1.2, sustained ≥ 2 seconds.
    threshold = float(np.mean(rms) * 1.2)
    high_segments: list[tuple[float, float]] = []
    in_seg, start = False, 0
    for i, r in enumerate(rms):
        if r > threshold and not in_seg:
            start, in_seg = i, True
        elif r <= threshold and in_seg:
            if i - start >= 2:
                high_segments.append((float(start), float(i)))
            in_seg = False
    if in_seg:
        # Tail segment that runs to end-of-track.
        if len(rms) - start >= 2:
            high_segments.append((float(start), float(len(rms))))

    log.info(
        "[beat_sync] %s: tempo=%.1f BPM, %d beats, %d downbeats, "
        "%d high-energy segments",
        Path(media_path).name,
        float(tempo),
        len(beat_times),
        len(downbeat_times),
        len(high_segments),
    )
    return BeatInfo(
        tempo=float(tempo),
        beat_times=beat_times,
        downbeat_times=downbeat_times,
        energy_envelope=energy_envelope,
        high_energy_segments=high_segments,
    )


def snap_to_beat(
    t: float,
    beat_times: list[float],
    *,
    max_offset: float = 0.15,
) -> float:
    """Pull `t` to the nearest beat-time when within `max_offset` seconds;
    otherwise leave `t` alone. Defaults to ±150 ms which feels tight
    without yanking long phrases off-rhythm.
    """
    if not beat_times:
        return t
    nearest = min(beat_times, key=lambda b: abs(b - t))
    if abs(nearest - t) <= max_offset:
        return float(nearest)
    return t


def is_downbeat(
    t: float,
    downbeat_times: list[float],
    *,
    tolerance: float = 0.1,
) -> bool:
    """True when `t` is within `tolerance` of any downbeat."""
    return any(abs(t - d) <= tolerance for d in downbeat_times)


def is_high_energy(
    t: float,
    segments: list[tuple[float, float]],
) -> bool:
    """True when `t` falls inside any high-energy / chorus segment."""
    return any(start <= t <= end for start, end in segments)


def average_beat_period(beat_times: list[float]) -> float | None:
    """Mean inter-beat interval in seconds. None if too few beats."""
    if len(beat_times) < 2:
        return None
    return (beat_times[-1] - beat_times[0]) / (len(beat_times) - 1)
