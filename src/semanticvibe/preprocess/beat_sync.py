"""Beat detection + classification for animation sync.

Wraps `librosa_beats.detect_beats` (already used by Stage 1 for the
`FeatureSummary.beat_times` field) with two extra capabilities:

1. **`snap_to_beat`** — pull an arbitrary timestamp to the nearest beat
   if it's within `max_offset` seconds. Used to align element start_times
   so overlays land *on* the beat instead of slightly before/after.
2. **`classify_beat`** — label a beat as "downbeat" (high RMS energy in
   the surrounding audio window) or "normal". Used to pick "heavy"
   entry animations (stamp / scale_pop / drop_in) on downbeats and
   "light" ones (fade / slide_in / wobble_in) elsewhere.

The audio loader is shared with `librosa_beats._load_audio` so we don't
re-decode the same track twice in one pipeline run.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np

from semanticvibe.preprocess.librosa_beats import _load_audio, detect_beats

BeatLabel = Literal["downbeat", "normal", "none"]


@dataclass(frozen=True)
class BeatInfo:
    """Pre-computed beat metadata for a single video.

    `beats` is sorted ascending. `downbeats` is a sorted subset.
    """

    beats: list[float]
    downbeats: set[float]

    @classmethod
    def from_video(cls, video_path: Path, *, downbeat_quantile: float = 0.75) -> "BeatInfo":
        """Run librosa beat tracking + label the high-energy beats as downbeats.

        `downbeat_quantile` controls the energy cutoff: 0.75 means the top
        25% loudest beats become downbeats. Lower → more downbeats; higher
        → fewer / stronger downbeats only.
        """
        beats = detect_beats(video_path)
        if not beats:
            return cls(beats=[], downbeats=set())

        y, sr = _load_audio(str(video_path))
        # RMS energy in a small window centred on each beat.
        window = int(sr * 0.1)  # 100 ms — half a sixteenth at 120 BPM
        energies: list[float] = []
        for t in beats:
            sample = int(t * sr)
            lo = max(0, sample - window)
            hi = min(len(y), sample + window)
            if hi <= lo:
                energies.append(0.0)
                continue
            seg = y[lo:hi]
            energies.append(float(np.sqrt(np.mean(seg * seg))))
        if not energies:
            return cls(beats=beats, downbeats=set())

        threshold = float(np.quantile(energies, downbeat_quantile))
        downbeats = {b for b, e in zip(beats, energies) if e >= threshold}
        return cls(beats=beats, downbeats=downbeats)


def snap_to_beat(t: float, beats: list[float], *, max_offset: float = 0.15) -> float:
    """Pull `t` to the nearest beat if within `max_offset` seconds.

    Returns the snapped timestamp. If no beat is close enough (or `beats`
    is empty), returns `t` unchanged. This is gentle enough that authored
    JSONs won't see surprising re-timing.
    """
    if not beats:
        return t
    nearest = min(beats, key=lambda b: abs(b - t))
    if abs(nearest - t) <= max_offset:
        return float(nearest)
    return t


def classify_beat(t: float, info: BeatInfo, *, tolerance: float = 0.15) -> BeatLabel:
    """Return the type of beat closest to `t` within `tolerance` seconds.

    "downbeat" / "normal" / "none" — "none" if no beat is close enough,
    suggesting `t` falls in a quiet/instrumental region between beats.
    """
    if not info.beats:
        return "none"
    nearest = min(info.beats, key=lambda b: abs(b - t))
    if abs(nearest - t) > tolerance:
        return "none"
    return "downbeat" if nearest in info.downbeats else "normal"
