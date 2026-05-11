"""Dancer motion peak detection — drives entry-animation intensity.

v12 — sibling to `beat_sync.py`. Where beat_sync extracts MUSICAL energy
peaks (downbeats + chorus), motion_detector extracts VISUAL energy peaks
from the dancer's body motion. Both feed into `_pick_entry` so the right
animation pool fires at the right instant.

Public surface mirrors beat_sync:

    info = detect_motion_peaks(video_path)
    if is_motion_peak(t, info["peak_times"]):
        ...
    intensity = motion_intensity_at(t, info)  # "high"/"medium"/"low"/None

Pipeline:
  1. Walk video at `sample_fps` (default 15 — dance gestures last 200-400 ms,
     so 15 fps with 0.3 s minimum peak spacing resolves them cleanly).
  2. For each sampled frame run MediaPipe Pose (Tasks API). Reuse the
     singleton `_pose_landmarker()` from preprocess.mediapipe_pose to
     avoid loading the model twice.
  3. Keep landmarks 0-22 (head + shoulders + arms + hands) of the largest-
     bbox subject. Filter visibility > 0.3.
  4. Per-frame energy = mean Euclidean velocity (frame_t vs frame_{t-1})
     of visible landmarks. Mean-not-sum so partial visibility doesn't dampen.
  5. Smooth with 0.3 s sliding mean to kill jitter.
  6. z-score normalise so peak detection is scale-invariant across videos.
  7. `scipy.signal.find_peaks(z, prominence=0.5, distance=int(sample_fps*0.3))`
     forces ≥ 0.3 s between peaks.
  8. Bucket each peak: z>1.5 → high, 0.8-1.5 → medium, 0.3-0.8 → low.
     Below 0.3 → drop (not a real peak).

Caching:
  - In-process `lru_cache(maxsize=8)` keyed on the resolved video path string.
  - TODO: a disk cache at `.cache/motion/<sha1(path+mtime+sample_fps)>.json`
    would save 10-20 s on re-renders across processes. Skipped for v12
    pending real-user demand.
"""

from __future__ import annotations

import logging
from functools import lru_cache
from pathlib import Path
from typing import Literal, TypedDict

import cv2
import numpy as np

log = logging.getLogger(__name__)


# MediaPipe Pose landmark indices 0-22 cover head, shoulders, arms, hands —
# the parts of a dancer that carry rhythmic motion. Hips (23-24) and legs
# (25-32) are excluded because they jitter with camera bounce more than
# real choreography, especially in idol-pop selfie footage.
_UPPER_BODY_LANDMARKS = list(range(23))
_VISIBILITY_THRESHOLD = 0.3
_SMOOTH_WINDOW_SEC = 0.3
_MIN_PEAK_DISTANCE_SEC = 0.3
_PEAK_PROMINENCE = 0.5
_INTENSITY_DROP_THRESHOLD = 0.3  # z-score below this → not a real peak
_INTENSITY_LOW_BAND = 0.8
_INTENSITY_HIGH_BAND = 1.5


Intensity = Literal["high", "medium", "low"]


class MotionInfo(TypedDict):
    peak_times: list[float]
    peak_intensities: dict[float, Intensity]
    energy_envelope: list[tuple[float, float]]  # (time_sec, z_score)
    sample_fps: float


def _empty_info(sample_fps: float) -> MotionInfo:
    return MotionInfo(
        peak_times=[],
        peak_intensities={},
        energy_envelope=[],
        sample_fps=sample_fps,
    )


def _bucket(z: float) -> Intensity | None:
    if z >= _INTENSITY_HIGH_BAND:
        return "high"
    if z >= _INTENSITY_LOW_BAND:
        return "medium"
    if z >= _INTENSITY_DROP_THRESHOLD:
        return "low"
    return None


@lru_cache(maxsize=8)
def detect_motion_peaks(
    video_path: str,
    *,
    sample_fps: float = 15.0,
) -> MotionInfo:
    """Pose-driven motion peak detection. See module docstring."""
    from scipy.signal import find_peaks

    from semanticvibe.preprocess.mediapipe_pose import _pose_landmarker

    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if n_frames <= 0:
        cap.release()
        log.warning("[motion_sync] %s: zero frames; returning empty MotionInfo",
                    Path(video_path).name)
        return _empty_info(sample_fps)

    step = max(1, int(round(fps / sample_fps)))
    landmarker, mp = _pose_landmarker()

    sampled_times: list[float] = []
    # Each sample: (n_landmarks, 2) array of (x, y) for visible landmarks,
    # or None when no subject was detected. NaNs would propagate through
    # the diff, so we store visibility per-landmark separately.
    sampled_coords: list[np.ndarray | None] = []
    sampled_vis: list[np.ndarray | None] = []

    idx = 0
    while idx < n_frames:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = cap.read()
        if not ok:
            break
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = landmarker.detect(mp_image)
        sampled_times.append(idx / fps)

        if not result.pose_landmarks:
            sampled_coords.append(None)
            sampled_vis.append(None)
            idx += step
            continue

        # Largest-bbox subject pick — same heuristic as pose_detector.py.
        def _area(p):
            xs = [lm.x for lm in p if getattr(lm, "visibility", 1.0) > 0.3]
            ys = [lm.y for lm in p if getattr(lm, "visibility", 1.0) > 0.3]
            if not xs or not ys:
                return 0
            return (max(xs) - min(xs)) * (max(ys) - min(ys))

        biggest = max(result.pose_landmarks, key=_area)
        coords = np.array(
            [[biggest[i].x, biggest[i].y] for i in _UPPER_BODY_LANDMARKS],
            dtype=np.float32,
        )
        vis = np.array(
            [getattr(biggest[i], "visibility", 1.0) for i in _UPPER_BODY_LANDMARKS],
            dtype=np.float32,
        )
        sampled_coords.append(coords)
        sampled_vis.append(vis)
        idx += step

    cap.release()

    if len(sampled_coords) < 4:
        log.warning("[motion_sync] %s: too few samples (%d) to detect motion",
                    Path(video_path).name, len(sampled_coords))
        return _empty_info(sample_fps)

    # Per-sample energy = mean Euclidean velocity of visible landmarks vs
    # previous sample. The very first sample has no predecessor → 0.
    energy = np.zeros(len(sampled_coords), dtype=np.float32)
    for i in range(1, len(sampled_coords)):
        a, b = sampled_coords[i - 1], sampled_coords[i]
        va, vb = sampled_vis[i - 1], sampled_vis[i]
        if a is None or b is None or va is None or vb is None:
            continue
        # Both endpoints of the velocity must be visible in BOTH frames.
        ok_mask = (va > _VISIBILITY_THRESHOLD) & (vb > _VISIBILITY_THRESHOLD)
        if not ok_mask.any():
            continue
        delta = np.linalg.norm(b - a, axis=1)  # (n_landmarks,)
        energy[i] = float(delta[ok_mask].mean())

    # 0.3 s sliding-mean smoothing.
    window = max(1, int(round(_SMOOTH_WINDOW_SEC * sample_fps)))
    if window > 1:
        kernel = np.ones(window, dtype=np.float32) / window
        energy = np.convolve(energy, kernel, mode="same")

    # z-score normalise. Guard against constant-energy edge cases.
    mu = float(energy.mean())
    sigma = float(energy.std())
    if sigma < 1e-6:
        log.info("[motion_sync] %s: constant motion energy (std≈0); no peaks",
                 Path(video_path).name)
        return _empty_info(sample_fps)
    z = (energy - mu) / sigma

    # Peak detection with minimum 0.3 s spacing + prominence floor.
    min_dist = max(1, int(round(_MIN_PEAK_DISTANCE_SEC * sample_fps)))
    peak_idxs, _props = find_peaks(z, prominence=_PEAK_PROMINENCE, distance=min_dist)

    peak_times: list[float] = []
    peak_intensities: dict[float, Intensity] = {}
    for pi in peak_idxs:
        t = float(sampled_times[int(pi)])
        bucket = _bucket(float(z[int(pi)]))
        if bucket is None:
            continue
        peak_times.append(t)
        peak_intensities[t] = bucket

    energy_envelope = [
        (float(sampled_times[i]), float(z[i])) for i in range(len(z))
    ]
    n_high = sum(1 for b in peak_intensities.values() if b == "high")
    n_med = sum(1 for b in peak_intensities.values() if b == "medium")
    n_low = sum(1 for b in peak_intensities.values() if b == "low")
    log.info(
        "[motion_sync] %s: %d peaks (%d high, %d medium, %d low) @ %.1f fps",
        Path(video_path).name, len(peak_times), n_high, n_med, n_low, sample_fps,
    )
    return MotionInfo(
        peak_times=peak_times,
        peak_intensities=peak_intensities,
        energy_envelope=energy_envelope,
        sample_fps=sample_fps,
    )


def is_motion_peak(
    t: float,
    peaks: list[float],
    *,
    tolerance: float = 0.3,
) -> bool:
    """True iff `t` is within `tolerance` seconds of any peak."""
    if not peaks:
        return False
    nearest = min(peaks, key=lambda p: abs(p - t))
    return abs(nearest - t) <= tolerance


def motion_intensity_at(
    t: float,
    info: MotionInfo,
    *,
    tolerance: float = 0.3,
) -> Intensity | None:
    """Bucket of the nearest motion peak within `tolerance`, else None.

    None falls through to the existing downbeat / strength-bucket fallback
    chain in `_pick_entry`.
    """
    peaks = info["peak_times"]
    if not peaks:
        return None
    nearest = min(peaks, key=lambda p: abs(p - t))
    if abs(nearest - t) > tolerance:
        return None
    return info["peak_intensities"].get(nearest)
