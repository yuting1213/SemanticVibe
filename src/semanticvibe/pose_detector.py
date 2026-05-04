"""Per-frame person occupancy masks via MediaPipe Pose.

This module is the v5 entry point for "where is the person at time t?".
The existing `preprocess/mediapipe_pose.py` returns subject *bounding
boxes* per sampled frame; this module returns *boolean masks* with a
safety padding so downstream layout can ask "is this rectangle clear?"
without re-doing geometry.

The mask is canvas-resolution-agnostic: it's stored at the source
video's native resolution, downstream layout scales it to the render
canvas as needed.

Usage:

    masks = detect_person_mask("samples/dance.mp4", sample_fps=2)
    # masks: dict[float_seconds, np.ndarray of bool, shape (H, W)]
    # True = occupied (don't place overlays here)
    # False = free zone

Reuses the existing `_pose_landmarker()` cache so we don't re-load the
MediaPipe model when both this module and the legacy occupancy code run
in the same process.
"""

from __future__ import annotations

import logging
from pathlib import Path

import cv2
import numpy as np

from semanticvibe.preprocess.mediapipe_pose import _pose_landmarker

log = logging.getLogger(__name__)


def detect_person_mask(
    video_path: Path | str,
    *,
    sample_fps: float = 2.0,
    padding_px: int = 30,
    multi_subject_union: bool = True,
) -> dict[float, np.ndarray]:
    """Walk `video_path` at `sample_fps`, return time → boolean occupancy mask.

    Args:
        video_path: video file (anything ffmpeg can decode).
        sample_fps: how many samples per second of source video. Default 2 fps
            is plenty — pose moves smoothly between samples and Stage 4 can
            interpolate. Higher fps just costs render-time RAM.
        padding_px: dilate each subject's bbox by this many px on every side
            so text/decorations don't sit immediately on the silhouette
            edge. The default 30 px is calibrated to human reading comfort
            on a 1080p source.
        multi_subject_union: if True (default), per-frame masks union
            across all detected subjects. If False, only the largest
            subject is kept (use this for subject-isolation behaviour like
            "follow the singer, ignore background dancers").

    Returns:
        dict mapping sampled timestamp (seconds) → bool ndarray of shape
        (H, W) where H, W match the source video's native resolution.
        True means "occupied — keep overlays out".

        Empty dict if the video can't be read.
    """
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if n_frames <= 0:
        cap.release()
        return {}

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    step = max(1, int(round(fps / sample_fps)))
    landmarker, mp = _pose_landmarker()

    masks: dict[float, np.ndarray] = {}
    idx = 0
    while idx < n_frames:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = cap.read()
        if not ok:
            break

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = landmarker.detect(mp_image)

        mask = np.zeros((height, width), dtype=bool)

        if result.pose_landmarks:
            poses = result.pose_landmarks
            if not multi_subject_union and len(poses) > 1:
                # Pick the pose with the largest bbox area.
                def _area(p):
                    xs = [lm.x for lm in p if getattr(lm, "visibility", 1.0) > 0.3]
                    ys = [lm.y for lm in p if getattr(lm, "visibility", 1.0) > 0.3]
                    if not xs or not ys:
                        return 0
                    return (max(xs) - min(xs)) * (max(ys) - min(ys))

                poses = [max(poses, key=_area)]

            for pose in poses:
                xs = [lm.x for lm in pose if getattr(lm, "visibility", 1.0) > 0.3]
                ys = [lm.y for lm in pose if getattr(lm, "visibility", 1.0) > 0.3]
                if not xs or not ys:
                    continue
                x_min = max(0, int(min(xs) * width) - padding_px)
                y_min = max(0, int(min(ys) * height) - padding_px)
                x_max = min(width, int(max(xs) * width) + padding_px)
                y_max = min(height, int(max(ys) * height) + padding_px)
                if x_max > x_min and y_max > y_min:
                    mask[y_min:y_max, x_min:x_max] = True

        masks[idx / fps] = mask
        idx += step

    cap.release()
    log.info(
        "pose_detector: %d sampled frames from %s, mask resolution %dx%d",
        len(masks),
        Path(video_path).name,
        width,
        height,
    )
    return masks


def pick_nearest_mask(
    masks: dict[float, np.ndarray], t: float
) -> np.ndarray | None:
    """Return the mask sampled at the time closest to `t`, or None if empty."""
    if not masks:
        return None
    nearest_t = min(masks.keys(), key=lambda k: abs(k - t))
    return masks[nearest_t]
