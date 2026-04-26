"""MediaPipe pose detection — feeds the layout occupancy map.

mediapipe 0.10+ dropped the legacy `mp.solutions` namespace; we use the
Tasks API (`mp.tasks.vision.PoseLandmarker`). The model file is auto-
downloaded into a per-user cache on first use.

We sample frames at a coarse rate (default 4 fps) since subjects move
smoothly; the layout stage interpolates between samples.
"""

from __future__ import annotations

import logging
import urllib.request
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import cv2

log = logging.getLogger(__name__)

_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
    "pose_landmarker_lite/float16/1/pose_landmarker_lite.task"
)


@dataclass(frozen=True)
class SubjectBox:
    """Pixel bounding box of a detected subject in a single frame."""

    frame_time: float
    x: int
    y: int
    w: int
    h: int


def _model_path() -> Path:
    cache_root = Path.home() / ".cache" / "semanticvibe" / "mediapipe"
    cache_root.mkdir(parents=True, exist_ok=True)
    p = cache_root / "pose_landmarker_lite.task"
    if not p.exists():
        log.info("Downloading MediaPipe pose model to %s", p)
        urllib.request.urlretrieve(_MODEL_URL, p)
    return p


@lru_cache(maxsize=1)
def _pose_landmarker():
    import mediapipe as mp
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision as mp_vision

    base_options = mp_python.BaseOptions(model_asset_path=str(_model_path()))
    options = mp_vision.PoseLandmarkerOptions(
        base_options=base_options,
        running_mode=mp_vision.RunningMode.IMAGE,
        num_poses=2,
        min_pose_detection_confidence=0.5,
    )
    return mp_vision.PoseLandmarker.create_from_options(options), mp


def detect_subjects(video_path: Path, *, sample_fps: float = 4.0) -> list[SubjectBox]:
    """Detect human subjects across `video_path`, sampled at `sample_fps`.

    Returns one SubjectBox per detected pose per sampled frame. Frames with no
    subject yield no output (rather than a None entry) so the layout occupancy
    builder can simply union all returned boxes.
    """
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if n_frames <= 0:
        cap.release()
        return []

    step = max(1, int(round(fps / sample_fps)))
    landmarker, mp = _pose_landmarker()

    out: list[SubjectBox] = []
    idx = 0
    while idx < n_frames:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = cap.read()
        if not ok:
            break

        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = landmarker.detect(mp_image)

        for pose in result.pose_landmarks or []:
            xs = [lm.x for lm in pose if getattr(lm, "visibility", 1.0) > 0.3]
            ys = [lm.y for lm in pose if getattr(lm, "visibility", 1.0) > 0.3]
            if xs and ys:
                x_min = max(0, int(min(xs) * w))
                y_min = max(0, int(min(ys) * h))
                x_max = min(w, int(max(xs) * w))
                y_max = min(h, int(max(ys) * h))
                if x_max > x_min and y_max > y_min:
                    out.append(
                        SubjectBox(
                            frame_time=idx / fps,
                            x=x_min,
                            y=y_min,
                            w=x_max - x_min,
                            h=y_max - y_min,
                        )
                    )
        idx += step

    cap.release()
    return out
